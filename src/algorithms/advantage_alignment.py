from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from src.algorithms.algorithm import TrainingAlgorithm
from src.data.trajectory import TrajectoryBatch
from src.utils.utils import cat_observations, torchify_actions, compute_gae_advantages

class AdvantageAlignment(TrainingAlgorithm):

    def get_trajectory_log_ps(self, agent, trajectory, is_first):
        log_ps = []
        prop_max = self.agent_1.actor.prop_max

        if is_first:
            observation = trajectory.data['obs_0']
            terms = trajectory.data['terms_0']
            props = trajectory.data['props_0']
            uttes = trajectory.data['uttes_0']
        else:
            observation = trajectory.data['obs_1']
            terms = trajectory.data['terms_1']
            props = trajectory.data['props_1']
            uttes = trajectory.data['uttes_1']

        hidden = None
        for t in range(self.trajectory_len):
            hidden, term_dist, utte_dist, prop_dist = agent.actor(
                x=observation[:, t, :],
                h_0=hidden
            )
            hidden = hidden.permute((1, 0, 2))
            
            log_p_term = term_dist.log_prob(terms[:, t]).squeeze(0)
            log_p_utte = utte_dist.log_prob(uttes[:, t]).sum(dim=-1)
            log_p_prop = prop_dist.log_prob(props[:, t]/prop_max).sum(dim=-1)

            log_p = log_p_term + log_p_utte + log_p_prop
            log_ps.append(log_p)
        return torch.stack(log_ps).permute((1, 0))

    def get_trajectory_values(self, agent, observation):
        values_c = torch.empty(
            (self.batch_size, self.trajectory_len), 
            device=self.agent_1.device, 
            dtype=torch.float32
        )
        values_t = torch.empty(
            (self.batch_size, self.trajectory_len), 
            device=self.agent_1.device, 
            dtype=torch.float32
        )
        hidden_c, hidden_t = None, None
        for t in range(self.trajectory_len):
            hidden_c, value_c = agent.critic(observation[:, t, :])
            hidden_t, value_t = agent.target(observation[:, t, :])
            values_c[:, t] = value_c.reshape(self.batch_size)
            values_t[:, t] = value_t.reshape(self.batch_size)
            hidden_c = hidden_c.permute((1, 0, 2))
            hidden_t = hidden_t.permute((1, 0, 2))
        return values_c, values_t

    def linear_aa_loss(self, A_1s, A_2s, log_ps):
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        A_1s = A_1s[:, :-1]
        A_2s = A_2s[:, 1:]

        if proximal:
            gamma = 1
            ratios = torch.exp(log_ps - log_ps.detach())
            action_term = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
        else:
            action_term = log_ps
        
        acc_surrogates = torch.empty_like(action_term)
        carry = torch.zeros_like(action_term[:, 0])

        for t in reversed(range(action_term.size(1) - 1)):
            carry = A_2s[:, t] * action_term[:, t] + gamma * carry
            acc_surrogates[:, t] = A_1s[:, t] * carry

        if torch.any(torch.isnan(acc_surrogates)):
            acc_surrogates = torch.nan_to_num(acc_surrogates)
        
        return acc_surrogates.mean()
    
    def actor_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        if is_first:
            agent = self.agent_1
            other_agent = self.agent_2
            observation_1 = trajectory.data['obs_0']
            rewards_1 = trajectory.data['rewards_0']
            observation_2 = trajectory.data['obs_1']
            rewards_2 = trajectory.data['rewards_1']
        else:
            agent = self.agent_2
            other_agent = self.agent_1
            observation_1 = trajectory.data['obs_1']
            rewards_1 = trajectory.data['rewards_1']
            observation_2 = trajectory.data['obs_0']
            rewards_2 = trajectory.data['rewards_0']
        
        _, V_1s = self.get_trajectory_values(agent, observation_1)
        _, V_2s = self.get_trajectory_values(other_agent, observation_2)

        log_ps = self.get_trajectory_log_ps(agent, trajectory, is_first)
        
        A_1s = compute_gae_advantages(rewards_1, V_1s, gamma)
        A_2s = compute_gae_advantages(rewards_2, V_2s, gamma)
        
        gammas = torch.pow(
            gamma, 
            torch.arange(self.trajectory_len)
        ).repeat(self.batch_size, 1)
        
        if proximal:
            ratios = torch.exp(log_ps - log_ps.detach())
            surrogates = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            
            loss_1 = (A_1s * surrogates[:, :-1]).mean()
        else:
            loss_1 = (gammas[:, :-1] * A_1s * log_ps[:, :-1]).mean()

        loss_2 = self.linear_aa_loss(A_1s, A_2s, log_ps[:, :-1])
        
        return -1*(loss_1 + loss_2)


    def critic_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        gamma = self.train_cfg.gamma

        if is_first:
            agent = self.agent_1
            observation = trajectory.data['obs_0']
            rewards = trajectory.data['rewards_0']
        else:
            agent = self.agent_2
            observation = trajectory.data['obs_1']
            rewards = trajectory.data['rewards_1']

        values_c, values_t = self.get_trajectory_values(agent, observation)
        values_c_shifted = values_c[:, :-1]
        values_t_shifted = values_t[:, 1:]
        rewards_shifted = rewards[:, :-1]
        
        critic_loss = F.huber_loss(values_c_shifted, rewards_shifted + gamma * values_t_shifted)
        return critic_loss
        
    def gen(self):
        observations, _ = self.env.reset()
        hidden_state_1, hidden_state_2 = None, None
        trajectory = TrajectoryBatch(
            batch_size=self.batch_size , 
            max_traj_len=self.trajectory_len,
            device=self.agent_1.device
        )

        for t in range(self.trajectory_len * 2):
            if t % 2 == 0:
                hidden_state_1, action, log_p_1 = self.agent_1.sample_action(
                    observations=observations,
                    h_0=hidden_state_1
                )
            else:
                hidden_state_2, action, log_p_2 = self.agent_2.sample_action(
                    observations=observations,
                    h_0=hidden_state_2
                )
            observations, rewards, _, _, _ = self.env.step(action)
            trajectory.add_step(
                action=torchify_actions(action, self.agent_1.device), 
                observations=cat_observations(observations, self.agent_1.device), 
                rewards=torch.tensor(rewards, device=self.agent_1.device), 
                t=t
            )
        trajectory.add_step(
            action=torchify_actions(action, self.agent_1.device), 
            observations=cat_observations(observations, self.agent_1.device), 
            rewards=torch.tensor(rewards, device=self.agent_1.device), 
            t=t
        )
        return trajectory