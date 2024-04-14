from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

from collections import deque

from torch import nn
from tqdm import tqdm

from src.agents.agents import Agent
from src.algorithms.algorithm import TrainingAlgorithm
from src.data.trajectory import TrajectoryBatch
from src.envs.eg_vectorized import ObligatedRatioDiscreteEG
from src.utils.utils import (
    compute_discounted_returns,
    cat_observations,
    compute_gae_advantages,
    get_categorical_entropy,
    get_cosine_similarity_torch,
    get_gaussian_entropy,
    get_observation, 
    torchify_actions
)

class AdvantageAlignment(TrainingAlgorithm):

    def get_total_entropy(self, term_dist, agent):
        term_probs = term_dist.probs.reshape((-1, term_dist.probs.size(2)))
        utte_stds = torch.exp(
            agent.actor.utte_log_std
        ).reshape((-1, agent.actor.utte_log_std.size(2)))
        prop_stds = torch.exp(
            agent.actor.prop_log_std
        ).reshape((-1, agent.actor.prop_log_std.size(2)))

        term_ent = get_categorical_entropy(term_probs)
        utte_ent = get_gaussian_entropy(torch.diag_embed(utte_stds))
        prop_ent = get_gaussian_entropy(torch.diag_embed(prop_stds))

        return term_ent, utte_ent, prop_ent

    def get_total_entropy_discrete(self, term_dist, prop_dist):
        prop_dist_1, prop_dist_2, prop_dist_3 = prop_dist
        term_ent = torch.tensor(0)
        utte_ent = torch.tensor(0)
        prop_ent = (
            prop_dist_1.entropy() +
            prop_dist_2.entropy() +
            prop_dist_3.entropy()
        ).mean()

        return term_ent, utte_ent, prop_ent

    def get_f1_entropy(self, agent):
        acc_stds = torch.exp(agent.actor.acc_log_std.squeeze(0))
        ste_stds = torch.exp(agent.actor.ste_log_std.squeeze(0))

        acc_ent = get_gaussian_entropy(torch.diag_embed(acc_stds))
        ste_ent = get_gaussian_entropy(torch.diag_embed(ste_stds))

        return acc_ent, ste_ent

    def get_entropy(self, term_dist, agent):
        term_probs = term_dist.probs.squeeze(0)
        utte_stds = torch.exp(agent.actor.utte_log_std.squeeze(0))
        prop_stds = torch.exp(agent.actor.prop_log_std.squeeze(0))

        term_ent = get_categorical_entropy(term_probs)
        utte_ent = get_gaussian_entropy(torch.diag_embed(utte_stds))
        prop_ent = get_gaussian_entropy(torch.diag_embed(prop_stds))

        return term_ent, utte_ent, prop_ent

    def get_entropy_discrete(self, term_dist, prop_dist):
        prop_dist_1, prop_dist_2, prop_dist_3 = prop_dist

        term_ent = torch.tensor(0)
        #TODO: Add utterance and term support
        utte_ent = torch.tensor(0)
        prop_ent = (
            prop_dist_1.entropy() +
            prop_dist_2.entropy() +
            prop_dist_3.entropy()
        ).mean()

        return term_ent, utte_ent, prop_ent

    def get_f1_trajectory_log_ps(self, agent, trajectory, is_first):
        log_p_list = []
        if is_first:
            observation = trajectory.data['obs_0']
            actions = trajectory.data['actions_0']
        else:
            observation = trajectory.data['obs_1']
            actions = trajectory.data['actions_1']
        
        acc_ent = torch.zeros(
            (self.trajectory_len), 
            device=self.agent_1.device, 
            dtype=torch.float32
        )
        ste_ent = torch.zeros(
            (self.trajectory_len), 
            device=self.agent_1.device, 
            dtype=torch.float32
        )
        hidden = None
        for t in range(self.trajectory_len):
            hidden, acc_dist, ste_dist = agent.actor(
                x=observation[:, t, :],
                h_0=hidden
            )
            hidden = hidden.permute((1, 0, 2))
            acc_t = actions[:, t, 0].unsqueeze(1)
            ste_t = actions[:, t, 1].unsqueeze(1)
            log_p = acc_dist.log_prob(acc_t).squeeze(1) + ste_dist.log_prob(ste_t).squeeze(1)

            acc_en, ste_en = self.get_f1_entropy(agent)
            acc_ent[t] = acc_en
            ste_ent[t] = ste_en

            log_p_list.append(log_p)
        log_ps = torch.stack(log_p_list).permute((1, 0))
        acc_ent = acc_ent.mean()
        ste_ent = ste_ent.mean()
        
        return log_ps, acc_ent, ste_ent



    def get_trajectory_log_ps(self, agent, trajectory, is_first):
        log_p_list = []
        if self.discrete:
            prop_max = 1
        else:
            prop_max = self.agent_1.actor.prop_max

        if is_first:
            terms = trajectory.data['terms_0']
            a_props = trajectory.data['a_props_0']
            props = trajectory.data['props_0']
            item_and_utility_1 = trajectory.data['i_and_u_0']
            item_and_utility_2 = trajectory.data['i_and_u_1']
        else:
            terms = trajectory.data['terms_1']
            a_props = trajectory.data['a_props_1']
            props = trajectory.data['props_1']
            item_and_utility_1 = trajectory.data['i_and_u_1']
            item_and_utility_2 = trajectory.data['i_and_u_0']

        if self.use_transformer:
            props = props.permute((1, 0, 2))

        item_and_utility = torch.cat([item_and_utility_1, item_and_utility_2], dim=-1)

        observation = {'prop': props, 'item_and_utility': item_and_utility}

        if self.use_transformer:
            _, term_dist, utte_dist, prop_dist = agent.actor(
                x=observation,
                use_transformer=self.use_transformer
            )

            if self.discrete:
                # log_ps_term = term_dist.log_prob(terms)
                log_ps_term = 0
                log_ps_utte = 0
                prop_cat_1, prop_cat_2, prop_cat_3 = prop_dist
                log_ps_prop = (
                    prop_cat_1.log_prob(a_props[:, :, 0]) + 
                    prop_cat_2.log_prob(a_props[:, :, 1]) + 
                    prop_cat_3.log_prob(a_props[:, :, 2])
                )

                term_ent, utte_ent, prop_ent = self.get_total_entropy_discrete(term_dist, prop_dist)
            else:
                log_ps_term = term_dist.log_prob(terms)
                log_ps_utte = utte_dist.log_prob(uttes).sum(dim=-1)
                log_ps_prop = prop_dist.log_prob(props/prop_max).sum(dim=-1)

                term_ent, utte_ent, prop_ent = self.get_total_entropy(term_dist, agent)

            log_ps = log_ps_term + log_ps_utte + log_ps_prop
        else:
            term_ent = torch.zeros(
                (self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            utte_ent = torch.zeros(
                (self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            prop_ent = torch.zeros(
                (self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )

            hidden = None
            for t in range(self.trajectory_len):
                hidden, term_dist, utte_dist, prop_dist = agent.actor(
                    x=get_observation(observation, t),
                    h_0=hidden
                )
                hidden = hidden.permute((1, 0, 2))
                
                if self.discrete:
                    term_en, utte_en, prop_en = self.get_entropy_discrete(term_dist, prop_dist)
                    log_p_term = 0
                    log_p_utte = 0
                    prop_cat_1, prop_cat_2, prop_cat_3 = prop_dist
                    log_p_prop = (
                        prop_cat_1.log_prob(a_props[:, t, 0]) + 
                        prop_cat_2.log_prob(a_props[:, t, 1]) + 
                        prop_cat_3.log_prob(a_props[:, t, 2])
                    ).squeeze(0) 
                else:
                    term_en, utte_en, prop_en = self.get_entropy(term_dist, agent)
                    log_p_term = term_dist.log_prob(terms[:, t]).squeeze(0)
                    log_p_utte = utte_dist.log_prob(uttes[:, t]).sum(dim=-1)
                    log_p_prop = prop_dist.log_prob(props[:, t]/prop_max).sum(dim=-1)
                term_ent[t] = term_en
                utte_ent[t] = utte_en
                prop_ent[t] = prop_en
                

                log_p = log_p_term + log_p_utte + log_p_prop
                log_p_list.append(log_p)
                log_ps = torch.stack(log_p_list).permute((1, 0))
            term_ent = term_ent.mean()
            utte_ent = utte_ent.mean()
            prop_ent = prop_ent.mean()

        return log_ps, term_ent, utte_ent, prop_ent

    def get_trajectory_values(self, agent, observation):
        if self.use_transformer:
            values_c = agent.critic(
                x=observation, 
                partial_forward=False
            )[1].permute((1, 0, 2)).squeeze(-1)
            values_t = agent.target(
                x=observation, 
                partial_forward=False
            )[1].permute((1, 0, 2)).squeeze(-1)
        else:
            values_c = torch.zeros(
                (self.batch_size, self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            values_t = torch.zeros(
                (self.batch_size, self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            hidden_c, hidden_t = None, None
            for t in range(self.trajectory_len):
                if self.is_f1:
                    obs = observation[:, t, :]
                else:
                    obs = get_observation(observation, t)
                hidden_c, value_c = agent.critic(obs, h_0=hidden_c)
                hidden_t, value_t = agent.target(obs, h_0=hidden_t)
                values_c[:, t] = value_c.reshape(self.batch_size)
                values_t[:, t] = value_t.reshape(self.batch_size)
                hidden_c = hidden_c.permute((1, 0, 2))
                hidden_t = hidden_t.permute((1, 0, 2))
        return values_c, values_t

    def get_trajectory_rewards(self, agent, observation):
        if self.use_transformer:
            rewards = agent.reward(
                x=observation.permute((1, 0, 2)), 
                partial_forward=False
            )[1].permute((1, 0, 2)).squeeze(-1)
        else:
            rewards = torch.empty(
                (self.batch_size, self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            hidden = None
            for t in range(self.trajectory_len):
                hidden, reward = agent.reward(observation[:, t, :])
                rewards[:, t] = reward.reshape(self.batch_size)
                hidden = hidden.permute((1, 0, 2))
        return rewards

    def get_trajectory_embeds(self, agent, observation, hidden_dim):
        if self.use_transformer:
            embeds = agent.spr(
                x=observation.permute((1, 0, 2)), 
                partial_forward=False
            )[1].permute((1, 0, 2)).squeeze(-1)
        else:
            embeds = torch.empty(
                (self.batch_size, self.trajectory_len, hidden_dim), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            hidden = None
            for t in range(self.trajectory_len):
                hidden, embed = agent.spr(observation[:, t, :])
                embeds[:, t, :] = embed.reshape(self.batch_size, -1)
                hidden = hidden.permute((1, 0, 2))
        return embeds

    def linear_aa_loss(self, A_1s, A_2s, log_ps, old_log_ps):
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        A_1s = A_1s[:, :-1]
        A_2s = A_2s[:, 1:]

        if proximal:
            # gamma = 1
            ratios = torch.exp(log_ps - old_log_ps.detach())
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

    def dot_product_aa_loss(self, A_1s, A_2s, log_ps, old_log_ps):
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range
        device = log_ps.device

        A_1s = A_1s[:, :-1]
        A_2s = A_2s[:, 1:]

        if proximal:
            # gamma = 1
            ratios = torch.exp(log_ps - old_log_ps.detach())
            action_term = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
        else:
            action_term = torch.exp(log_ps - old_log_ps.detach())
        
        acc_surrogates = torch.zeros_like(action_term)

        for t in range(0, action_term.size(1) - 1):
            A_1_t = A_1s[:, :t+1]
            A_2_t = A_2s[:, t].unsqueeze(1) * torch.ones(action_term.size(0), t+1).to(device)
            adv_alignment = get_cosine_similarity_torch(A_1_t, A_2_t)
            acc_surrogates[:, t] = torch.min(adv_alignment * action_term[:, t], adv_alignment * log_ps[:, t])   
        
        return acc_surrogates.mean()
    
    def actor_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        actor_loss_dict = {}
        gamma = self.train_cfg.gamma
        vanilla = self.train_cfg.vanilla
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        if is_first:
            agent = self.agent_1
            other_agent = self.agent_2
            props_1 = trajectory.data['props_0']
            item_and_utility_1 = trajectory.data['i_and_u_0']
            props_2 = trajectory.data['props_1']
            item_and_utility_2 = trajectory.data['i_and_u_1']
            observation_1 = trajectory.data['obs_0']
            rewards_1 = trajectory.data['rewards_0']
            observation_2 = trajectory.data['obs_1']
            rewards_2 = trajectory.data['rewards_1']
            old_log_ps = trajectory.data['log_ps_0']
        else:
            agent = self.agent_2
            other_agent = self.agent_1
            props_1 = trajectory.data['props_1']
            item_and_utility_1 = trajectory.data['i_and_u_1']
            props_2 = trajectory.data['props_0']
            item_and_utility_2 = trajectory.data['i_and_u_0']
            observation_1 = trajectory.data['obs_1']
            rewards_1 = trajectory.data['rewards_1']
            observation_2 = trajectory.data['obs_0']
            rewards_2 = trajectory.data['rewards_0']
            old_log_ps = trajectory.data['log_ps_1']

        if self.use_transformer:
            props_1 = props_1.permute((1, 0, 2))
            props_2 = props_2.permute((1, 0, 2))

        if not self.is_f1:
            item_and_utility_1_cat = torch.cat([item_and_utility_1, item_and_utility_2], dim=-1)
            item_and_utility_2_cat = torch.cat([item_and_utility_2, item_and_utility_1], dim=-1)

            observation_1 = {'prop': props_1, 'item_and_utility': item_and_utility_1_cat}
            observation_2 = {'prop': props_2, 'item_and_utility': item_and_utility_2_cat}

        if self.sum_rewards:
            rewards_1 = rewards_2 = rewards_1 + rewards_2
        
        _, V_1s = self.get_trajectory_values(agent, observation_1)
        _, V_2s = self.get_trajectory_values(other_agent, observation_2)

        if self.is_f1:
            log_ps, acc_ent, ste_ent = self.get_f1_trajectory_log_ps(agent, trajectory, is_first)
            actor_loss_dict['acc_ent'] = acc_ent
            actor_loss_dict['ste_ent'] = ste_ent
        else:
            log_ps, term_ent, utte_ent, prop_ent = self.get_trajectory_log_ps(agent, trajectory, is_first)
            actor_loss_dict['term_ent'] = term_ent
            actor_loss_dict['utte_ent'] = utte_ent
            actor_loss_dict['prop_ent'] = prop_ent

        A_1s = compute_gae_advantages(rewards_1, V_1s.detach(), gamma)
        A_2s = compute_gae_advantages(rewards_2, V_2s.detach(), gamma)
        
        gammas = torch.pow(
            gamma, 
            torch.arange(self.trajectory_len)
        ).repeat(self.batch_size, 1).to(A_1s.device)
        
        if proximal:
            ratios = torch.exp(log_ps - old_log_ps.detach())
            clipped_log_ps = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            surrogates = torch.min(A_1s * clipped_log_ps[:, :-1], A_1s * log_ps[:, :-1])
            loss_1 = -surrogates.mean()
        else:
            # loss_1 = -(gammas[:, :-1] * A_1s * log_ps[:, :-1]).mean()
           loss_1 = -(A_1s * log_ps[:, :-1]).mean()

        if not vanilla:
            loss_2 = -self.dot_product_aa_loss(A_1s, A_2s, log_ps[:, :-1], old_log_ps[:, :-1])
        else:
            loss_2 = torch.zeros(1, device=loss_1.device)
        
        actor_loss_dict['loss'] = loss_1 + self.train_cfg.aa_weight * loss_2
        return actor_loss_dict


    def critic_loss(self,
                    trajectory: TrajectoryBatch,
                    is_first: bool,
                    ) -> float:
        gamma = self.train_cfg.gamma

        if is_first:
            agent = self.agent_1
            props = trajectory.data['props_0']
            observation = trajectory.data['obs_0']
            item_and_utility_1 = trajectory.data['i_and_u_0']
            item_and_utility_2 = trajectory.data['i_and_u_1']
            rewards_1 = trajectory.data['rewards_0']
            rewards_2 = trajectory.data['rewards_1']
        else:
            agent = self.agent_2
            props = trajectory.data['props_1']
            observation = trajectory.data['obs_1']
            item_and_utility_1 = trajectory.data['i_and_u_1']
            item_and_utility_2 = trajectory.data['i_and_u_0']
            rewards_1 = trajectory.data['rewards_1']
            rewards_2 = trajectory.data['rewards_0']

        if not self.is_f1:
            if self.use_transformer:
                props= props.permute((1, 0, 2))
            
            item_and_utility = torch.cat([item_and_utility_1, item_and_utility_2], dim=-1)

            observation = {
                'prop': props,
                'item_and_utility': item_and_utility
            }

        if self.sum_rewards:
            rewards_1 = rewards_2 = rewards_1 + rewards_2

        values_c, values_t = self.get_trajectory_values(agent, observation)
        if self.train_cfg.critic_loss_mode == 'td-1':
            values_c_shifted = values_c[:, :-1]
            values_t_shifted = values_t[:, 1:]
            rewards_shifted = rewards_1[:, :-1]
            critic_loss = F.huber_loss(values_c_shifted, rewards_shifted + gamma * values_t_shifted)
        elif self.train_cfg.critic_loss_mode == 'MC':
            discounted_returns = compute_discounted_returns(gamma, rewards_1)
            critic_loss = F.huber_loss(values_c, discounted_returns)

        return critic_loss

    def reward_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        if is_first:
            agent = self.agent_1
            props = trajectory.data['props_0']
            rewards_1 = trajectory.data['rewards_0']
        else:
            agent = self.agent_2
            props = trajectory.data['props_1']
            rewards_1 = trajectory.data['rewards_1']

        reward_preds = self.get_trajectory_rewards(agent, props)
        reward_preds = reward_preds[:, :-1]
        rewards_shifted = rewards_1[:, 1:]
        
        critic_loss = F.huber_loss(reward_preds, rewards_shifted)
        return critic_loss

    def spr_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        if is_first:
            agent = self.agent_1
            props = trajectory.data['props_0']
        else:
            agent = self.agent_2
            props = trajectory.data['props_1']

        hidden_dim = agent.spr.in_size

        target_y = agent.actor.model.encoder.get_embeds(props).detach()
        target_y = target_y[:, 1:, :].reshape(-1, hidden_dim)
        y_preds = self.get_trajectory_embeds(agent, props, hidden_dim)
        y_preds = y_preds[:, :-1, :].reshape(-1, hidden_dim)

        norms_t = torch.norm(target_y, dim=1, keepdim=True) + 1e-8
        norms_p = torch.norm(y_preds, dim=1, keepdim=True) + 1e-8

        normalized_t = target_y/norms_t
        normalized_p = y_preds/norms_p

        spr_loss = -torch.mean(torch.bmm(normalized_t.unsqueeze(1), normalized_p.unsqueeze(2)))
        return spr_loss

    def gen_sim(self):
        observations, _ = self.env.reset()
        observations_1, observations_2 = observations

        item_and_utility_1 = torch.cat(
            [
                observations_1['item_and_utility'],
                observations_2['item_and_utility']
            ], dim=1
        )
        item_and_utility_2 = torch.cat(
            [
                observations_2['item_and_utility'],
                observations_1['item_and_utility']
            ], dim=1
        )
        observations_1['item_and_utility'] = item_and_utility_1
        observations_2['item_and_utility'] = item_and_utility_2

        hidden_state_1, hidden_state_2 = None, None
        history = None
        trajectory = TrajectoryBatch(
            batch_size=self.batch_size , 
            max_traj_len=self.trajectory_len,
            device=self.agent_1.device
        )

        if self.use_transformer:
            history = deque(maxlen=self.agent_1.actor.model.encoder.max_seq_len)

        for t in range(self.trajectory_len):
            hidden_state_1, action_1, log_p_1 = self.agent_1.sample_action(
                observations=observations_1,
                h_0=hidden_state_1,
                history=history,
                use_transformer=self.use_transformer,
                extend_history=True,
                is_first=True,
            )
            hidden_state_2, action_2, log_p_2 = self.agent_2.sample_action(
                observations=observations_2,
                h_0=hidden_state_2,
                history=history,
                use_transformer=self.use_transformer,
                extend_history=False,
                is_first=False,
            )
            observations, rewards, done, _, info = self.env.step((action_1, action_2))
            observations_1, observations_2 = observations

            trajectory.add_step_sim(
                action=(action_1, action_2), 
                observations=(
                    observations_1,
                    observations_2
                ), 
                rewards=rewards,
                log_ps=(log_p_1, log_p_2),
                done=done,
                info=info,
                t=t
            )

            item_and_utility_1 = torch.cat(
                [
                    observations_1['item_and_utility'],
                    observations_2['item_and_utility']
                ], dim=1
            )
            item_and_utility_2 = torch.cat(
                [
                    observations_2['item_and_utility'],
                    observations_1['item_and_utility']
                ], dim=1
            )
            observations_1['item_and_utility'] = item_and_utility_1
            observations_2['item_and_utility'] = item_and_utility_2
        
        return trajectory
    
    def gen_f1(self):
        device = self.agent_1.device
        observations, _ = self.env.reset()
        observations_1, observations_2 = observations

        hidden_state_1, hidden_state_2 = None, None
        history = None
        trajectory = TrajectoryBatch(
            batch_size=self.batch_size , 
            max_traj_len=self.trajectory_len,
            device=device
        )

        if self.use_transformer:
            history = deque(maxlen=self.agent_1.actor.model.encoder.max_seq_len)

        observations_1 = torch.tensor(observations_1.reshape((self.batch_size, -1))).to(device)
        observations_2 = torch.tensor(observations_2.reshape((self.batch_size, -1))).to(device)

        for t in range(self.trajectory_len):
            hidden_state_1, action_1, log_p_1 = self.agent_1.sample_f1_action(
                observations=observations_1,
                h_0=hidden_state_1,
                history=history,
                use_transformer=self.use_transformer,
                extend_history=True,
                is_first=True,
            )
            hidden_state_2, action_2, log_p_2 = self.agent_2.sample_f1_action(
                observations=observations_2,
                h_0=hidden_state_2,
                history=history,
                use_transformer=self.use_transformer,
                extend_history=False,
                is_first=False,
            )
            observations, rewards, done, _, info = self.env.step((action_1, action_2))
            observations_1, observations_2 = observations
            observations_1 = torch.tensor(observations_1.reshape((self.batch_size, -1))).to(device)
            observations_2 = torch.tensor(observations_2.reshape((self.batch_size, -1))).to(device)

            trajectory.add_step_f1(
                action=(torch.tensor(action_1).to(device), torch.tensor(action_2).to(device)), 
                observations=(
                    observations_1,
                    observations_2
                ), 
                rewards=torch.tensor(rewards).to(device),
                log_ps=(log_p_1, log_p_2),
                done=torch.tensor(done).to(device),
                info=info,
                t=t
            )
        return trajectory

    def eval(self, agent):
        # todo: assert eg, check the code, etc
        assert  self.env.name == 'eg_obligated_ratio'
        # evaluate against always cooperate and always defect
        always_cooperate_agent = AlwaysCooperateAgent(device=agent.device)
        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len, agent_1=agent, agent_2=always_cooperate_agent, use_transformer=self.use_transformer)
        ac_rewards_agent = trajectory.data['rewards_0'].mean().item()
        ac_rewards_ac = trajectory.data['rewards_1'].mean().item()
        always_defect_agent = AlwaysDefectAgent(device=agent.device)
        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len, agent_1=agent, agent_2=always_defect_agent, use_transformer=self.use_transformer)
        ad_rewards_agent = trajectory.data['rewards_0'].mean().item()
        ad_rewards_ad = trajectory.data['rewards_1'].mean().item()
        metrics = {
            'ac_rewards_agent': ac_rewards_agent,
            'ac_rewards_ac': ac_rewards_ac,
            'ad_rewards_agent': ad_rewards_agent,
            'ad_rewards_ad': ad_rewards_ad
        }
        return metrics



class AlwaysCooperateAgent(Agent):
    def __init__(self, device):
        self.device = device

    def sample_action(self,
                      observations,
                      is_first: bool,
                      *args,
                      **kwargs):

        item_and_utility = observations['item_and_utility']
        assert item_and_utility.shape[1] == 12
        # this is the format: [number of  item 1, ..., number of item 3, utility item 1 for me, ..., utility item 3 for me, utility item 1 for you, ..., utility item 3 for you]
        # grab the utilities for me
        my_utilities = item_and_utility[:, 3:6]
        your_utilities = item_and_utility[:, 9:12]
        item_counts = item_and_utility[:, 0:3]
        assert torch.allclose(item_counts, item_and_utility[:, 6:9])  # check that the counts are the same, as they should be, they are copy paste of each other

        ans = torch.where(my_utilities >= your_utilities, 1, 0).to(self.device)
        log_p = torch.tensor(0.).to(self.device)

        return None, {'prop': ans}, log_p


class AlwaysDefectAgent(Agent):
    def __init__(self, device):
        self.device = device

    def sample_action(self,
                      observations,
                      is_first: bool,
                        *args,
                        **kwargs):
        item_and_utility = observations['item_and_utility']
        assert item_and_utility.shape[1] == 12
        # this is the format: [number of  item 1, ..., number of item 3, utility item 1 for me, ..., utility item 3 for me, utility item 1 for you, ..., utility item 3 for you]
        # grab the utilities for me
        my_utilities = item_and_utility[:, 3:6]
        your_utilities = item_and_utility[:, 9:12]
        item_counts = item_and_utility[:, 0:3]
        assert torch.allclose(item_counts, item_and_utility[:, 6:9])  # check that the counts are the same, as they should be, they are copy paste of each other

        ans = torch.ones_like(my_utilities).to(self.device)
        log_p = torch.tensor(0.).to(self.device)

        return None, {'prop': ans}, log_p


def gen_episodes_between_agents(env, batch_size, trajectory_len,  agent_1, agent_2, use_transformer=False):
    observations, _ = env.reset()
    observations_1, observations_2 = observations

    item_and_utility_1 = torch.cat(
        [
            observations_1['item_and_utility'],
            observations_2['item_and_utility']
        ], dim=1
    )
    item_and_utility_2 = torch.cat(
        [
            observations_2['item_and_utility'],
            observations_1['item_and_utility']
        ], dim=1
    )
    observations_1['item_and_utility'] = item_and_utility_1
    observations_2['item_and_utility'] = item_and_utility_2

    hidden_state_1, hidden_state_2 = None, None
    history = None
    trajectory = TrajectoryBatch(
        batch_size=batch_size,
        max_traj_len=trajectory_len,
        device=agent_1.device
    )

    if use_transformer:
        history = deque(maxlen=agent_1.actor.model.encoder.max_seq_len)

    for t in range(trajectory_len):
        hidden_state_1, action_1, log_p_1 = agent_1.sample_action(
            observations=observations_1,
            h_0=hidden_state_1,
            history=history,
            use_transformer=use_transformer,
            extend_history=True,
            is_first=True,
        )
        hidden_state_2, action_2, log_p_2 = agent_2.sample_action(
            observations=observations_2,
            h_0=hidden_state_2,
            history=history,
            use_transformer=use_transformer,
            extend_history=False,
            is_first=False,
        )
        observations, rewards, done, _, info = env.step((action_1, action_2))
        observations_1, observations_2 = observations

        trajectory.add_step_sim(
            action=(action_1, action_2),
            observations=(
                observations_1,
                observations_2
            ),
            rewards=rewards,
            log_ps=(log_p_1, log_p_2),
            done=done,
            info=info,
            t=t
        )

        item_and_utility_1 = torch.cat(
            [
                observations_1['item_and_utility'],
                observations_2['item_and_utility']
            ], dim=1
        )
        item_and_utility_2 = torch.cat(
            [
                observations_2['item_and_utility'],
                observations_1['item_and_utility']
            ], dim=1
        )
        observations_1['item_and_utility'] = item_and_utility_1
        observations_2['item_and_utility'] = item_and_utility_2

    return trajectory


if __name__ == '__main__':
    batch_size=17
    env = ObligatedRatioDiscreteEG(batch_size=batch_size, device='cpu')
    # agent_1 = AlwaysCooperateAgent('cpu')
    # agent_2 = AlwaysCooperateAgent('cpu')
    agent_1 = AlwaysDefectAgent('cpu')
    agent_2 = AlwaysDefectAgent('cpu')
    trajectory = gen_episodes_between_agents(env, batch_size, 10, agent_1, agent_2)
    print(trajectory.data.keys())
    # print rewards
    print(trajectory.data['rewards_0'])
    print(trajectory.data['rewards_1'])
    print((trajectory.data['rewards_0'] + trajectory.data['rewards_1']).mean())
    # print actions
    print(trajectory.data['a_props_0'][-1])
    print(trajectory.data['a_props_1'][-1])



