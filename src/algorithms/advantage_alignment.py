from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from collections import deque

from torch import nn
from tqdm import tqdm

from src.algorithms.algorithm import TrainingAlgorithm
from src.data.trajectory import TrajectoryBatch
from src.utils.utils import (
    cat_observations,
    torchify_actions,
    compute_gae_advantages,
    get_categorical_entropy,
    get_gaussian_entropy,
    get_observation
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

    def get_trajectory_log_ps(self, agent, trajectory, is_first):
        log_p_list = []
        if self.discrete:
            prop_max = 1
        else:
            prop_max = self.agent_1.actor.prop_max

        if is_first:
            # observation = trajectory.data['obs_0']
            terms = trajectory.data['terms_0']
            a_props = trajectory.data['a_props_0']
            uttes = trajectory.data['uttes_0']
            props = trajectory.data['props_0']
            item_and_utility_1 = trajectory.data['i_and_u_0']
            item_and_utility_2 = trajectory.data['i_and_u_1']
        else:
            # observation = trajectory.data['obs_1']
            terms = trajectory.data['terms_1']
            a_props = trajectory.data['a_props_1']
            uttes = trajectory.data['uttes_1']
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
            term_ent = torch.empty(
                (self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            utte_ent = torch.empty(
                (self.trajectory_len), 
                device=self.agent_1.device, 
                dtype=torch.float32
            )
            prop_ent = torch.empty(
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
                hidden_c, value_c = agent.critic(get_observation(observation, t))
                hidden_t, value_t = agent.target(get_observation(observation, t))
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
            gamma = 1
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
    
    def actor_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
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
            # observation_1 = trajectory.data['obs_0']
            rewards_1 = trajectory.data['rewards_0']
            # observation_2 = trajectory.data['obs_1']
            rewards_2 = trajectory.data['rewards_1']
            old_log_ps = trajectory.data['log_ps_0']
        else:
            agent = self.agent_2
            other_agent = self.agent_1
            props_1 = trajectory.data['props_1']
            item_and_utility_1 = trajectory.data['i_and_u_1']
            props_2 = trajectory.data['props_0']
            item_and_utility_2 = trajectory.data['i_and_u_0']
            # observation_1 = trajectory.data['obs_1']
            rewards_1 = trajectory.data['rewards_1']
            # observation_2 = trajectory.data['obs_0']
            rewards_2 = trajectory.data['rewards_0']
            old_log_ps = trajectory.data['log_ps_1']

        if self.use_transformer:
            props_1 = props_1.permute((1, 0, 2))
            props_2 = props_2.permute((1, 0, 2))
        
        item_and_utility_1_cat = torch.cat([item_and_utility_1, item_and_utility_2], dim=-1)
        item_and_utility_2_cat = torch.cat([item_and_utility_2, item_and_utility_1], dim=-1)

        observation_1 = {'prop': props_1, 'item_and_utility': item_and_utility_1_cat}
        observation_2 = {'prop': props_2, 'item_and_utility': item_and_utility_2_cat}

        if self.sum_rewards:
            rewards_1 = rewards_2 = rewards_1 + rewards_2

        _, V_1s = self.get_trajectory_values(agent, observation_1)
        _, V_2s = self.get_trajectory_values(other_agent, observation_2)

        log_ps, term_ent, utte_ent, prop_ent = self.get_trajectory_log_ps(agent, trajectory, is_first)

        A_1s = compute_gae_advantages(rewards_1, V_1s.detach(), gamma)
        A_2s = compute_gae_advantages(rewards_2, V_2s.detach(), gamma)
        
        gammas = torch.pow(
            gamma, 
            torch.arange(self.trajectory_len)
        ).repeat(self.batch_size, 1).to(A_1s.device)
        
        if proximal:
            ratios = torch.exp(log_ps - old_log_ps.detach())
            surrogates = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            loss_1 = -(A_1s * surrogates[:, :-1]).mean()
        else:
            loss_1 = -(gammas[:, :-1] * A_1s * log_ps[:, :-1]).mean()

        if not vanilla:
            loss_2 = -self.linear_aa_loss(A_1s, A_2s, log_ps[:, :-1], old_log_ps[:, :-1])
        else:
            loss_2 = torch.zeros(1, device=loss_1.device)
        
        return (loss_1 + loss_2), term_ent, utte_ent, prop_ent


    def critic_loss(self, trajectory: TrajectoryBatch, is_first: bool) -> float:
        gamma = self.train_cfg.gamma

        if is_first:
            agent = self.agent_1
            props = trajectory.data['props_0']
            item_and_utility_1 = trajectory.data['i_and_u_0']
            item_and_utility_2 = trajectory.data['i_and_u_1']
            rewards_1 = trajectory.data['rewards_0']
            rewards_2 = trajectory.data['rewards_1']
        else:
            agent = self.agent_2
            props = trajectory.data['props_1']
            item_and_utility_1 = trajectory.data['i_and_u_1']
            item_and_utility_2 = trajectory.data['i_and_u_0']
            rewards_1 = trajectory.data['rewards_1']
            rewards_2 = trajectory.data['rewards_0']

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
        values_c_shifted = values_c[:, :-1]
        values_t_shifted = values_t[:, 1:]
        rewards_shifted = rewards_1[:, :-1]
        
        critic_loss = F.huber_loss(values_c_shifted, rewards_shifted + gamma * values_t_shifted)
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
        
    def gen(self):
        observations, _ = self.env.reset()
        hidden_state_1, hidden_state_2 = None, None
        history = None
        trajectory = TrajectoryBatch(
            batch_size=self.batch_size , 
            max_traj_len=self.trajectory_len,
            device=self.agent_1.device
        )

        if self.use_transformer:
            history = deque(maxlen=self.agent_1.actor.model.transformer.max_seq_len)

        for t in range(self.trajectory_len * 2):
            if t % 2 == 0:
                hidden_state_1, action, log_p_1 = self.agent_1.sample_action(
                    observations=observations,
                    h_0=hidden_state_1,
                    history=history,
                    use_transformer=self.use_transformer
                )
            else:
                hidden_state_2, action, log_p_2 = self.agent_2.sample_action(
                    observations=observations,
                    h_0=hidden_state_2,
                    history=history,
                    use_transformer=self.use_transformer
                )
            observations, rewards, done, _, info = self.env.step(action)
            trajectory.add_step(
                action=torchify_actions(action, self.agent_1.device), 
                observations=cat_observations(observations, self.agent_1.device), 
                rewards=torch.tensor(rewards, device=self.agent_1.device),
                done=torch.tensor(done, device=self.agent_1.device),
                info=info,
                t=t
            )
        trajectory.add_step(
            action=torchify_actions(action, self.agent_1.device), 
            observations=cat_observations(observations, self.agent_1.device), 
            rewards=torch.tensor(rewards, device=self.agent_1.device),
            done=torch.tensor(done, device=self.agent_1.device), 
            info=info,
            t=t
        )
        return trajectory

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
                extend_history=True
            )
            hidden_state_2, action_2, log_p_2 = self.agent_2.sample_action(
                observations=observations_2,
                h_0=hidden_state_2,
                history=history,
                use_transformer=self.use_transformer,
                extend_history=False
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