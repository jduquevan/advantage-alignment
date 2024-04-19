import gymnasium as gym
import hydra
from omegaconf import OmegaConf

from utils import seed_all, instantiate_agent

from utils import (
    compute_discounted_returns,
    compute_gae_advantages,
    get_cosine_similarity_torch,
    get_gaussian_entropy,
    get_observation,
)

import gym
import wandb

from omegaconf import DictConfig
from utils import (
    update_target_network,
    wandb_stats,
    save_checkpoint,
    merge_train_step_metrics
)
from tqdm import tqdm

import torch.distributions as D



import math
import torch

import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod


class LinearModel(nn.Module):
    def __init__(self, in_size, out_size, device, num_hidden=1, encoder=None, use_gru=True):
        super(LinearModel, self).__init__()
        self.encoder = encoder

        self.use_gru = use_gru
        self.in_size = in_size

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, in_size) for i in range(num_hidden)])

        self.linear = nn.Linear(in_size, out_size)
        self.to(device)

    def forward(self, x, h_0=None, partial_forward=True):
        assert self.use_gru
        output, x = self.encoder(x, h_0)
        x = x.squeeze(0)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return output, self.linear(x)


class CategoricalPolicy(nn.Module):
    def __init__(self, device, model):
        super(CategoricalPolicy, self).__init__()
        self.model = model
        self.to(device)

    def forward(self, x, h_0=None, partial_forward=False):
        output, prop = self.model(x, h_0)

        prop_1, prop_2, prop_3 = prop

        prop_1 = prop_1.squeeze(0)
        prop_2 = prop_2.squeeze(0)
        prop_3 = prop_3.squeeze(0)

        prop_dist_1 = D.Categorical(prop_1)
        prop_dist_2 = D.Categorical(prop_2)
        prop_dist_3 = D.Categorical(prop_3)

        return output, (prop_dist_1, prop_dist_2, prop_dist_3)

    def sample_action(self, x, h_0=None):
        action = {}

        output, prop_cat = self.forward(x, h_0, partial_forward=True)
        prop_cat_1, prop_cat_2, prop_cat_3 = prop_cat

        prop_1 = prop_cat_1.sample()
        prop_2 = prop_cat_2.sample()
        prop_3 = prop_cat_3.sample()

        prop = torch.cat([
            prop_1.unsqueeze(1),
            prop_2.unsqueeze(1),
            prop_3.unsqueeze(1)
        ], dim=1)

        log_p_prop = (
            prop_cat_1.log_prob(prop_1) +
            prop_cat_2.log_prob(prop_2) +
            prop_cat_3.log_prob(prop_3)
        )
        log_p = log_p_prop

        action['prop'] = prop

        return output, action, log_p

class Agent(ABC):
    @abstractmethod
    def sample_action(self):
        pass

class GruModel(nn.Module):
    def __init__(self, in_size, device, hidden_size=40, num_layers=1):
        super(GruModel, self).__init__()

        self.in_size = in_size
        self.device = device
        self.hidden_size = hidden_size

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.to(device)

    def forward(self, x, h_0=None):
        self.gru.flatten_parameters()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        output, x = self.gru(x.unsqueeze(1), h_0)
        return output, x

    def get_embeds(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return x


class MLPModelDiscrete(nn.Module):
    def __init__(self, in_size, device, output_size, num_layers=1, hidden_size=40, encoder=None, use_gru=True):
        super(MLPModelDiscrete, self).__init__()
        # self.transformer = transformer
        self.num_layers = num_layers
        self.encoder = encoder
        self.use_gru = use_gru

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])

        # Proposition heads
        assert output_size == 6, f"when designing the game we fixated on 5 being max items, if that changed, remove this: output_size now is: {output_size}"
        self.prop_heads = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(3)])
        self.temp = nn.Parameter(torch.ones(1))

        self.to(device)

    def forward(self, x, h_0=None, partial_forward=True):
        assert self.use_gru
        output, x = self.encoder(x, h_0)
        x = x.squeeze(0)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Prop
        prop_probs = [F.softmax(prop_head(x) / self.temp, dim=-1) for prop_head in self.prop_heads]

        return output, prop_probs


class AdvantageAlignmentAgent(Agent):
    def __init__(
        self,
        device: torch.device,
        actor: nn.Module,
        critic: LinearModel,
        target: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()
        self.device = device
        self.actor = actor
        self.critic = critic
        self.target = target
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer

    def sample_action(
        self,
        observations,
        h_0=None,
        history=None,
        extend_history=False,
        is_first=False,
        **kwargs
    ):
        if h_0 is not None:
            h_0 = torch.permute(h_0, (1, 0, 2))

        return self.actor.sample_action(observations, h_0)

class TrajectoryBatch():
    def __init__(self, batch_size: int, max_traj_len: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.max_traj_len = max_traj_len
        self.device = device
        self.data = {
            "obs_0": torch.ones(
                (batch_size, max_traj_len, 15),
                dtype=torch.float32,
                device=device,
            ),
            "obs_1": torch.ones(
                (batch_size, max_traj_len, 15),
                dtype=torch.float32,
                device=device,
            ),
            "props_0": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "props_1": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "item_pool": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "utility_1": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "utility_2": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "rewards_0": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "rewards_1": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_0": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_1": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "dones": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.bool,
            ),
            "cos_sims": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "max_sum_rewards": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "max_sum_reward_ratio": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
        }

    def add_step_sim(self, action, observations, rewards, log_ps, done, info, t, item_pool, utilities_1, utilities_2):
        self.data['props_0'][:, t, :] = action[0]['prop']
        self.data['props_1'][:, t, :] = action[1]['prop']
        self.data['item_pool'][:, t, :] = item_pool
        self.data['utility_1'][:, t, :] = utilities_1
        self.data['utility_2'][:, t, :] = utilities_2
        self.data['obs_0'][:, t, :] = observations[0]
        self.data['obs_1'][:, t, :] = observations[1]
        self.data['rewards_0'][:, t] = rewards[0]
        self.data['rewards_1'][:, t] = rewards[1]
        self.data['log_ps_0'][:, t] = log_ps[0]
        self.data['log_ps_1'][:, t] = log_ps[1]
        self.data['max_sum_rewards'][:, t] = info['max_sum_rewards']
        self.data['max_sum_reward_ratio'][:, t] = info['max_sum_reward_ratio']
        self.data['cos_sims'][:, t] = info['cos_sims']
        self.data['dones'][:, t] = done


class TrainingAlgorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        agent_1: Agent,
        agent_2: Agent,
        train_cfg: DictConfig,
        sum_rewards: bool,
        simultaneous: bool,
        discrete: bool,
    ) -> None:
        self.env = env
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.train_cfg = train_cfg
        self.trajectory_len = train_cfg.max_traj_len
        self.sum_rewards = sum_rewards
        self.simultaneous = simultaneous
        self.discrete = discrete
        if self.simultaneous:
            self.batch_size = env.batch_size

    def train_step(
        self,
        trajectory,
        agent,
        is_first=True
    ):
        agent.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(trajectory, is_first)
        critic_loss.backward()
        critic_grad_norm = torch.sqrt(sum([torch.norm(p.grad) ** 2 for p in agent.critic.parameters()]))
        agent.critic_optimizer.step()

        agent.actor_optimizer.zero_grad()

        actor_loss_dict = self.actor_loss(trajectory, is_first)
        actor_loss = actor_loss_dict['loss']
        ent = actor_loss_dict['prop_ent']
        train_step_metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "prop_entropy": actor_loss_dict['prop_ent'].detach()
        }

        total_loss = actor_loss - self.train_cfg.entropy_beta * ent
        total_loss.backward()
        actor_grad_norm = torch.sqrt(sum([torch.norm(p.grad) ** 2 for p in agent.actor.parameters()]))
        agent.actor_optimizer.step()

        update_target_network(agent.target, agent.critic, self.train_cfg.tau)
        train_step_metrics["critic_grad_norm"] = critic_grad_norm.detach()
        train_step_metrics["actor_grad_norm"] = actor_grad_norm.detach()

        return train_step_metrics

    def eval(self, agent):
        pass

    def save(self):
        pass

    def train_loop(self):

        pbar = tqdm(range(self.train_cfg.total_steps))

        for step in pbar:
            if step % self.train_cfg.save_every == 0:
                save_checkpoint(self.agent_1, step, self.train_cfg.checkpoint_dir)

            trajectory = self.gen_sim()

            wandb_metrics = wandb_stats(trajectory)

            for i in range(self.train_cfg.updates_per_batch):
                train_step_metrics_1 = self.train_step(
                    trajectory=trajectory,
                    agent=self.agent_1,
                    is_first=True
                )

                train_step_metrics_2 = self.train_step(
                    trajectory=trajectory,
                    agent=self.agent_2,
                    is_first=False
                )

                train_step_metrics = merge_train_step_metrics(
                    train_step_metrics_1,
                    train_step_metrics_2,
                )

                for key, value in train_step_metrics.items():
                    print(key + ":", value)
                print()

            wandb_metrics.update(train_step_metrics)

            eval_metrics = self.eval(self.agent_1)
            print("Evaluation metrics:", eval_metrics)
            wandb_metrics.update(eval_metrics)

            wandb.log(step=step, data=wandb_metrics)

        return

    """ TO BE DEFINED BY INDIVIDUAL ALGORITHMS"""

    @abstractmethod
    def actor_loss(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_loss(batch: dict) -> float:
        pass


class AdvantageAlignment(TrainingAlgorithm):

    def get_entropy(self, agent):
        prop_stds = torch.exp(agent.actor.prop_log_std.squeeze(0))
        prop_ent = get_gaussian_entropy(torch.diag_embed(prop_stds))
        return prop_ent

    def get_entropy_discrete(self, prop_dist):
        prop_dist_1, prop_dist_2, prop_dist_3 = prop_dist
        prop_ent = (
            prop_dist_1.entropy() +
            prop_dist_2.entropy() +
            prop_dist_3.entropy()
        ).mean()

        return prop_ent

    def get_trajectory_log_ps(self, agent, trajectory, is_first):  # todo: make sure logps are the same as when sampling
        log_p_list = []
        if self.discrete:
            prop_max = 1
        else:
            prop_max = self.agent_1.actor.prop_max

        if is_first:
            props = trajectory.data['props_0']
            observation = trajectory.data['obs_0']
        else:
            observation = trajectory.data['obs_1']
            props = trajectory.data['props_1']

        prop_ent = torch.zeros(self.trajectory_len, device=self.agent_1.device, dtype=torch.float32)

        hidden = None
        for t in range(self.trajectory_len):
            hidden, prop_dist = agent.actor(x=get_observation(observation, t), h_0=hidden)
            hidden = hidden.permute((1, 0, 2))

            if self.discrete:
                prop_en = self.get_entropy_discrete(prop_dist)
                prop_cat_1, prop_cat_2, prop_cat_3 = prop_dist
                log_p_prop = (
                    prop_cat_1.log_prob(props[:, t, 0]) +
                    prop_cat_2.log_prob(props[:, t, 1]) +
                    prop_cat_3.log_prob(props[:, t, 2])
                ).squeeze(0)
            else:
                prop_en = self.get_entropy(agent)
                log_p_prop = prop_dist.log_prob(props[:, t] / prop_max).sum(dim=-1)
            prop_ent[t] = prop_en

            log_p = log_p_prop
            log_p_list.append(log_p)
            log_ps = torch.stack(log_p_list).permute((1, 0))
        prop_ent = prop_ent.mean()

        return log_ps, prop_ent

    def get_trajectory_values(self, agent, observation):
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
            obs = get_observation(observation, t)
            hidden_c, value_c = agent.critic(obs, h_0=hidden_c)
            hidden_t, value_t = agent.target(obs, h_0=hidden_t)
            values_c[:, t] = value_c.reshape(self.batch_size)
            values_t[:, t] = value_t.reshape(self.batch_size)
            hidden_c = hidden_c.permute((1, 0, 2))
            hidden_t = hidden_t.permute((1, 0, 2))
        return values_c, values_t

    def get_trajectory_rewards(self, agent, observation):
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
            A_1_t = A_1s[:, :t + 1]
            A_2_t = A_2s[:, t].unsqueeze(1) * torch.ones(action_term.size(0), t + 1).to(device)
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
            observation_1 = trajectory.data['obs_0']  # todo: milad, fix this 1 vs 0 switching to 1 vs 2 mix
            rewards_1 = trajectory.data['rewards_0']
            observation_2 = trajectory.data['obs_1']
            rewards_2 = trajectory.data['rewards_1']
            old_log_ps = trajectory.data['log_ps_0']
        else:
            agent = self.agent_2
            other_agent = self.agent_1
            observation_1 = trajectory.data['obs_1']
            rewards_1 = trajectory.data['rewards_1']
            observation_2 = trajectory.data['obs_0']
            rewards_2 = trajectory.data['rewards_0']
            old_log_ps = trajectory.data['log_ps_1']

        if self.sum_rewards:
            rewards_1 = rewards_2 = rewards_1 + rewards_2

        _, V_1s = self.get_trajectory_values(agent, observation_1)
        _, V_2s = self.get_trajectory_values(other_agent, observation_2)

        log_ps, prop_ent = self.get_trajectory_log_ps(agent, trajectory, is_first)
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
            observation = trajectory.data['obs_0']
            rewards_1 = trajectory.data['rewards_0']
            rewards_2 = trajectory.data['rewards_1']
        else:
            agent = self.agent_2
            observation = trajectory.data['obs_1']
            rewards_1 = trajectory.data['rewards_1']
            rewards_2 = trajectory.data['rewards_0']

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

    def gen_sim(self):
        return self._gen_sim(self.env, self.batch_size, self.trajectory_len, self.agent_1, self.agent_2)

    @staticmethod
    def _gen_sim(env, batch_size, trajectory_len, agent_1, agent_2):
        observations, _ = env.reset()
        observations_1, observations_2 = observations

        hidden_state_1, hidden_state_2 = None, None
        history = None
        trajectory = TrajectoryBatch(
            batch_size=batch_size,
            max_traj_len=trajectory_len,
            device=agent_1.device
        )

        for t in range(trajectory_len):
            hidden_state_1, action_1, log_p_1 = agent_1.sample_action(observations=observations_1, h_0=hidden_state_1, history=history, extend_history=True, is_first=True,
                                                                      item_pool=env.item_pool, utilities_1=env.utilities_1, utilities_2=env.utilities_2)
            hidden_state_2, action_2, log_p_2 = agent_2.sample_action(observations=observations_2, h_0=hidden_state_2, history=history, extend_history=False, is_first=False,
                                                                      item_pool=env.item_pool, utilities_1=env.utilities_1, utilities_2=env.utilities_2)
            utilities_1 = env.utilities_1
            utilities_2 = env.utilities_2  # storing them for putting them in the trajectory later, step method modifies them

            next_observations, rewards, done, _, info = env.step((action_1, action_2))

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
                t=t,
                item_pool=env.item_pool,
                utilities_1=utilities_1,
                utilities_2=utilities_2,
            )

            observations_1, observations_2 = next_observations

        return trajectory

    def eval(self, agent):
        gen_episodes_between_agents = self._gen_sim
        # todo: assert eg, check the code, etc
        assert self.env.name == 'eg_obligated_ratio'
        # evaluate against always cooperate and always defect
        always_cooperate_agent = AlwaysCooperateAgent(max_possible_prop=self.env.item_max_quantity, device=agent.device)
        always_defect_agent = AlwaysDefectAgent(max_possible_prop=self.env.item_max_quantity, device=agent.device)

        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len,
                                                 agent_1=agent, agent_2=always_cooperate_agent)
        agent_vs_ac_rewards_agent = trajectory.data['rewards_0'].mean().item()
        agent_vs_ac_rewards_ac = trajectory.data['rewards_1'].mean().item()

        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len,
                                                 agent_1=agent, agent_2=always_defect_agent)
        agent_vs_ad_rewards_agent = trajectory.data['rewards_0'].mean().item()
        agent_vs_ad_rewards_ad = trajectory.data['rewards_1'].mean().item()

        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len,
                                                 agent_1=always_defect_agent, agent_2=always_defect_agent)
        ad_vs_ad_rewards_ad_0 = trajectory.data['rewards_0'].mean().item()
        ad_vs_ad_rewards_ad_1 = trajectory.data['rewards_1'].mean().item()

        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len,
                                                 agent_1=always_cooperate_agent, agent_2=always_cooperate_agent)
        ac_vs_ac_rewards_ac_0 = trajectory.data['rewards_0'].mean().item()
        ac_vs_ac_rewards_ac_1 = trajectory.data['rewards_1'].mean().item()

        trajectory = gen_episodes_between_agents(env=self.env, batch_size=self.batch_size, trajectory_len=self.trajectory_len,
                                                 agent_1=always_defect_agent, agent_2=always_cooperate_agent)
        ad_vs_ac_rewards_ad = trajectory.data['rewards_0'].mean().item()
        ad_vs_ac_rewards_ac = trajectory.data['rewards_1'].mean().item()

        metrics = {
            'agent_vs_ac_rewards_agent': agent_vs_ac_rewards_agent,
            'agent_vs_ac_rewards_ac': agent_vs_ac_rewards_ac,
            'agent_vs_ad_rewards_agent': agent_vs_ad_rewards_agent,
            'agent_vs_ad_rewards_ad': agent_vs_ad_rewards_ad,
            'ad_vs_ad_rewards_ad_0': ad_vs_ad_rewards_ad_0,
            'ad_vs_ad_rewards_ad_1': ad_vs_ad_rewards_ad_1,
            'ac_vs_ac_rewards_ac_0': ac_vs_ac_rewards_ac_0,
            'ac_vs_ac_rewards_ac_1': ac_vs_ac_rewards_ac_1,
            'ad_vs_ac_rewards_ac': ad_vs_ac_rewards_ac,
            'ad_vs_ac_rewards_ad': ad_vs_ac_rewards_ad
        }
        return metrics


class ObligatedRatioDiscreteEG(gym.Env):
    def __init__(
        self,
        item_max_quantity,
        max_turns={
            "lower": 3,
            "mean": 3,
            "upper": 3,
        },
        nb_items=3,
        utility_max=5,
        device=None,
        batch_size=128,
        sampling_type="no_sampling",
        normalize_rewards=False,
    ):
        super().__init__()

        self.name = 'eg_obligated_ratio'
        self._max_turns = max_turns
        self.nb_items = nb_items
        self.item_max_quantity = item_max_quantity
        self.utility_max = utility_max
        self.device = device
        self.batch_size = batch_size
        self.sampling_type = sampling_type
        self.normalize_rewards = normalize_rewards
        self.info = {}

    def _get_max_sum_rewards(self):
        utility_comparer = (self.utilities_1 >= self.utilities_2)
        agent_1_gets = torch.where(utility_comparer, self.item_pool, 0)
        agent_2_gets = self.item_pool - agent_1_gets
        max_sum_rewards = (agent_1_gets * self.utilities_1).sum(dim=-1) + (agent_2_gets * self.utilities_2).sum(dim=-1)
        return max_sum_rewards

    @staticmethod
    def _get_obs(item_pool, utilities_1, utilities_2, prop_1, prop_2):
        item_context_1 = torch.cat([item_pool, utilities_1, prop_1, utilities_2, prop_2], dim=1).float()
        item_context_2 = torch.cat([item_pool, utilities_2, prop_2, utilities_1, prop_1], dim=1).float()
        # scale them down
        item_context_1 = item_context_1 / 100 # todo: milad, check if this is necessary
        item_context_2 = item_context_2 / 100
        return item_context_1, item_context_2

    def _sample_utilities(self):
        if self.sampling_type == "uniform":
            utilities_1 = torch.randint(
                1,
                self.utility_max + 1,
                (self.batch_size, self.nb_items)
            ).to(self.device)
            utilities_2 = torch.randint(
                1,
                self.utility_max + 1,
                (self.batch_size, self.nb_items)
            ).to(self.device)
            item_pool = torch.randint(
                1,
                self.item_max_quantity + 1,
                (self.batch_size, self.nb_items)
            ).to(self.device)
        elif self.sampling_type == "no_sampling":
            # For debugging purposes
            utilities_1 = torch.tensor([10, 3, 2]).repeat(self.batch_size, 1).to(self.device)
            utilities_2 = torch.tensor([3, 10, 2]).repeat(self.batch_size, 1).to(self.device)
            item_pool = torch.tensor([4, 4, 4]).repeat(self.batch_size, 1).to(self.device)
        elif self.sampling_type == "high_contrast":
            utilities_1 = torch.randint(0, 2, (self.batch_size, self.nb_items)).to(self.device) * (self.utility_max - 1) + 1
            utilities_2 = self.utility_max + 1 - utilities_1
            item_pool = torch.tensor([[1, 1, 1]]).repeat(self.batch_size, 1).to(self.device)
        elif self.sampling_type == "high_contrast_no_sampling":
            utilities_1 = torch.tensor([[5, 5, 1]]).repeat(self.batch_size, 1).to(self.device)
            utilities_2 = torch.tensor([[1, 1, 5]]).repeat(self.batch_size, 1).to(self.device)
            item_pool = torch.tensor([[1, 1, 1]]).repeat(self.batch_size, 1).to(self.device)
        else:
            raise Exception(f"Unsupported sampling type: '{self.sampling_type}'.")

        return utilities_1, utilities_2, item_pool

    def reset(self):
        self.a1_is_done = torch.torch.full(
            (self.batch_size,),
            False,
            dtype=torch.bool
        ).to(self.device)
        self.a2_is_done = torch.torch.full(
            (self.batch_size,),
            False,
            dtype=torch.bool
        ).to(self.device)
        self.env_is_done = torch.torch.full(
            (self.batch_size,),
            False,
            dtype=torch.bool
        ).to(self.device)

        self.turn = torch.zeros((self.batch_size,)).to(self.device)

        self.utilities_1, self.utilities_2, self.item_pool = self._sample_utilities()

        max_turns = torch.poisson(
            torch.ones(self.batch_size) * self._max_turns["mean"]
        ).to(self.device)
        self.max_turns = torch.clamp(
            max_turns,
            self._max_turns["lower"],
            self._max_turns["upper"]
        )

        self.prop_dummy = torch.zeros((self.batch_size, self.nb_items)).to(self.device)
        obs = self._get_obs(self.item_pool, self.utilities_1, self.utilities_2, self.prop_dummy, self.prop_dummy)

        self.info["max_sum_rewards"] = self._get_max_sum_rewards()

        return obs, self.info

    def step(self, actions):

        self.turn += 1
        actions_1, actions_2 = actions
        prop_1 = actions_1["prop"]
        prop_2 = actions_2["prop"]

        rewards_1 = ((prop_1 / (prop_1 + prop_2 + 1e-8)) * self.item_pool * self.utilities_1).sum(dim=-1)
        rewards_2 = ((prop_2 / (prop_1 + prop_2 + 1e-8)) * self.item_pool * self.utilities_2).sum(dim=-1)

        assert (self.utilities_1 >= 0).all() and (self.utilities_2 >= 0).all()
        max_sum_rewards = self._get_max_sum_rewards()
        if self.sampling_type in ['uniform', 'no_sampling', 'high_contrast', 'high_contrast_no_sampling']:
            max_reward_1 = (self.item_pool * self.utilities_1).sum(dim=-1)
            max_reward_2 = (self.item_pool * self.utilities_2).sum(dim=-1)
        else:
            raise Exception(f"Unsupported sampling type: '{self.sampling_type}'.")

        if self.normalize_rewards:
            norm_rewards_1 = rewards_1 / max_reward_1
            norm_rewards_2 = rewards_2 / max_reward_2
        else:
            norm_rewards_1 = rewards_1
            norm_rewards_2 = rewards_2

        max_sum_reward_ratio = (rewards_1 + rewards_2) / max_sum_rewards

        done = (self.turn >= self.max_turns)

        new_utilities_1, new_utilities_2, new_item_pool = self._sample_utilities()

        rewards_1 = torch.where(done, norm_rewards_1, 0)
        rewards_2 = torch.where(done, norm_rewards_2, 0)

        reset = done.unsqueeze(1).repeat(1, 3)
        self.utilities_1 = torch.where(reset, new_utilities_1, self.utilities_1)
        self.utilities_2 = torch.where(reset, new_utilities_2, self.utilities_2)
        self.item_pool = torch.where(reset, new_item_pool, self.item_pool)
        self.turn = torch.where(done, 0, self.turn)

        max_sum_rewards = torch.where(done, max_sum_rewards, 0)
        max_sum_reward_ratio = torch.where(done, max_sum_reward_ratio, 0)

        self.info["max_sum_rewards"] = max_sum_rewards
        self.info["max_sum_reward_ratio"] = max_sum_reward_ratio
        self.info["cos_sims"] = get_cosine_similarity_torch(self.utilities_1, self.utilities_2)

        obs = self._get_obs(self.item_pool, self.utilities_1, self.utilities_2, prop_1, prop_2)

        return obs, (rewards_1, rewards_2), done, None, self.info


def test_obligated_ratio_discrete_eg():
    env = ObligatedRatioDiscreteEG(item_max_quantity=3)
    env.reset()
    prop_1 = torch.randint(0, 3, (128, 3))
    prop_2 = torch.randint(0, 3, (128, 3))
    actions_1 = {
        "term": torch.zeros((128)).to(env.device),
        "prop": prop_1
    }
    actions_2 = {
        "term": torch.zeros((128)).to(env.device),
        "prop": prop_2
    }
    env.step((actions_1, actions_2))


class AlwaysCooperateAgent(Agent):
    def __init__(self,
                 max_possible_prop,
                 device):
        self.device = device
        self.max_possible_prop = max_possible_prop

    def sample_action(self,
                      observations,
                      is_first: bool,
                      **kwargs):

        if is_first:
            my_utilities = kwargs.pop('utilities_1')
            your_utilities = kwargs.pop('utilities_2')
        else:
            my_utilities = kwargs.pop('utilities_2')
            your_utilities = kwargs.pop('utilities_1')

        ans = torch.where(my_utilities > your_utilities, self.max_possible_prop, 0).to(self.device)

        # breaking ties
        equal_places = torch.where(my_utilities == your_utilities, self.max_possible_prop, ans)
        coin_flip = torch.randint(0, 2, equal_places.shape).to(self.device)
        ans = torch.where(coin_flip == 1, equal_places, ans)

        log_p = torch.tensor(0.).to(self.device)  # probability of doing that is 1, not for the equal actions as they are log(0.5) but we ignore, we don't train this agent

        return None, {'prop': ans}, log_p


class AlwaysDefectAgent(Agent):
    def __init__(self,
                 max_possible_prop,
                 device):
        self.device = device
        self.max_possible_prop = max_possible_prop

    def sample_action(self,
                      observations,
                      is_first: bool,
                      **kwargs):

        if is_first:
            my_utilities = kwargs.pop('utilities_1')
            your_utilities = kwargs.pop('utilities_2')
        else:
            my_utilities = kwargs.pop('utilities_2')
            your_utilities = kwargs.pop('utilities_1')

        ans = torch.ones_like(my_utilities).to(self.device) * self.max_possible_prop

        log_p = torch.tensor(0.).to(self.device)  # probability of doing that is 1

        return None, {'prop': ans}, log_p


def test_fixed_agents():
    gen_episodes_between_agents = AdvantageAlignment._gen_sim
    batch_size = 1024
    item_max_quantity = 5
    trajectory_len = 50
    env = ObligatedRatioDiscreteEG(item_max_quantity=item_max_quantity, batch_size=batch_size, device='cpu', sampling_type=
    'high_contrast')
    agent_1 = AlwaysCooperateAgent(max_possible_prop=item_max_quantity, device='cpu')
    agent_2 = AlwaysCooperateAgent(max_possible_prop=item_max_quantity, device='cpu')
    trajectory = gen_episodes_between_agents(env, batch_size, trajectory_len, agent_1, agent_2)
    print(trajectory.data.keys())
    # print rewards
    print("Always Cooperate vs Always Cooperate")
    print(f"mean rewards 1: {trajectory.data['rewards_0'].mean()}")
    print(f"mean rewards 2: {trajectory.data['rewards_1'].mean()}")
    print(f"mean rewards 1 + 2: {(trajectory.data['rewards_0'] + trajectory.data['rewards_1']).mean()}q")
    # print actions
    # print(trajectory.data['a_props_0'][-1])
    # print(trajectory.data['a_props_1'][-1])

    agent_1 = AlwaysDefectAgent(max_possible_prop=item_max_quantity, device='cpu')
    agent_2 = AlwaysDefectAgent(max_possible_prop=item_max_quantity, device='cpu')
    trajectory = gen_episodes_between_agents(env, batch_size, trajectory_len, agent_1, agent_2)
    print(trajectory.data.keys())
    # print rewards
    print("Always Defect vs Always Defect")
    print(f"mean rewards 1: {trajectory.data['rewards_0'].mean()}")
    print(f"mean rewards 2: {trajectory.data['rewards_1'].mean()}")
    print(f"mean rewards 1 + 2: {(trajectory.data['rewards_0'] + trajectory.data['rewards_1']).mean()}q")

    agent_1 = AlwaysCooperateAgent(max_possible_prop=item_max_quantity, device='cpu')
    agent_2 = AlwaysDefectAgent(max_possible_prop=item_max_quantity, device='cpu')
    trajectory = gen_episodes_between_agents(env, batch_size, trajectory_len, agent_1, agent_2)
    print(trajectory.data.keys())
    # print rewards
    print("Always Cooperate vs Always Defect")
    print(f"mean rewards 1: {trajectory.data['rewards_0'].mean()}")
    print(f"mean rewards 2: {trajectory.data['rewards_1'].mean()}")
    print(f"mean rewards 1 + 2: {(trajectory.data['rewards_0'] + trajectory.data['rewards_1']).mean()}q")

    agent_1 = AlwaysDefectAgent(max_possible_prop=item_max_quantity, device='cpu')
    agent_2 = AlwaysCooperateAgent(max_possible_prop=item_max_quantity, device='cpu')
    trajectory = gen_episodes_between_agents(env, batch_size, trajectory_len, agent_1, agent_2)
    print(trajectory.data.keys())
    # print rewards
    print("Always Defect vs Always Cooperate")
    print(f"mean rewards 1: {trajectory.data['rewards_0'].mean()}")
    print(f"mean rewards 2: {trajectory.data['rewards_1'].mean()}")
    print(f"mean rewards 1 + 2: {(trajectory.data['rewards_0'] + trajectory.data['rewards_1']).mean()}q")


@hydra.main(version_base="1.3", config_path="configs", config_name="f1.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    seed_all(cfg['seed'])
    wandb.init(
        config=dict(cfg),
        project="Advantage Alignment",
        dir=cfg["wandb_dir"],
        reinit=True,
        anonymous="allow",
        mode=cfg["wandb"],
        tags=cfg.wandb_tags,
    )
    if cfg['env']['type'] == 'eg-obligated_ratio':
        env = ObligatedRatioDiscreteEG(max_turns={
            'lower': cfg['env']['max_turns_lower'],
            'mean': cfg['env']['max_turns_mean'],
            'upper': cfg['env']['max_turns_upper']
        },
            sampling_type=cfg['env']['sampling_type'],
            item_max_quantity=cfg['env']['item_max_quantity'],
            batch_size=cfg['batch_size'],
            device=cfg['env']['device'])
    else:
        raise ValueError(f"Environment type {cfg['env']['type']} not supported.")

    if cfg['self_play']:
        agent_1 = agent_2 = instantiate_agent(cfg)
    else:
        agent_1 = instantiate_agent(cfg)
        agent_2 = instantiate_agent(cfg)

    algorithm = AdvantageAlignment(
        env=env,
        agent_1=agent_1,
        agent_2=agent_2,
        train_cfg=cfg.training,
        sum_rewards=cfg['sum_rewards'],
        simultaneous=cfg['simultaneous'],
        discrete=cfg['discrete'],
    )
    algorithm.train_loop()


if __name__ == "__main__":
    # test_fixed_agents()
    # test_eg_obligated_ratio()
    OmegaConf.register_new_resolver("eval", eval)
    main()
