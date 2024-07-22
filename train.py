import gymnasium as gym
import hydra
from omegaconf import OmegaConf

from utils import seed_all, instantiate_agent

from utils import (
    compute_discounted_returns,
    compute_gae_advantages,
    get_cosine_similarity_torch,
    get_observation,
)

import copy
import gym
import math
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

import torch

import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod
from collections import deque
from torch.distributions import Categorical
from torch.distributions.transforms import SigmoidTransform
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.meltingpot import MeltingpotEnv
from torch.func import functional_call, stack_module_state

class LinearModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, device, num_layers=1, encoder=None, append_time: bool = False):
        super(LinearModel, self).__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size

        self.in_size = in_size + (1 if append_time else 0)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.in_size, self.hidden_size) if i==0 
            else nn.Linear(self.hidden_size, self.hidden_size) 
            for i in range(num_layers)])

        self.linear = nn.Linear(self.in_size, out_size)
        self.append_time = append_time
        self.to(device)

    def forward(self, obs, actions, t: int = None):
        output = self.encoder(obs, actions)[:, -1, :]

        for layer in self.hidden_layers:
            output = F.relu(layer(output))

        return self.linear(output)

    
# Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MeltingpotTransformer(nn.Module):
    def __init__(
        self,
        in_size, 
        d_model, 
        device,
        dim_feedforward=40,  
        num_layers=1,
        num_embed_layers=1,
        nhead=4, 
        max_seq_len=100, 
        dropout=0.1
        ):

        super(MeltingpotTransformer, self).__init__()
        self.layers = nn.ModuleList()
        
        self.in_size = in_size
        self.d_model = d_model
        self.device = device

        self.num_layers = num_layers
        self.n_heads = nhead
        self.max_seq_len = max_seq_len

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.embed_obs_layer = nn.Linear(64*7*7, d_model)
        self.embed_action = nn.Embedding(num_embeddings=4, embedding_dim=d_model)

        self.embed_layers = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(num_embed_layers)])

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        for _ in range(self.num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                norm_first=True
            )
            # Move the layer to the specified device
            encoder_layer = encoder_layer.to(device)
            self.layers.append(encoder_layer)
    
    def forward(self, obs, actions):
        src = self.get_embeds(obs, actions)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = src
        for layer in self.layers:
            output = layer(output) + src
        return output

    def get_embeds(self, obs, actions):
        T, _, _, _ = obs.shape
        H = self.d_model

        obs = F.relu(self.conv1(obs.permute(0, 3, 1, 2)))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        embed_obs = F.relu(self.embed_obs_layer(obs.reshape(T, -1)))
        
        embed_actions = self.embed_action(actions)

        for embed_layer in self.embed_layers:
            embed_obs = F.relu(embed_layer(embed_obs))
            embed_actions = F.relu(embed_layer(embed_actions))

        # Interleave states and actions: a, s, a, ...
        src = torch.stack((embed_actions, embed_obs), dim=2).view(1, -1, H)

        return src

class NormalSigmoidPolicy(nn.Module):
    def __init__(self, log_std_min, log_std_max, prop_max, device, model):
        super(NormalSigmoidPolicy, self).__init__()
        self.model = model
        self.log_std_min = torch.tensor(log_std_min).to(device)
        self.log_std_max = torch.tensor(log_std_max).to(device)
        self.prop_max = prop_max
        self.device = device
        self.to(device)

    def forward(self, x, h_0=None):
        output, prop = self.model(x, h_0)
        prop_mean, prop_log_std = prop
        print(f"####prop_mean_mean: {prop_mean.mean()}")

        prop_log_std = torch.clamp(input=prop_log_std, min=self.log_std_min, max=self.log_std_max)

        self.prop_log_std = prop_log_std

        base_prop_dist = D.normal.Normal(
            loc=prop_mean.squeeze(0),
            scale=torch.exp(prop_log_std.squeeze(0))
        )

        prop_normal_sigmoid = D.TransformedDistribution(base_prop_dist, [SigmoidTransform()])

        return output, prop_normal_sigmoid

    def sample_action(self, x, h_0=None):
        action = {}

        output, prop_normal_sigmoid = self.forward(x, h_0)
        prop = prop_normal_sigmoid.sample()

        log_p_prop = prop_normal_sigmoid.log_prob(prop).sum(dim=-1)  # Sum log probabilities over dimensions
        log_p = log_p_prop

        action['prop'] = (prop * self.prop_max)

        return output, action, log_p

class NormalTanhDistribution():
    def __init__(self, scale, bias, base_dist):
        self.scale = scale
        self.bias = bias
        self.base_dist = base_dist

    def sample(self):
        x_t = self.base_dist.sample()
        y_t = torch.tanh(x_t)
        action = y_t * self.scale + self.bias
        return action

    def log_prob(self, action):
        y_t = (action - self.bias) / self.scale
        y_t = torch.clamp(input=y_t, min=-0.99, max=0.99)
        x_t = torch.atanh(y_t)
        log_prob = self.base_dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return log_prob

class NormalTanhPolicy(nn.Module):
    def __init__(self, log_std_min, log_std_max, prop_max, device, model):
        super(NormalTanhPolicy, self).__init__()
        self.model = model
        self.log_std_min = torch.tensor(log_std_min).to(device)
        self.log_std_max = torch.tensor(log_std_max).to(device)
        self.prop_max = prop_max
        self.device = device
        self.action_scale = prop_max / 2.
        self.action_bias = 2.5
        self.to(device)

    def forward(self, x, h_0=None):
        output, prop = self.model(x, h_0)
        prop_mean, prop_log_std = prop
        prop_mean = torch.tanh(prop_mean) * 2. # todo: milad
        prop_log_std = torch.clamp(input=prop_log_std, min=self.log_std_min, max=self.log_std_max)

        prop_normal = D.normal.Normal(loc=prop_mean.squeeze(0), scale=torch.exp(prop_log_std.squeeze(0)))
        action_dist = NormalTanhDistribution(self.action_scale, self.action_bias, prop_normal)
        return output, action_dist

    def sample_action(self, x, h_0=None):
        output, action_dist = self.forward(x, h_0=h_0)
        action = {'prop': action_dist.sample()}
        log_prob = action_dist.log_prob(action['prop'])
        return output, action, log_prob

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
        # self.gru.flatten_parameters()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        output, x = self.gru(x.unsqueeze(1), h_0)
        return output, x

    def get_embeds(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return x


class MLPModelDiscrete(nn.Module):
    def __init__(self, in_size, device, output_size, num_layers=1, hidden_size=40, encoder=None, use_gru=True, soften_more=False):
        super(MLPModelDiscrete, self).__init__()
        # self.transformer = transformer
        self.num_layers = num_layers
        self.encoder = encoder
        self.use_gru = use_gru
        self.soften_more = soften_more

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, hidden_size) if i == 0 else nn.Linear(hidden_size, hidden_size) for i in range(num_layers)])

        # Proposition heads
        assert output_size == 6, f"when designing the game we fixated on 5 being max items, if that changed, remove this: output_size now is: {output_size}"
        self.prop_heads = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(3)])
        self.temp = nn.Parameter(torch.ones(1))

        self.to(device)

    def forward(self, x, h_0=None):
        assert self.use_gru
        output, x = self.encoder(x, h_0)
        x = x.squeeze(0)

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        if self.soften_more:
            prop_probs = [F.softmax(torch.sigmoid(prop_head(x)) * 5. / self.temp, dim=-1) for prop_head in self.prop_heads]
        else:
            prop_probs = [F.softmax(prop_head(x) / self.temp, dim=-1) for prop_head in self.prop_heads]

        return output, prop_probs


class AdvantageAlignmentAgent(Agent, nn.Module):
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

    def sample_action(self):
        # TODO: Remove, this interface is no longer necessary
        pass


class TrajectoryBatch():
    def __init__(self, sample_observation_1, batch_size: int, max_traj_len: int, device: torch.device, data=None) -> None:
        self.batch_size = batch_size
        self.max_traj_len = max_traj_len
        self.device = device
        obs_size = sample_observation_1.size(-1)
        if data is None:
            self.data = {
                "obs_0": torch.ones(
                    (batch_size, max_traj_len, obs_size),
                    dtype=torch.float32,
                    device=device,
                ),
                "obs_1": torch.ones(
                    (batch_size, max_traj_len, obs_size),
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
        else:
            self.data = data
            for key in self.data.keys():
                assert self.data[key].size(0) == batch_size
                assert self.data[key].size(1) == max_traj_len

    def subsample(self, batch_size: int):
        idxs = torch.randint(0, self.batch_size, (batch_size,))
        return TrajectoryBatch.create_from_data({k: v[idxs] for k, v in self.data.items()})

    @staticmethod
    def create_from_data(data):
        batch_size = data['obs_0'].size(0)
        max_traj_len = data['obs_0'].size(1)
        sample_obs = data['obs_0']
        device = data['obs_0'].device
        for key in data.keys():
            assert data[key].size(0) == batch_size
            assert data[key].size(1) == max_traj_len
        return TrajectoryBatch(sample_obs, batch_size, max_traj_len, device, data)

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
        self.data['log_ps_0'][:, t] = log_ps[0].detach()
        self.data['log_ps_1'][:, t] = log_ps[1].detach()
        self.data['max_sum_rewards'][:, t] = info['max_sum_rewards']
        self.data['max_sum_reward_ratio'][:, t] = info['max_sum_reward_ratio']
        self.data['cos_sims'][:, t] = info['cos_sims']
        self.data['dones'][:, t] = done

    def merge_two_trajectories(self, other):
        for key in self.data.keys():
            self.data[key] = torch.cat((self.data[key], other.data[key]), dim=0)
        self.batch_size += other.batch_size
        return self


class ReplayBuffer():
    def __init__(self,
                 env,
                 replay_buffer_size: int,
                 trajectory_len: int,
                 device: torch.device,):
        (obs_1, _), _ = env.reset()
        self.replay_buffer_size = replay_buffer_size
        self.marker = 0
        self.number_of_added_trajectories = 0
        self.trajectory_len = trajectory_len
        self.device = device
        self.trajectory_batch = TrajectoryBatch(obs_1, self.replay_buffer_size, self.trajectory_len, self.device)

    def add_trajectory_batch(self, new_batch: TrajectoryBatch):
        B = new_batch.data['obs_0'].size(0)
        assert (self.marker + B) <= self.replay_buffer_size # WARNING: we are assuming new data always comes in the same size, and the replay buffer is a multiple of that size
        for key in self.trajectory_batch.data.keys():
            self.trajectory_batch.data[key][self.marker:self.marker+B] = new_batch.data[key]
        self.marker = (self.marker + B) % self.replay_buffer_size
        self.number_of_added_trajectories += B

    def sample(self, batch_size: int):
        idxs = torch.randint(0, self.__len__(), (batch_size,))
        return TrajectoryBatch.create_from_data({k: v[idxs] for k, v in self.trajectory_batch.data.items()})

    def __len__(self):
        return min(self.number_of_added_trajectories, self.replay_buffer_size)

class AgentReplayBuffer:
    def __init__(self,
                 main_cfg: DictConfig, # to instantiate the agents from
                 replay_buffer_size: int,
                 agent_number: int, # either 1 or 2  as it is a two player game
                 ):
        self.main_cfg = main_cfg
        self.replay_buffer_size = replay_buffer_size
        self.marker = 0
        self.num_added_agents = 0
        self.agents = []
        for i in range(replay_buffer_size):
            self.agents.append(None) # create a [None, ..., None] list
        self.agent_number = agent_number

    def add_agent(self, agent):
        self.agents[self.marker] = agent.state_dict()
        self.num_added_agents += 1
        self.marker = (self.marker + 1) % self.replay_buffer_size

    def sample(self):
        idx = torch.randint(0, self.__len__(), (1,))
        state_dict = self.agents[idx]
        agent = instantiate_agent(self.main_cfg)
        agent.load_state_dict(state_dict)
        return agent

    def __len__(self):
        return min(self.num_added_agents, self.replay_buffer_size)


class TrainingAlgorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        agents: list[Agent],
        main_cfg: DictConfig,
        train_cfg: DictConfig,
        sum_rewards: bool,
        simultaneous: bool,
        discrete: bool,
        clip_grad_norm: float = None,
    ) -> None:
        self.env = env
        self.agents = agents
        self.main_cfg = main_cfg
        self.train_cfg = train_cfg
        self.trajectory_len = train_cfg.max_traj_len
        self.sum_rewards = sum_rewards
        self.simultaneous = simultaneous
        self.discrete = discrete
        if self.simultaneous:
            self.batch_size = env.batch_size
        self.clip_grad_norm = clip_grad_norm


    def train_step(
        self,
        trajectory,
        agent,
        is_first=True,
        compute_aux=True,
        optimize_critic=True,
    ):
        train_step_metrics = {}
        # --- optimize actor ---
        agent.actor_optimizer.zero_grad()
        actor_loss_dict = self.actor_loss(trajectory, is_first)
        actor_loss = actor_loss_dict['loss']

        ent = actor_loss_dict['prop_ent']

        if compute_aux:
            actor_loss_1 = actor_loss_dict['loss_1']
            actor_loss_2 = actor_loss_dict['loss_2']
            loss_1_grads = torch.autograd.grad(actor_loss_1, agent.actor.parameters(), create_graph=False, retain_graph=True)
            loss_1_norm = torch.sqrt(sum([torch.norm(p) ** 2 for p in loss_1_grads]))

            train_step_metrics["loss_1_grad_norm"] = loss_1_norm.detach()
            if actor_loss_2.requires_grad:
                loss_2_grads = torch.autograd.grad(actor_loss_2, agent.actor.parameters(), create_graph=False, retain_graph=True)
                loss_2_norm = torch.sqrt(sum([torch.norm(p) ** 2 for p in loss_2_grads]))
                train_step_metrics["loss_2_grad_norm"] = loss_2_norm.detach()
            else:
                train_step_metrics["loss_2_grad_norm"] = torch.tensor(0.0)

        total_loss = actor_loss - self.train_cfg.entropy_beta * ent
        total_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(agent.actor.parameters(), self.train_cfg.clip_grad_norm)
        actor_grad_norm = torch.sqrt(sum([torch.norm(p.grad) ** 2 for p in agent.actor.parameters()]))
        agent.actor_optimizer.step()

        # --- optimize critic ---
        agent.critic_optimizer.zero_grad()
        critic_loss, aux = self.critic_loss(trajectory, is_first)
        critic_values = aux['critic_values']
        target_values = aux['target_values']
        critic_loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(agent.critic.parameters(), self.train_cfg.clip_grad_norm)
        critic_grad_norm = torch.sqrt(sum([torch.norm(p.grad) ** 2 for p in agent.critic.parameters()]))

        if optimize_critic:
            agent.critic_optimizer.step()
        else:
            agent.critic_optimizer.zero_grad()

        # --- update target network ---

        update_target_network(agent.target, agent.critic, self.train_cfg.tau)

        # --- log metrics ---
        train_step_metrics.update({
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "prop_entropy": actor_loss_dict['prop_ent'].detach(),
            "critic_values": critic_values.mean().detach(),
            "target_values": target_values.mean().detach(),
        })
        train_step_metrics["critic_grad_norm"] = critic_grad_norm.detach()
        train_step_metrics["actor_grad_norm"] = actor_grad_norm.detach()
        for key in actor_loss_dict['log_ps_info'].keys():
            train_step_metrics['auto_merge_' + key] = actor_loss_dict['log_ps_info'][key].detach()

        return train_step_metrics

    def eval(self, agent):
        pass

    def save(self):
        pass

    def train_loop(self):
        # assert either replay buffer or agent replay buffer is used, but not both
        # assert self.use_replay_buffer != self.use_agent_replay_buffer, "Cannot use both replay buffer and agent replay buffer at the same time."
        pbar = tqdm(range(self.train_cfg.total_steps))

        for step in pbar:

            if step % self.train_cfg.save_every == 0:
                save_checkpoint(self.agents[0], step, self.train_cfg.checkpoint_dir)

            trajectory = self.gen_sim()
            
            wandb_metrics = wandb_stats(trajectory)

            for i in range(self.train_cfg.updates_per_batch):

                augmented_trajectories = trajectory # do not augment with off-policy data, as we don't have a replay buffer

                train_step_metrics_1 = self.train_step(
                    trajectory=augmented_trajectories,
                    agent=self.agent_1,
                    is_first=True,
                    compute_aux=(step % 20 == 0),
                    optimize_critic= (i % 2 == 0),
                )

                train_step_metrics_2 = self.train_step(
                    trajectory=augmented_trajectories,
                    agent=self.agent_2,
                    is_first=False,
                    compute_aux=(step % 20 == 0),
                    optimize_critic= (i % 2 == 0),
                )

                train_step_metrics = merge_train_step_metrics(
                    train_step_metrics_1,
                    train_step_metrics_2,
                )

                for key, value in train_step_metrics.items():
                    print(key + ":", value)
                print()

            wandb_metrics.update(train_step_metrics)
            print(f"train step metrics: {train_step_metrics}")

            eval_metrics = self.eval(self.agent_1)
            print("Evaluation metrics:", eval_metrics)
            wandb_metrics.update(eval_metrics)

            for i in range(10):
                mini_traj_table = wandb.Table(columns=["idx", "step", "item_pool", "utilities_1", "utilities_2", "props_1", "props_2", "rewards_1", "rewards_2"])
                mini_trajectory = {
                    "item_pool": trajectory.data['item_pool'][i, 0:10, :],
                    "utilities_1": trajectory.data['utility_1'][i, 0:10, :],
                    "utilities_2": trajectory.data['utility_2'][i, 0:10, :],
                    "props_1": trajectory.data['props_0'][i, 0:10, :],
                    "props_2": trajectory.data['props_1'][i, 0:10, :],
                    "rewards_1": trajectory.data['rewards_0'][i, 0:10],
                    "rewards_2": trajectory.data['rewards_1'][i, 0:10],
                }
                def torch_to_str(x):
                    return str(x.cpu().numpy().tolist())
                mini_trajectory = {k: torch_to_str(v) for k, v in mini_trajectory.items()}
                mini_traj_table.add_data(i,step, mini_trajectory["item_pool"], mini_trajectory["utilities_1"], mini_trajectory["utilities_2"], mini_trajectory["props_1"], mini_trajectory["props_2"], mini_trajectory["rewards_1"], mini_trajectory["rewards_2"])

            if step % 50 == 0:
                print("Mini trajectory:", mini_trajectory)
                wandb_metrics.update({"mini_trajectory_table": mini_traj_table})

            # log temperature of actor's softmax
            if self.discrete:
                wandb_metrics["actor_temp"] = self.agent_1.actor.model.temp.item()
            else:
                wandb_metrics['actor_log_std'] = self.agent_1.actor.model.prop_log_std.item()

            agent1_actor_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agent_1.actor.parameters()]))
            agent1_critic_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agent_1.critic.parameters()]))
            agent2_actor_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agent_2.actor.parameters()]))
            agent2_critic_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agent_2.critic.parameters()]))

            wandb_metrics.update({
                "agent1_actor_weight_norms": agent1_actor_weight_norms.detach(),
                "agent1_critic_weight_norms": agent1_critic_weight_norms.detach(),
                "agent2_actor_weight_norms": agent2_actor_weight_norms.detach(),
                "agent2_critic_weight_norms": agent2_critic_weight_norms.detach(),
            })
            wandb.log(step=step, data=wandb_metrics)

            # early stop if already cooperative
            if wandb_metrics['Average reward 1'] > 0.4:
                #save the model
                save_checkpoint(self.agent_1, 'last', self.train_cfg.checkpoint_dir)
                break
        return

    """ TO BE DEFINED BY INDIVIDUAL ALGORITHMS"""

    @abstractmethod
    def actor_loss(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_loss(batch: dict) -> float:
        pass


class AdvantageAlignment(TrainingAlgorithm):

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
                log_probs = prop_dist.log_prob(props[:, t]) # todo: milad, add back /prop_max if normal sigmoid policy
                # count the number of log_probs lower than -10
                log_probs_lower_than_n10 = torch.where(log_probs < -10, torch.tensor(1.), torch.tensor(0.)).sum()
                log_probs = torch.clamp_min(log_probs, -10)  # todo: milad, check if this is necessary

                # estimate entropy via KL w.r.t uniform distribution - which is just expected value of -log(p)
                prop_en = -1 * torch.clamp_min(log_probs, -10).mean()

            prop_ent[t] = prop_en

            log_p = log_probs # todo: milad, add back a summataion over the last dimension for normal sigmoid policy
            log_p_list.append(log_p)
            log_ps = torch.stack(log_p_list).permute((1, 0))
        prop_ent = prop_ent.mean()

        return log_ps, prop_ent, {'log_probs_lower_than_n10': log_probs_lower_than_n10}

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
            hidden_c, value_c = agent.critic(obs, h_0=hidden_c, t=t)
            hidden_t, value_t = agent.target(obs, h_0=hidden_t, t=t)
            values_c[:, t] = value_c.reshape(self.batch_size)
            values_t[:, t] = value_t.reshape(self.batch_size)
            hidden_c = hidden_c.permute((1, 0, 2))
            hidden_t = hidden_t.permute((1, 0, 2))
        return values_c, values_t

    def linear_aa_loss(self, A_1s, A_2s, log_ps, old_log_ps):
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        A_2s = A_2s[:, 1:]
        A_1s = torch.cumsum(A_1s[:, :-1], dim=1)
        log_ps = log_ps[:, 1:]
        old_log_ps = old_log_ps[:, 1:]

        batch, time = A_1s.shape
        A_1s = A_1s / (torch.arange(1, A_1s.shape[1] + 1).repeat(batch, 1).to(A_1s.device)+1e-6)  # todo: milad, is this 1e-6 necessary?

        if proximal:
            ratios = torch.exp(log_ps - old_log_ps.detach())
            clipped_log_ps = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            surrogates = torch.min(A_1s * A_2s * clipped_log_ps, A_1s * A_2s * log_ps)
            aa_loss = surrogates.mean()
        else:
            aa_loss = (A_1s * A_2s * log_ps).mean()

        return aa_loss

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
            print("Summing rewards, yes...")
            rewards_1 = rewards_2 = rewards_1 + rewards_2

        _, V_1s = self.get_trajectory_values(agent, observation_1)
        _, V_2s = self.get_trajectory_values(other_agent, observation_2)

        log_ps, prop_ent, log_ps_info = self.get_trajectory_log_ps(agent, trajectory, is_first)
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
            loss_2 = -self.linear_aa_loss(A_1s, A_2s, log_ps[:, :-1], old_log_ps[:, :-1])
        else:
            loss_2 = torch.zeros(1, device=loss_1.device)

        actor_loss_dict['loss'] = loss_1 + self.train_cfg.aa_weight * loss_2
        actor_loss_dict['loss_1'] = loss_1
        actor_loss_dict['loss_2'] = loss_2
        actor_loss_dict['log_ps_info'] = log_ps_info
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
        elif self.train_cfg.critic_loss_mode == 'interpolate':
            values_c_shifted = values_c[:, :-1]
            values_t_shifted = values_t[:, 1:]
            rewards_shifted = rewards_1[:, :-1]
            td_0_error = rewards_shifted + gamma * values_t_shifted
            discounted_returns = compute_discounted_returns(gamma, rewards_1)[:, :-1]
            critic_loss = F.huber_loss(values_c_shifted, (discounted_returns+td_0_error)/2)

        return critic_loss, {'target_values': values_t, 'critic_values': values_c, }

    def gen_sim(self):
        return self._gen_sim(self.env, self.batch_size, self.trajectory_len, self.agents)

    @staticmethod
    def _gen_sim(env, batch_size, trajectory_len, agents):
        B = int(batch_size[0])
        N = len(agents) 
        state = env.reset()
        history = deque(maxlen=agents[0].actor.model.encoder.max_seq_len)
        history_actions = deque(maxlen=agents[0].actor.model.encoder.max_seq_len)
        history_full_maps = deque(maxlen=agents[0].actor.model.encoder.max_seq_len)

        # Self-play only TODO: Add replay buffer of agents
        actors = [agents[0].actor.model for i in range(B*N)]

        for i in range(trajectory_len):

            full_maps = state['RGB']
            full_maps = full_maps.reshape(-1, 1, *full_maps.shape[2:])
            actions = torch.zeros((B*N, 1)).to(agents[0].device)
            observations = state['agents']['observation']['RGB']
            observations = observations.reshape(-1, 1, *observations.shape[2:])

            history.append(observations)
            history_actions.append(actions)
            history_full_maps.append(history_full_maps)

            observations = torch.cat(list(history), dim=1)
            actions = torch.cat(list(history_actions), dim=1)
            full_maps = torch.cat(list(full_maps), dim=1)
            
            action_logits = call_actors_forward(
                actors, 
                observations,
                actions, 
                agents[0].device
            )
            actions, log_probs = sample_actions_categorical(action_logits.reshape(B*N, -1))
            actions = actions.reshape(B, N)

            import pdb;pdb.set_trace()

        
        return trajectory

    def eval(self, agent):
        pass


def sample_actions_categorical(action_logits):
    distribution = Categorical(logits=action_logits)
    actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    return actions, log_probs


def call_actors_forward(actors, observations, actions, device):
    # Stack all agent parameters
    observations = observations.to(device).float()
    params, buffers = stack_module_state(actors)

    base_model = copy.deepcopy(actors[0])
    base_model = base_model.to('meta')

    def fmodel(params, buffers, x, actions):
        return functional_call(base_model, (params, buffers), (x, actions))

    logits_vmap_fmodel = torch.vmap(fmodel, randomness='different')(params, buffers, observations, actions.long())
    return logits_vmap_fmodel


@hydra.main(version_base="1.3", config_path="configs", config_name="meltingpot.yaml")
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
    if cfg['env']['type'] == 'meltingpot':
        scenario = cfg['env']['scenario']
        env = ParallelEnv(cfg['env']['batch'], lambda: MeltingpotEnv(scenario))
    else:
        raise ValueError(f"Environment type {cfg['env_conf']['type']} not supported.")

    if cfg['self_play']:
        agents = []
        agent = instantiate_agent(cfg)
        for i in range(cfg['env']['num_agents']):
            agents.append(agent)
    else:
        agents = []
        for i in range(cfg['env']['num_agents']):
            agents.append(instantiate_agent(cfg))

    algorithm = AdvantageAlignment(
        env=env,
        agents=agents,
        main_cfg=cfg,
        train_cfg=cfg.training,
        sum_rewards=cfg['sum_rewards'],
        simultaneous=cfg['simultaneous'],
        discrete=cfg['discrete'],
        clip_grad_norm=cfg['training']['clip_grad_norm'],
    )
    algorithm.train_loop()


if __name__ == "__main__":
    # test_fixed_agents()
    # test_eg_obligated_ratio()
    OmegaConf.register_new_resolver("eval", eval)
    main()
