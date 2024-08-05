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
    save_checkpoint
)
from tensordict import TensorDict
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

    def forward(self, obs, actions, full_seq: int=1, batched: bool=False):
        if full_seq==0:
            output = self.encoder(obs, actions, batched)[:, -1, :]
        elif full_seq==1:
            output = self.encoder(obs, actions, batched)[:,::2, :]
        elif full_seq==2:
            output = self.encoder(obs, actions, batched)[:,1::2, :]
        else:
           raise NotImplementedError("The full_seq value is invalid")

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
        self.embed_action = nn.Embedding(num_embeddings=8, embedding_dim=d_model)

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
    
    def forward(self, obs, actions, batched=False):
        src = self.get_embeds(obs, actions, batched)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        output = src
        for layer in self.layers:
            output = layer(output) + src
        return output

    def get_embeds(self, obs, actions, batched=False):
        if batched:
            B, T, H, W, C = obs.shape
            obs = obs.permute(0, 1, 4, 2, 3).reshape(B*T, C, H, W)
        else:
            T, _, _, _ = obs.shape
            obs = obs.permute(0, 3, 1, 2)
        H = self.d_model
        
        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        if batched:
            obs = obs.reshape(B*T, -1)
            actions = actions.reshape(B*T)
        else:
            obs = obs.reshape(T, -1)
        embed_obs = F.relu(self.embed_obs_layer(obs))
        embed_actions = self.embed_action(actions)

        for embed_layer in self.embed_layers:
            embed_obs = F.relu(embed_layer(embed_obs))
            embed_actions = F.relu(embed_layer(embed_actions))

        # Interleave states and actions: a, s, a, ...
        if batched:
            obs = obs.reshape((B, T, -1))
            actions = actions.reshape((B, T))
            src = torch.stack((embed_actions, embed_obs), dim=2).view(B, -1, H)
        else:
            src = torch.stack((embed_actions, embed_obs), dim=2).view(1, -1, H)

        return src


class Agent(ABC):
    @abstractmethod
    def sample_action(self):
        pass


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
    def __init__(self, 
                 sample_obs, 
                 sample_full_map, 
                 batch_size: int,
                 n_agents: int, 
                 max_traj_len: int, 
                 device: torch.device, 
                 data=None
                 ) -> None:
        
        self.batch_size = batch_size
        self.device = device
        
        B, N, H, W, C = sample_obs.shape
        _, G, L, _ = sample_full_map.shape
        T = max_traj_len

        if data is None:
            self.data = {
                "obs": torch.ones(
                    (B*N, T, H, W, C),
                    dtype=torch.float32,
                    device=device,
                ),
                "full_maps": torch.ones(
                    (B, T, G, L, C),
                    dtype=torch.float32,
                    device=device,
                ),
                "rewards": torch.ones(
                    (B*N, T),
                    device=device,
                    dtype=torch.float32,
                ),
                "log_ps": torch.ones(
                    (B*N, T),
                    device=device,
                    dtype=torch.float32,
                ),
                "actions": torch.ones(
                    (B*N, T),
                    device=device,
                    dtype=torch.long,
                ),
            }

    def subsample(self, batch_size: int):
        idxs = torch.randint(0, self.batch_size, (batch_size,))
        return TrajectoryBatch.create_from_data({k: v[idxs] for k, v in self.data.items()})

    def add_step(self, obs, full_maps, rewards, log_ps, actions, t):
        self.data['obs'][:, t, :, :, :] = obs[:, 0, :, :, :]
        self.data['full_maps'][:, t, :, :, :] = full_maps[:, 0, :, :, :]
        self.data['rewards'][:, t] = rewards
        self.data['log_ps'][:, t] = log_ps
        self.data['actions'][:, t] = actions[:, 0]

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
        self.clip_grad_norm = clip_grad_norm
        self.num_agents = len(agents)
        self.batch_size = int(env.batch_size[0])

    def train_step(
        self,
        trajectory,
        agents,
        optimize_critic=True,
    ):
        train_step_metrics = {}
        self_play = self.main_cfg['self_play']

        if self_play:
            agent = agents[0]
            agent.actor_optimizer.zero_grad()
        else:
            # TODO: Implement no self-play
            raise NotImplementedError('Only self-play supported')

        # --- optimize actor ---
        actor_loss_dict = self.actor_losses(trajectory)

        if self_play:
            actor_loss = actor_loss_dict['losses'].mean()
            ent = actor_loss_dict['entropy'].mean()
            actor_loss_1 = actor_loss_dict['losses_1'].mean()
            actor_loss_2 = actor_loss_dict['losses_2'].mean()
            loss_1_grads = torch.autograd.grad(actor_loss_1, agents[0].actor.parameters(), create_graph=False, retain_graph=True)
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
        if self_play:
            agent.critic_optimizer.zero_grad()
            critic_losses, aux = self.critic_losses(trajectory)
            critic_loss = critic_losses.mean()
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
                "entropy": ent.detach(),
                "critic_values": critic_values.mean().detach(),
                "target_values": target_values.mean().detach(),
            })
            train_step_metrics["critic_grad_norm"] = critic_grad_norm.detach()
            train_step_metrics["actor_grad_norm"] = actor_grad_norm.detach()

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
            
            wandb_metrics = wandb_stats(trajectory, self.batch_size, self.num_agents)

            for i in range(self.train_cfg.updates_per_batch):

                # augmented_trajectories = trajectory # do not augment with off-policy data, as we don't have a replay buffer

                train_step_metrics= self.train_step(
                    trajectory=trajectory,
                    agents=self.agents,
                    optimize_critic=True
                )

                for key, value in train_step_metrics.items():
                    print(key + ":", value)
                print()

            wandb_metrics.update(train_step_metrics)
            print(f"train step metrics: {train_step_metrics}")

            # eval_metrics = self.eval(self.agents[0])
            # print("Evaluation metrics:", eval_metrics)
            # wandb_metrics.update(eval_metrics)

            agent_actor_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agents[0].actor.parameters()]))
            agent_critic_weight_norms = torch.sqrt(sum([torch.norm(p.data) ** 2 for p in self.agents[0].critic.parameters()]))

            wandb_metrics.update({
                "agent_actor_weight_norms": agent_actor_weight_norms.detach(),
                "agent_critic_weight_norms": agent_critic_weight_norms.detach()
            })
            wandb.log(step=step, data=wandb_metrics)

        return

    """ TO BE DEFINED BY INDIVIDUAL ALGORITHMS"""

    @abstractmethod
    def actor_losses(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_losses(batch: dict) -> float:
        pass


class AdvantageAlignment(TrainingAlgorithm):

    def get_trajectory_log_ps(self, agents, observations, actions):  # todo: make sure logps are the same as when sampling
        B = self.batch_size
        N = self.num_agents
        T = self.trajectory_len
        device = agents[0].device 
        
        action_logits = agents[0].actor(
            observations,
            actions,
            full_seq=2,
            batched=True
        ).reshape((B*N*T, -1))

        _, log_ps = sample_actions_categorical(action_logits, actions.reshape(B*N*T))
        entropy = get_categorical_entropy(action_logits)

        return log_ps.reshape((B*N, T)), entropy.reshape((B*N, T))

    def get_trajectory_values(self, agents, observations, actions):
        B = self.batch_size
        N = self.num_agents
        device = agents[0].device 
        
        values_c = agents[0].critic(
            observations,
            actions,
            full_seq=1,
            batched=True
        ).squeeze(2)
        values_t = agents[0].target(
            observations,
            actions,
            full_seq=1,
            batched=True
        ).squeeze(2)

        return values_c, values_t

    def aa_loss(self, A_s, log_ps, old_log_ps):
        B = self.batch_size
        N = self.num_agents
        device = A_s.device
        gamma = self.train_cfg.gamma
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        A_1s = torch.cumsum(A_s[:, :-1], dim=1)
        A_s = A_s[:, 1:]
        log_ps = log_ps[:, 1:]
        old_log_ps = old_log_ps[:, 1:]

        def other_A_s(A_s, N):
            mask = torch.ones((N, N), device=device) - torch.eye(N, device=device)
            return mask@A_s

        A_2s = torch.vmap(other_A_s, in_dims=(0, None))(A_s.view(B, N, -1), N).reshape((B*N, -1))
        
        _, T = A_1s.shape
        gammas = torch.pow(gamma, torch.arange(T)).repeat(B*N, 1).to(device)

        if proximal:
            ratios = torch.exp(log_ps - old_log_ps.detach())
            clipped_log_ps = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            surrogates = torch.min(A_1s * A_2s * clipped_log_ps * gammas, A_1s * A_2s * log_ps * gammas)
            aa_loss = surrogates.view((B, N, -1)).mean(dim=(0, 2))
        else:
            aa_loss = (A_1s * A_2s * log_ps * gammas).view((B, N, -1)).mean(dim=(0, 2))

        return aa_loss

    def actor_losses(self, trajectory: TrajectoryBatch) -> float:
        B = self.batch_size
        N = self.num_agents

        actor_loss_dict = {}
        gamma = self.train_cfg.gamma
        vanilla = self.train_cfg.vanilla
        proximal = self.train_cfg.proximal
        clip_range = self.train_cfg.clip_range

        obs = trajectory.data['obs']
        full_maps = trajectory.data['full_maps']
        rewards = trajectory.data['rewards']
        old_log_ps = trajectory.data['log_ps']
        actions = trajectory.data['actions']
        
        if self.sum_rewards:
            print("Summing rewards...")
            rewards = torch.sum(rewards.reshape(B, N, -1), dim=1).unsqueeze(1).repeat(1, N, 1).reshape((B*N, -1))

        _, V_s = self.get_trajectory_values(self.agents, obs, actions)

        log_ps, entropy = self.get_trajectory_log_ps(self.agents, obs, actions)
        actor_loss_dict['entropy'] = entropy.view((B, N, -1)).mean(dim=(0, 2))

        A_s = compute_gae_advantages(rewards, V_s.detach(), gamma)

        if proximal:
            ratios = torch.exp(log_ps - old_log_ps.detach())
            clipped_log_ps = torch.clamp(ratios, 1 - clip_range, 1 + clip_range)
            surrogates = torch.min(A_s * clipped_log_ps[:, :-1], A_s * log_ps[:, :-1])
            losses_1 = -surrogates.view((B, N, -1)).mean(dim=(0, 2))
        else:
            losses_1 = -(A_s * log_ps[:, :-1]).view((B, N, -1)).mean(dim=(0, 2))

        if not vanilla:
            losses_2 = -self.aa_loss(A_s, log_ps[:, :-1], old_log_ps[:, :-1])
        else:
            losses_2 = torch.zeros(N, device=losses_1.device)

        actor_loss_dict['losses'] = losses_1 + self.train_cfg.aa_weight * losses_2
        actor_loss_dict['losses_1'] = losses_1
        actor_loss_dict['losses_2'] = losses_2

        return actor_loss_dict

    def critic_losses(self, trajectory: TrajectoryBatch) -> float:
        B = self.batch_size
        N = self.num_agents

        gamma = self.train_cfg.gamma

        obs = trajectory.data['obs']
        full_maps = trajectory.data['full_maps']
        rewards = trajectory.data['rewards']
        actions = trajectory.data['actions']

        if self.sum_rewards:
            print("Summing rewards...")
            rewards = torch.sum(rewards.reshape(B, N, -1), dim=1).unsqueeze(1).repeat(1, N, 1).reshape((B*N, -1))

        values_c, values_t = self.get_trajectory_values(self.agents, obs, actions)
        if self.train_cfg.critic_loss_mode == 'td-1':
            values_c_shifted = values_c[:, :-1]
            values_t_shifted = values_t[:, 1:]
            rewards_shifted = rewards[:, :-1]
            critic_loss = F.huber_loss(values_c_shifted, rewards_shifted + gamma * values_t_shifted, reduction='none')
        elif self.train_cfg.critic_loss_mode == 'MC':
            discounted_returns = compute_discounted_returns(gamma, rewards)
            critic_loss = F.huber_loss(values_c, discounted_returns, reduction='none')
        elif self.train_cfg.critic_loss_mode == 'interpolate':
            values_c_shifted = values_c[:, :-1]
            values_t_shifted = values_t[:, 1:]
            rewards_shifted = rewards_1[:, :-1]
            td_0_error = rewards_shifted + gamma * values_t_shifted
            discounted_returns = compute_discounted_returns(gamma, rewards)[:, :-1]
            critic_loss = F.huber_loss(values_c_shifted, (discounted_returns+td_0_error)/2, reduction='none')

        critic_losses = critic_loss.view((B, N, -1)).mean(dim=(0, 2))

        return critic_losses, {'target_values': values_t, 'critic_values': values_c, }

    def gen_sim(self):
        return self._gen_sim(
            self.env, 
            self.batch_size, 
            self.trajectory_len, 
            self.agents
        )

    @staticmethod
    def _gen_sim(env, batch_size, trajectory_len, agents):
        B = batch_size
        N = len(agents)
        device = agents[0].device 
        state = env.reset()
        history = deque(maxlen=trajectory_len)
        history_actions = deque(maxlen=trajectory_len)
        history_full_maps = deque(maxlen=trajectory_len)

        # Self-play only TODO: Add replay buffer of agents
        actors = [agents[i%N].actor for i in range(B*N)]

        trajectory = TrajectoryBatch(
            sample_obs=state['agents']['observation']['RGB'],
            sample_full_map=state['RGB'],
            batch_size=B,
            n_agents=N,
            max_traj_len=trajectory_len,
            device=device
        )

        rewards = torch.zeros((B*N)).to(device)
        log_probs = torch.zeros((B*N)).to(device)
        actions = torch.zeros((B*N, 1)).to(device)

        for i in range(trajectory_len):
            full_maps = state['RGB']
            full_maps = full_maps.unsqueeze(1)
            
            observations = state['agents']['observation']['RGB']
            observations = observations.reshape(-1, 1, *observations.shape[2:])

            trajectory.add_step(observations, full_maps, rewards, log_probs.detach(), actions, i)

            history.append(observations)
            history_actions.append(actions)
            history_full_maps.append(history_full_maps)

            observations = torch.cat(list(history), dim=1)
            actions = torch.cat(list(history_actions), dim=1)
            full_maps = torch.cat(list(full_maps), dim=1)
            
            action_logits = call_models_forward(
                actors,
                observations,
                actions, 
                agents[0].device
            )
            actions, log_probs = sample_actions_categorical(action_logits.reshape(B*N, -1))
            actions = actions.reshape(B*N, 1)
            actions_dict = TensorDict(
                source={'agents': TensorDict(source={'action': actions.reshape(B, N)})}, 
                batch_size=[B]
            )

            state = env.step(actions_dict)['next']
            rewards = state['agents']['reward'].reshape(B*N)
        
        return trajectory

    def eval(self, agent):
        pass


def sample_actions_categorical(action_logits, actions=None):
    distribution = Categorical(logits=action_logits)
    if actions is None:
        actions = distribution.sample()
    log_probs = distribution.log_prob(actions)
    return actions, log_probs


def get_categorical_entropy(action_logits):
    distribution = Categorical(logits=action_logits)
    return distribution.entropy()


def call_models_forward(models, observations, actions, device, full_seq=0):
    # Stack all agent parameters
    observations = observations.to(device).float()
    params, buffers = stack_module_state(models)

    base_model = copy.deepcopy(models[0])
    base_model = base_model.to('meta')

    def fmodel(params, buffers, x, actions, full_seq):
        return functional_call(base_model, (params, buffers), (x, actions, full_seq))

    vmap_fmodel = torch.vmap(fmodel, in_dims=(0, 0, 0, 0, None), randomness='different')(
        params, buffers, observations, actions.long(), full_seq)
    return vmap_fmodel


@hydra.main(version_base="1.3", config_path="configs", config_name="meltingpot.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    seed_all(cfg['seed'])
    wandb.init(
        config=dict(cfg),
        project="Meltingpot",
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
    OmegaConf.register_new_resolver("eval", eval)
    main()
