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

    def forward(self, obs, actions, full_seq: bool=False):
        if full_seq==0:
            output = self.encoder(obs, actions)[:, -1, :]
        elif full_seq==1:
            output = self.encoder(obs, actions)[:,::2, :]
        elif full_seq==2:
            output = self.encoder(obs, actions)[:,1::2, :]
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
                    dtype=torch.float32,
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
        # agent.actor_optimizer.zero_grad()
        actor_loss_dict = self.actor_losses(trajectory)
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
            
            # wandb_metrics = wandb_stats(trajectory)

            for i in range(self.train_cfg.updates_per_batch):

                # augmented_trajectories = trajectory # do not augment with off-policy data, as we don't have a replay buffer

                train_step_metrics_1 = self.train_step(
                    trajectory=trajectory,
                    agent=self.agents,
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
    def actor_losses(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_loss(batch: dict) -> float:
        pass


class AdvantageAlignment(TrainingAlgorithm):

    def get_trajectory_log_ps(self, agents, observations, actions):  # todo: make sure logps are the same as when sampling
        B = int(self.batch_size[0])
        N = len(agents)
        T = self.trajectory_len
        device = agents[0].device 
        actors = [agents[i%N].actor for i in range(B*N)]

        action_logits = call_models_forward(
            actors, 
            observations,
            actions, 
            device,
            full_seq=2
        ).reshape((B*N*T, -1))

        _, log_ps = sample_actions_categorical(action_logits, actions.reshape(B*N*T))
        entropy = get_categorical_entropy(action_logits)

        return log_ps.reshape((B*N, T)), entropy.reshape((B*N, T))

    def get_trajectory_values(self, agents, observations, actions):
        B = int(self.batch_size[0])
        N = len(agents)
        device = agents[0].device 
        critics = [agents[i%N].critic for i in range(B*N)]
        targets = [agents[i%N].target for i in range(B*N)]

        values_c = call_models_forward(
            critics,
            observations,
            actions,
            device,
            full_seq=1
        ).reshape((B*N, -1))
        values_t = call_models_forward(
            targets,
            observations,
            actions,
            device,
            full_seq=1
        ).reshape((B*N, -1))

        return values_c, values_t

    def aa_loss(self, A_s, log_ps, old_log_ps):
        B = int(self.batch_size[0])
        N = len(self.agents)
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

    def actor_losses(self, trajectory: TrajectoryBatch) -> float:
        B = int(self.batch_size[0])
        N = len(self.agents)

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
            torch.sum(rewards, dim=0).repeat((B*N, 1))

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
            losses_2 = torch.zeros(N, device=loss_1.device)

        actor_loss_dict['losses'] = losses_1 + self.train_cfg.aa_weight * losses_2
        actor_loss_dict['losses_1'] = losses_1
        actor_loss_dict['losses_2'] = losses_2
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

            trajectory.add_step(observations, full_maps, rewards, log_probs, actions, i)

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
