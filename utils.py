from pathlib import Path

import hydra
import numpy as np
import os
import random
import torch
import wandb

from omegaconf import DictConfig


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_discounted_returns(gamma, rewards):
        batch_size, trajectory_len = rewards.shape
        returns = torch.empty(
            (batch_size, trajectory_len), 
            device=rewards.device,
            dtype=torch.float32
        )
        returns[:, -1] = rewards[:, -1]
        for t in reversed(range(trajectory_len-1)):
            returns[:, t] = rewards[:, t] + gamma * returns[:, t+1]
        return returns

def compute_gae_advantages(rewards, values, gamma=0.96, tau=0.95):
    # Compute deltas
    with torch.no_grad():
        deltas = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
        
        advantages = torch.empty_like(deltas)
        advantage = 0
        for t in reversed(range(deltas.size(1))):
            advantages[:, t] = advantage = deltas[:, t] + gamma * tau * advantage
        
        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages

def instantiate_agent(cfg: DictConfig):
    assert cfg.use_transformer is True, "Only Transformer is supported for now."
    use_gru = True
    
    encoder = hydra.utils.instantiate(cfg.encoder)
    actor = hydra.utils.instantiate(
        cfg.mlp_model,
        encoder=encoder,
    )
    
    if not cfg.share_encoder:
        encoder_c = hydra.utils.instantiate(cfg.encoder)
        critic = hydra.utils.instantiate(
            cfg.linear_model, 
            encoder=encoder_c,
        )
    else:
        critic = hydra.utils.instantiate(
            cfg.linear_model,
            encoder=encoder,
        )

    encoder_t = hydra.utils.instantiate(cfg.encoder)
    target = hydra.utils.instantiate(
        cfg.linear_model, 
        encoder=encoder_t,
    )

    ss_module = hydra.utils.instantiate(
        cfg.ss_module,
        encoder=encoder,
    )

    target.load_state_dict(critic.state_dict())

    optimizer_actor = hydra.utils.instantiate(cfg.optimizer_actor, params=actor.parameters())
    optimizer_critic = hydra.utils.instantiate(cfg.optimizer_critic, params=critic.parameters())
    optimizer_ss = hydra.utils.instantiate(cfg.optimizer_ss, params=ss_module.parameters())

    agent = hydra.utils.instantiate(
        cfg.agent,
        actor=actor,
        critic=critic,
        target=target,
        ss_module=ss_module,
        critic_optimizer=optimizer_critic,
        actor_optimizer=optimizer_actor,
        ss_optimizer=optimizer_ss,
    )
    return agent

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def save_checkpoint(agent, replay_buffer, agent_replay_buffer, step, checkpoint_dir):
    # Get the W&B run ID
    run_id = wandb.run.id

    from train import ReplayBuffer, AgentReplayBuffer
    replay_buffer: ReplayBuffer
    agent_replay_buffer: AgentReplayBuffer

    # Create a folder with the run ID if it doesn't exist
    run_folder = Path(checkpoint_dir) / run_id
    os.makedirs(run_folder, exist_ok=True)

    # Generate a unique filename
    checkpoint_path = run_folder / ('step_'+str(step))
    os.makedirs(checkpoint_path, exist_ok=True)

    agent_checkpoint = {
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'target_state_dict': agent.target.state_dict(),
    }

    torch.save(agent_checkpoint, checkpoint_path / 'agent.pt')
    if replay_buffer is not None:
        torch.save(replay_buffer.trajectory_batch.data, checkpoint_path / 'replay_buffer_data.pt')
    if agent_replay_buffer is not None:
        agents_state_dict = [agent_state_dict for agent_state_dict in agent_replay_buffer.agents if agent_state_dict is not None]
        torch.save(agents_state_dict, checkpoint_path / 'agent_replay_buffer_state_dicts.pt')
    print(f"(@@-o)Agent Checkpoint saved at: {checkpoint_path}")


def wandb_stats(trajectory, B, N):
    stats = {}
    stats['Average reward'] = trajectory.data['rewards'].mean().detach()
    stats['Average sum rewards'] = torch.sum(
        trajectory.data['rewards'].reshape(B, N, -1), dim=1
    ).unsqueeze(1).repeat(1, N, 1).mean().detach()
    return stats


from omegaconf import OmegaConf


def flatten_native_dict(cfg, parent_key='', sep='.'):
    """
    Recursively flattens a Python dictionary (or list) into a list of key-value pairs.
    The keys will be in dot notation.

    Args:
        cfg: Python dict or list to flatten.
        parent_key: Current key prefix (used in recursion).
        sep: Separator between keys (default is a dot).

    Returns:
        A list of tuples containing flattened key-value pairs.
    """
    items = []

    if isinstance(cfg, dict):
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(flatten_native_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (dict, list)):
                items.extend(flatten_native_dict(v, new_key, sep=sep))
            else:
                items.append((new_key, v))

    return items