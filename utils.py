import hydra
import math
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

def get_observation(observations, t):
    return observations[:, t, :]

def get_cosine_similarity_torch(x1, x2, eps=1e-8):
    dot_product = torch.sum(x1 * x2, dim=-1)
    norm1 = torch.norm(x1.float(), dim=-1)
    norm2 = torch.norm(x2.float(), dim=-1)
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    return cosine_sim

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
    encoder_in_size = 23232
    
    encoder = hydra.utils.instantiate(cfg.encoder, in_size=encoder_in_size)
    actor = hydra.utils.instantiate(
        cfg.mlp_model,
        encoder=encoder,
    )
    
    if not cfg.share_encoder:
        encoder_c = hydra.utils.instantiate(cfg.encoder, in_size=encoder_in_size)
        critic = hydra.utils.instantiate(
            cfg.linear_model, 
            encoder=encoder_c,
        )
    else:
        critic = hydra.utils.instantiate(
            cfg.linear_model,
            encoder=encoder,
        )

    encoder_t = hydra.utils.instantiate(cfg.encoder, in_size=encoder_in_size)
    target = hydra.utils.instantiate(
        cfg.linear_model, 
        encoder=encoder_t,
    )

    target.load_state_dict(critic.state_dict())

    optimizer_actor = hydra.utils.instantiate(cfg.optimizer_actor, params=actor.parameters())
    optimizer_critic = hydra.utils.instantiate(cfg.optimizer_critic, params=critic.parameters())

    agent = hydra.utils.instantiate(
        cfg.agent,
        actor=actor,
        critic=critic,
        target=target,
        critic_optimizer=optimizer_critic,
        actor_optimizer=optimizer_actor,
    )
    return agent

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def save_checkpoint(agent, epoch, checkpoint_dir):
    # Get the W&B run ID
    run_id = wandb.run.id

    # Create a folder with the run ID if it doesn't exist
    run_folder = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)

    # Generate a unique filename
    checkpoint_name = f"model_{epoch}.pt"
    checkpoint_path = os.path.join(run_folder, checkpoint_name)
    
    checkpoint = {
        'epoch': epoch,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def wandb_stats(trajectory, B, N):
    stats = {}
    stats['Average reward'] = trajectory.data['rewards'].mean().detach()
    stats['Average sum rewards'] = torch.sum(
        trajectory.data['rewards'].reshape(B, N, -1), dim=1
    ).unsqueeze(1).repeat(1, N, 1).mean().detach()
    return stats