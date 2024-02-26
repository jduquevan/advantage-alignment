import hydra
import math
import numpy as np
import random
import torch

from omegaconf import DictConfig

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cat_observations(observations, device):
    observations_cat = torch.cat([
        torch.tensor(observations['item_and_utility'], device=device),
        torch.tensor(observations['prop'], device=device),
        torch.tensor(observations['utte'], device=device)
    ], dim=1)
    return observations_cat

def get_categorical_entropy(log_ps):
    probs = torch.exp(log_ps)
    return ((probs * torch.log(probs)).sum(dim=1)).mean()

def get_gaussian_entropy(covariance_matrices):
    """
    Calculate entropy of a multivariate Gaussian distribution given its covariance matrix.
    """
    batch_size = covariance_matrices.size(0)
    dimension = covariance_matrices.size(-1)
    det_covariance = torch.det(covariance_matrices)
    entropy = 0.5 * (dimension * (torch.log(2 * torch.tensor(math.pi)) + 1) + torch.log(det_covariance))
    return entropy.mean()

def get_cosine_similarity(x1, x2, eps=1e-8):
    dot_product = np.sum(x1 * x2, axis=-1)
    norm1 = np.linalg.norm(x1, axis=-1)
    norm2 = np.linalg.norm(x2, axis=-1)
    cosine_sim = dot_product / (norm1 * norm2 + eps)
    return cosine_sim

def unit_vector_from_angles(theta, phi):
    # Convert angles to radians
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    
    # Calculate the components of the unit vector
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    
    # Return the unit vector
    return np.array([x, y, z])

def compute_discounted_returns(gamma, rewards, agent):
        batch_size, trajectory_len = rewards.shape
        returns = torch.empty(
            (batch_size, trajectory_len), 
            device=agent.device, 
            dtype=torch.float32
        )
        returns[:, -1] = rewards[:, -1]
        for t in reversed(range(trajectory_len-1)):
            returns[:, t] = rewards[:, t] + gamma * returns[:, t+1]
        return returns

def compute_gae_advantages(rewards, values, gamma=0.96, tau=0.95):
    # Compute deltas
    deltas = rewards[:, :-1] + gamma * values[:, 1:] - values[:, :-1]
    
    advantages = torch.empty_like(deltas)
    advantage = 0
    for t in reversed(range(deltas.size(1))):
        advantages[:, t] = advantage = deltas[:, t] + gamma * tau * advantage
    
    # Normalize advantages
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages

def torchify_actions(action, device):
    action_torch = {}
    action_torch['term'] = torch.tensor(action['term'], device=device)
    action_torch['utte'] = torch.tensor(action['utte'], device=device)
    action_torch['prop'] = torch.tensor(action['prop'], device=device)
    return action_torch

def instantiate_agent(cfg: DictConfig):
    if cfg.use_transformer:
        gru_model = None
        gru_target = None
        transformer = hydra.utils.instantiate(cfg.transformer_model)
        transformer_t = hydra.utils.instantiate(cfg.transformer_model)
    else:
        transformer = None
        transformer_t = None
        gru_model = hydra.utils.instantiate(cfg.gru_model)
        gru_target = hydra.utils.instantiate(cfg.gru_model)

    mlp_model = hydra.utils.instantiate(
        cfg.mlp_model, 
        gru=gru_model, 
        transformer=transformer
    )

    actor = hydra.utils.instantiate(cfg.policy, model=mlp_model)
    critic = hydra.utils.instantiate(
        cfg.linear_model, 
        gru=gru_model,
        transformer=transformer
    )

    target = hydra.utils.instantiate(
        cfg.linear_model, 
        gru=gru_target,
        transformer=transformer_t
    )

    optimizer_actor = hydra.utils.instantiate(cfg.optimizer_actor, params=actor.parameters())
    optimizer_critic = hydra.utils.instantiate(cfg.optimizer_critic, params=critic.parameters())
    
    agent = hydra.utils.instantiate(
        cfg.agent,
        actor=actor,
        critic=critic,
        target=target,
        critic_optimizer=optimizer_critic,
        actor_optimizer=optimizer_actor
    )
    return agent

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

def wandb_stats(trajectory):
    stats = {}
    dones = trajectory.data['dones']
    stats['Average reward 1'] = trajectory.data['rewards_0'].mean().detach()
    stats['Average reward 2'] = trajectory.data['rewards_1'].mean().detach()
    stats['Average sum rewards'] = (
        trajectory.data['rewards_0'] +
        trajectory.data['rewards_1']
    ).mean().detach()
    stats['Average traj len'] = ((dones.size(0) * dones.size(1))/torch.sum(dones)).detach()
    stats['Average cos sim'] = trajectory.data['cos_sims'].mean().detach()
    stats['Average A1 prop1'] = trajectory.data['props_0'].reshape((-1, 3)).mean(dim=0)[0]
    stats['Average A1 prop2'] = trajectory.data['props_0'].reshape((-1, 3)).mean(dim=0)[1]
    stats['Average A1 prop3'] = trajectory.data['props_0'].reshape((-1, 3)).mean(dim=0)[2]
    stats['Average A2 prop1'] = trajectory.data['props_1'].reshape((-1, 3)).mean(dim=0)[0]
    stats['Average A2 prop2'] = trajectory.data['props_1'].reshape((-1, 3)).mean(dim=0)[1]
    stats['Average A2 prop3'] = trajectory.data['props_1'].reshape((-1, 3)).mean(dim=0)[2]
    return stats