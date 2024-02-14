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
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages

def torchify_actions(action, device):
    action_torch = {}
    action_torch['term'] = torch.tensor(action['term'], device=device)
    action_torch['utte'] = torch.tensor(action['utte'], device=device)
    action_torch['prop'] = torch.tensor(action['prop'], device=device)
    return action_torch

def instantiate_agent(cfg: DictConfig):
    gru_model = hydra.utils.instantiate(cfg.gru_model)
    mlp_model = hydra.utils.instantiate(cfg.mlp_model, gru=gru_model)
    actor = hydra.utils.instantiate(cfg.policy, model=mlp_model)
    critic = hydra.utils.instantiate(cfg.linear_model, gru=gru_model)

    gru_target = hydra.utils.instantiate(cfg.gru_model)
    target = hydra.utils.instantiate(cfg.linear_model, gru=gru_target)

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
    stats['Average reward 1'] = trajectory.data['rewards_0'].mean().detach()
    stats['Average reward 2'] = trajectory.data['rewards_1'].mean().detach()
    return stats