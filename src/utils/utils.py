import hydra
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
    # hydra.utils.instantiate(cfg.wandb)
    return agent

def update_target_network(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)