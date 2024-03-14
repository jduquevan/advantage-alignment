from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.models.models import LinearModel
from src.utils.utils import cat_observations
from torch import nn

class Agent(ABC):
    @abstractmethod
    def sample_action(self):
        pass

class AdvantageAlignmentAgent(Agent):
    def __init__(
        self,
        device: torch.device,
        actor: nn.Module,
        critic: LinearModel,
        target: nn.Module,
        reward: nn.Module,
        spr: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        actor_optimizer: torch.optim.Optimizer,
        reward_optimizer: torch.optim.Optimizer,
        spr_optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()
        self.device = device
        self.actor = actor
        self.critic = critic
        self.target = target
        self.reward = reward
        self.spr = spr
        self.critic_optimizer = critic_optimizer
        self.actor_optimizer = actor_optimizer
        self.reward_optimizer = reward_optimizer
        self.spr_optimizer = spr_optimizer
    
    def sample_action(
        self, 
        observations, 
        h_0=None, 
        history=None, 
        use_transformer=False, 
        extend_history=False
    ):
        props = observations['prop']
        item_and_utility = observations['item_and_utility']
        # observations = cat_observations(observations, self.device)
        
        if use_transformer:
            props = props.unsqueeze(0)
            if extend_history:
                history.append(props)
            props = torch.cat(list(history), dim=0)

        if h_0 is not None:
            h_0 = torch.permute(h_0, (1, 0, 2))

        observations = {
            'prop': props,
            'item_and_utility': item_and_utility
        }
        
        return self.actor.sample_action(observations, h_0, use_transformer)


