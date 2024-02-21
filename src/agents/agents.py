from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from omegaconf import DictConfig
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
        critic: nn.Module,
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
    
    def sample_action(self, observations, h_0=None, history=None, use_transformer=False):
        observations = cat_observations(observations, self.device)
        
        if use_transformer:
            observations = observations.unsqueeze(0)
            history.append(observations)
            observations = torch.cat(list(history), dim=0)

        if h_0 is not None:
            h_0 = torch.permute(h_0, (1, 0, 2))
        
        return self.actor.sample_action(observations, h_0)


