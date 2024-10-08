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

    def to(self, device):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, nn.Module):
                setattr(self, attribute_name, attribute.to(device))

    def eval(self):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, nn.Module):
                attribute.eval()

    def train(self):
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, nn.Module):
                attribute.train()

    def load_state_dicts(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)  # Load the checkpoint
        for attribute_name in dir(self):
            # Check if the attribute is an nn.Module and the key exists in the checkpoint
            key = f"{attribute_name}_state_dict"
            if isinstance(getattr(self, attribute_name, None), nn.Module) and key in checkpoint:
                # Get the attribute and load the state dict
                getattr(self, attribute_name).load_state_dict(checkpoint[key])

    def save_checkpoint(self, path):
        checkpoint = {'epoch': self.epoch}
        for attribute_name in dir(self):
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, nn.Module):
                checkpoint[f'{attribute_name}_state_dict'] = attribute.state_dict()
        
        torch.save(checkpoint, path)
        return
    
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
        extend_history=False,
        is_first=False,
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

    
    def sample_f1_action(
        self, 
        observations,
        actions, 
        h_0=None, 
        history=None,
        history_actions=None,
        use_transformer=False, 
        extend_history=False,
        is_first=False,
    ):
        # TODO: Not tested
        if use_transformer:    
            observations = observations.unsqueeze(1)
            actions = actions.unsqueeze(1)
            if extend_history:
                history.append(observations)
                history_actions.append(actions)
            observations = torch.cat(list(history), dim=1)
            actions = torch.cat(list(history_actions), dim=1)

        if h_0 is not None:
            h_0 = torch.permute(h_0, (1, 0, 2))
        
        return self.actor.sample_action(observations, actions, h_0, use_transformer)


class OptimalCooperativeAgent(Agent):

    def __init__(self,
                 device,
                 actor: nn.Module,
                 critic: LinearModel,
                 target: nn.Module,
                 reward: nn.Module,
                 spr: nn.Module,
                 critic_optimizer: torch.optim.Optimizer,
                 actor_optimizer: torch.optim.Optimizer,
                 reward_optimizer: torch.optim.Optimizer,
                 spr_optimizer: torch.optim.Optimizer,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        is_first,
        h_0=None,
        history=None,
        use_transformer=False,
        extend_history=False,
    ):
        props = observations['prop']
        item_and_utility = observations['item_and_utility']
        assert item_and_utility.shape[1] == 12
        # this is the format: [number of  item 1, ..., number of item 3, utility item 1 for me, ..., utility item 3 for me, utility item 1 for you, ..., utility item 3 for you]
        # grab the utilities for me
        my_utilities = item_and_utility[:, 3:6]
        your_utilities = item_and_utility[:, 9:12]
        item_counts = item_and_utility[:, 0:3]
        assert torch.allclose(item_counts, item_and_utility[:, 6:9]) # check that the counts are the same, as they should be, they are copy paste of each other

        if is_first:
            ans = torch.where(my_utilities >= your_utilities, item_counts, 0)
        else:
            ans = torch.where(my_utilities > your_utilities, item_counts, 0)

        log_p = torch.tensor(0.).to(self.device)

        return None, {'prop': ans}, log_p



