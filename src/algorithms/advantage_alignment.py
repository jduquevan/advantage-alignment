from __future__ import annotations
import torch
import math

from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from src.algorithms.algorithm import TrainingAlgorithm
from src.data.trajectory import TrajectoryBatch
from src.utils.utils import cat_observations, torchify_actions

class AdvantageAlignment(TrainingAlgorithm):
    
    def actor_loss(self, trajectory: TrajectoryBatch) -> float:
        pass

    def critic_loss(self, trajectory: TrajectoryBatch) -> float:
        import pdb; pdb.set_trace()
        pass
        
    def gen(self):
        observations, _ = self.env.reset()
        hidden_state_1, hidden_state_2 = None, None
        trajectory = TrajectoryBatch(
            batch_size=len(self.env.envs) , 
            max_traj_len=self.train_cfg.max_traj_len,
            device=self.agent_1.device
        )

        for t in range(self.train_cfg.max_traj_len * 2):
            if t % 2 == 0:
                hidden_state_1, action, log_p_1 = self.agent_1.sample_action(
                    observations=observations,
                    h_0=hidden_state_1
                )
            else:
                hidden_state_2, action, log_p_2 = self.agent_2.sample_action(
                    observations=observations,
                    h_0=hidden_state_2
                )
            observations, rewards, _, _, _ = self.env.step(action)
            trajectory.add_step(
                action=torchify_actions(action, self.agent_1.device), 
                observations=cat_observations(observations, self.agent_1.device), 
                rewards=torch.tensor(rewards, device=self.agent_1.device), 
                t=t
            )
        trajectory.add_step(
            action=torchify_actions(action, self.agent_1.device), 
            observations=cat_observations(observations, self.agent_1.device), 
            rewards=torch.tensor(rewards, device=self.agent_1.device), 
            t=t
        )
        return trajectory