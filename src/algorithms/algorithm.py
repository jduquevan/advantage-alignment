from __future__ import annotations

import time
import torch
import wandb
import gym

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.agents.agents import Agent
from src.utils.utils import update_target_network, wandb_stats
from tqdm import tqdm
from torch import nn
from typing import List

class TrainingAlgorithm(ABC):
    def __init__(
        self,
        env: gym.Env,
        agent_1: Agent,
        agent_2: Agent,
        train_cfg: DictConfig,
        use_transformer: bool,
        sum_rewards: bool,
    ) -> None:
        self.env = env
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.train_cfg = train_cfg
        self.batch_size = (len(env.envs))
        self.trajectory_len = train_cfg.max_traj_len
        self.use_transformer = use_transformer
        self.sum_rewards = sum_rewards

    def train_step(
        self,
        trajectory,
        agent,
        is_first=True
    ):
        agent.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(trajectory, is_first)
        critic_loss.backward()
        agent.critic_optimizer.step()

        agent.actor_optimizer.zero_grad()
        actor_loss, term_ent, utte_ent, prop_ent = self.actor_loss(trajectory, is_first)
        total_loss = actor_loss + self.train_cfg.entropy_beta * term_ent
        total_loss.backward()
        agent.actor_optimizer.step()

        update_target_network(agent.target, agent.critic, self.train_cfg.tau)
        return (
            critic_loss.detach(), 
            actor_loss.detach(), 
            term_ent.detach(), 
            utte_ent.detach(), 
            prop_ent.detach()
        )

    def eval(self, step: int, start_time: float):
        eval_res = test_agent(
            self.env,
            self.network
        )
        eval_res["Step"] = step
        eval_res["Time (s)"] = time.time() - start_time

        # Log
        tqdm.write(f"Eval results: {eval_res}")
        wandb.log(eval_res)

    def save(self):
        pass

    def train_loop(self):
        print("Generating initial trajectories")
        start_time = time.time()

        pbar = tqdm(range(self.train_cfg.total_steps))
        for step in pbar:
            # # Eval
            # if step % self.train_cfg.eval_every == 0:
            #     self.network.eval()
            #     self.eval(step, start_time)
            #     self.network.train()

            # Collect trajectories
            trajectory = self.gen()
            wandb.log(wandb_stats(trajectory))

            # Train
            critic_loss_1, actor_loss_1, term_ent_1, utte_ent_1, prop_ent_1 = self.train_step(
                trajectory=trajectory,
                agent=self.agent_1,
                is_first=True
            )
            print("Critic loss Agent 1: ", critic_loss_1)
            print("Actor loss Agent 1: ", actor_loss_1)
            print("Term Entropy Agent 1: ", term_ent_1)
            print("Utte Entropy Agent 1: ", utte_ent_1)
            print("Prop Entropy Agent 1: ", prop_ent_1)
            # print(trajectory.data['obs_0'][0, 0:20])
            print()

            critic_loss_2, actor_loss_2, term_ent_2, utte_ent_2, prop_ent_2 = self.train_step(
                trajectory=trajectory,
                agent=self.agent_2,
                is_first=False
            )
            print("Critic loss Agent 2: ", critic_loss_2)
            print("Actor loss Agent 2: ", actor_loss_2)
            print("Term Entropy Agent 2: ", term_ent_2)
            print("Utte Entropy Agent 2: ", utte_ent_2)
            print("Prop Entropy Agent 2: ", prop_ent_2)
            # print(trajectory.data['obs_1'][0, 0:20])
            print()
            wandb.log({
                "Actor loss 1": actor_loss_1, 
                "Critic loss 1": critic_loss_1,
                "Actor loss 2": actor_loss_2,
                "Critic loss 2": critic_loss_2,
                "Term Entropy 1": term_ent_1,
                "Utte Entropy 1": utte_ent_1,
                "Prop Entropy 1": prop_ent_1,
                "Term Entropy 2": term_ent_2,
                "Utte Entropy 2": utte_ent_2,
                "Prop Entropy 2": prop_ent_2,
                })
        return

    """ TO BE DEFINED BY INDIVIDUAL ALGORITHMS"""

    @abstractmethod
    def actor_loss(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_loss(batch: dict) -> float:
        pass

    @abstractmethod
    def gen(self, num_samples: int, initial=False):
        pass