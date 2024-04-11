from __future__ import annotations

import gym
import time
import torch
import wandb

from abc import ABC, abstractmethod
from omegaconf import DictConfig
from src.agents.agents import Agent
from src.utils.utils import (
    update_target_network,
    wandb_stats,
    save_checkpoint,
    merge_train_step_metrics
)
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
        simultaneous: bool,
        discrete: bool,
        use_reward_loss: bool,
        use_spr_loss: bool,
        is_f1: bool, 
    ) -> None:
        self.env = env
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.train_cfg = train_cfg
        self.trajectory_len = train_cfg.max_traj_len
        self.use_transformer = use_transformer
        self.sum_rewards = sum_rewards
        self.simultaneous = simultaneous
        self.discrete = discrete
        self.use_reward_loss = use_reward_loss
        self.use_spr_loss = use_spr_loss
        self.is_f1 = is_f1
        if self.is_f1:
            self.batch_size = len(env.env_fns)
        elif self.simultaneous:
            self.batch_size = env.batch_size

    def train_step(
        self,
        trajectory,
        agent,
        is_first=True
    ):
        agent.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(trajectory, is_first)
        critic_loss.backward()
        critic_grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in agent.critic.parameters()]))
        agent.critic_optimizer.step()

        agent.actor_optimizer.zero_grad()
        if self.is_f1:
            actor_loss_dict = self.actor_loss(trajectory, is_first)
            actor_loss = actor_loss_dict['loss']
            ent = actor_loss_dict['acc_ent'] + actor_loss_dict['ste_ent']
            train_step_metrics = {
                "critic_loss": critic_loss.detach(),
                "actor_loss": actor_loss.detach(),
                "acc_entropy": actor_loss_dict['acc_ent'].detach(),
                "ste_entropy": actor_loss_dict['ste_ent'].detach(),
            }
        else:
            actor_loss_dict = self.actor_loss(trajectory, is_first)
            actor_loss = actor_loss_dict['loss']
            ent = actor_loss_dict['prop_ent']
            train_step_metrics = {
                "critic_loss": critic_loss.detach(),
                "actor_loss": actor_loss.detach(),
                "term_entropy": actor_loss_dict['term_ent'].detach(),
                "prop_entropy": actor_loss_dict['prop_ent'].detach()
            }

        total_loss = actor_loss - self.train_cfg.entropy_beta * ent
        total_loss.backward()
        actor_grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in agent.actor.parameters()]))
        agent.actor_optimizer.step()

        update_target_network(agent.target, agent.critic, self.train_cfg.tau)
        train_step_metrics["critic_grad_norm"] = critic_grad_norm.detach()
        train_step_metrics["actor_grad_norm"] = actor_grad_norm.detach()

        return train_step_metrics

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
        start_time = time.time()

        pbar = tqdm(range(self.train_cfg.total_steps))
       
        for step in pbar:
            # if step % self.train_cfg.eval_every == 0:
            #     self.network.eval()
            #     self.eval(step, start_time)
            #     self.network.train()
            if step % self.train_cfg.save_every == 0:
                save_checkpoint(self.agent_1, step, self.train_cfg.checkpoint_dir)

            if self.is_f1:
                trajectory = self.gen_f1()
            elif self.simultaneous:
                trajectory = self.gen_sim()
                
            wandb.log(wandb_stats(trajectory, self.is_f1))

            for i in range(self.train_cfg.updates_per_batch):
                train_step_metrics_1 = self.train_step(
                    trajectory=trajectory,
                    agent=self.agent_1,
                    is_first=True
                )

                train_step_metrics_2 = self.train_step(
                    trajectory=trajectory,
                    agent=self.agent_2,
                    is_first=False
                )

                train_step_metrics = merge_train_step_metrics(
                    train_step_metrics_1,
                    train_step_metrics_2,
                    self.is_f1
                )

                for key, value in train_step_metrics.items():
                    print(key + ":", value)

                wandb.log(train_step_metrics)

        return

    """ TO BE DEFINED BY INDIVIDUAL ALGORITHMS"""

    @abstractmethod
    def actor_loss(batch: dict) -> float:
        pass

    @abstractmethod
    def critic_loss(batch: dict) -> float:
        pass
