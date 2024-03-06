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
        simultaneous: bool,
        discrete: bool,
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
        if self.simultaneous:
            self.batch_size = env.batch_size
        else:
            self.batch_size = (len(env.envs))

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
        actor_loss, term_ent, utte_ent, prop_ent = self.actor_loss(trajectory, is_first)
        total_loss = actor_loss - self.train_cfg.entropy_beta * prop_ent
        total_loss.backward()
        actor_grad_norm = torch.sqrt(sum([torch.norm(p.grad)**2 for p in agent.actor.parameters()]))
        agent.actor_optimizer.step()

        update_target_network(agent.target, agent.critic, self.train_cfg.tau)

        train_step_metrics = {
            "critic_loss": critic_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "term_entropy": term_ent.detach(),
            "utterance_entropy": utte_ent.detach(),
            "prop_entropy": prop_ent.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "actor_grad_norm": actor_grad_norm.detach()
        }

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
        print("Generating initial trajectories")
        start_time = time.time()

        pbar = tqdm(range(self.train_cfg.total_steps))
        for step in pbar:
            # if step % self.train_cfg.eval_every == 0:
            #     self.network.eval()
            #     self.eval(step, start_time)
            #     self.network.train()

            if self.simultaneous:
                trajectory = self.gen_sim()
            else:
                trajectory = self.gen()
            wandb.log(wandb_stats(trajectory))

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

                critic_loss_1 = train_step_metrics_1["critic_loss"]
                actor_loss_1 = train_step_metrics_1["actor_loss"]
                term_ent_1 = train_step_metrics_1["term_entropy"]
                utterance_ent_1 = train_step_metrics_1["utterance_entropy"]
                prop_ent_1 = train_step_metrics_1["prop_entropy"]
                critic_grad_norm_1 = train_step_metrics_1["critic_grad_norm"]
                actor_grad_norm_1 = train_step_metrics_1["actor_grad_norm"]

                critic_loss_2 = train_step_metrics_2["critic_loss"]
                actor_loss_2 = train_step_metrics_2["actor_loss"]
                term_ent_2 = train_step_metrics_2["term_entropy"]
                utterance_ent_2 = train_step_metrics_2["utterance_entropy"]
                prop_ent_2 = train_step_metrics_2["prop_entropy"]
                critic_grad_norm_2 = train_step_metrics_2["critic_grad_norm"]
                actor_grad_norm_2 = train_step_metrics_2["actor_grad_norm"]

                print("Agent 1 Metrics:")
                print("Actor loss 1:", actor_loss_1)
                print("Critic loss 1:", critic_loss_1)
                print("Term Entropy 1:", term_ent_1)
                print("Utterance Entropy 1:", utterance_ent_1)
                print("Prop Entropy 1:", prop_ent_1)
                print("Critic Grad Norm 1:", critic_grad_norm_1)
                print("Actor Grad Norm 1:", actor_grad_norm_1)
                print()

                print("Agent 2 Metrics:")
                print("Actor loss 2:", actor_loss_2)
                print("Critic loss 2:", critic_loss_2)
                print("Term Entropy 2:", term_ent_2)
                print("Utterance Entropy 2:", utterance_ent_2)
                print("Prop Entropy 2:", prop_ent_2)
                print("Critic Grad Norm 2:", critic_grad_norm_2)
                print("Actor Grad Norm 2:", actor_grad_norm_2)
                print()

                wandb.log({
                    "Actor loss 1": actor_loss_1, 
                    "Critic loss 1": critic_loss_1,
                    "Term Entropy 1": term_ent_1,
                    "Utterance Entropy 1": utterance_ent_1,
                    "Prop Entropy 1": prop_ent_1,
                    "Critic Grad Norm 1": critic_grad_norm_1,
                    "Actor Grad Norm 1": actor_grad_norm_1,
                    "Actor loss 2": actor_loss_2,
                    "Critic loss 2": critic_loss_2,
                    "Term Entropy 2": term_ent_2,
                    "Utterance Entropy 2": utterance_ent_2,
                    "Prop Entropy 2": prop_ent_2,
                    "Critic Grad Norm 2": critic_grad_norm_2,
                    "Actor Grad Norm 2": actor_grad_norm_2,
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