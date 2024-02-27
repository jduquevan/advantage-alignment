import gym
from gym import spaces
import numpy as np
import torch
from copy import deepcopy
import random
import os
import sys

grandparent_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(grandparent_folder)
from src.utils.utils import get_cosine_similarity, unit_vector_from_angles
from src.utils.format_utils import (
    empty_lists_in_nested_dict,
    t2a,
    a2t_in_nested_dict,
    compare,
    compare_eps
)


# define the exchange environment
# reference: Emergent Communication through Negotiation, https://arxiv.org/abs/1804.03980
class ExchangeEnv(gym.Env):
    def __init__(
        self,
        nb_agents=2,  # number of agents
        max_turns={
            "lower": 4,
            "mean": 7,
            "upper": 10,
        },  # turn ~ poisson(mean) and clipped by [lower, upper]
        alpha=[
            1,
            0,
        ],  # the prosocial parameters, [1, 0] means selfish agents, [1, 1] means prosocial agents
        utte_channel=False,  # if yes, the agent can "see" the previous utterance
        prop_channel=True,  # if yes, the agent can "see" the previous proposal
        utte_max_len=6,  # the maximum length of utterance is 6
        nb_items=3,  # there are 3 kinds of items in the pool
        item_max_quantity=5,  # each kind of item has a maximum quantity of 5
        utility_max=10,  # the utility are random generated, the maximum utility for each item is 10
        terminate_condition="by_agent",  # by_agent or equal_prop or equal_eps or satisfied_prop
        device=None,
    ):
        super(ExchangeEnv, self).__init__()
        self.utte_max_len = utte_max_len
        self.nb_items = nb_items
        self.item_max_quantity = item_max_quantity
        self.utility_max = utility_max
        self._max_turns = max_turns
        self.terminate_condition = terminate_condition
        prop_low = np.zeros(self.nb_items, dtype=np.float32)
        prop_high = np.full((self.nb_items,), item_max_quantity, dtype=np.float32)
        item_continuous_low = np.zeros(self.nb_items, dtype=np.float32)
        item_continuous_high = np.full(
            self.nb_items, self.item_max_quantity, dtype=np.float32
        )
        utility_continuous_low = np.zeros(self.nb_items, dtype=np.float32)
        utility_continuous_high = np.full(
            self.nb_items, self.utility_max, dtype=np.float32
        )
        # Combine low and high bounds for items and utilities
        low_bound = np.concatenate((item_continuous_low, utility_continuous_low))
        high_bound = np.concatenate((item_continuous_high, utility_continuous_high))

        if self.terminate_condition.lower() == "by_agent":
            self.action_space = spaces.Dict(
                {
                    "term": spaces.Discrete(2),
                    "utte": spaces.Box(
                        low=0, high=1, shape=(self.utte_max_len,), dtype=np.int64
                    ),
                    "prop": spaces.Box(low=prop_low, high=prop_high, dtype=np.float32),
                }
            )
        else:
            self.action_space = spaces.Dict(
                {
                    "utte": spaces.Box(
                        low=0, high=1, shape=(self.utte_max_len,), dtype=np.int64
                    ),
                    "prop": spaces.Box(low=prop_low, high=prop_high, dtype=np.float32),
                }
            )
        self.observation_space = spaces.Dict(
            {
                "item_and_utility": spaces.Box(
                    low=low_bound, high=high_bound, dtype=np.float32
                ),
                "utte": spaces.Box(
                    low=0, high=1, shape=(self.utte_max_len,), dtype=np.int64
                ),
                "prop": spaces.Box(low=prop_low, high=prop_high, dtype=np.float32),
            }
        )
        self.nb_agents = nb_agents
        # prosocial agent: alpha = [1, 1]
        # selfish agent: alpha = [1, 0]
        self.alpha = np.array(alpha)  # prosocial parameter
        # import pdb; pdb.set_trace()
        self.utte_channel = utte_channel
        self.prop_channel = prop_channel

        assert self.terminate_condition.lower() in [
            "by_agent",
            "equal_prop",
            "equal_eps",
            "satisfied_prop",
        ], f"terminate_condition should be by_agent, \
        equal_prop or satisfied_prop, not {self.terminate_condition}"

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def reset(self):
        # self.item_pool = np.random.randint(
        #     0, self.item_max_quantity + 1, size=self.nb_items
        # )
        # self.utilities = np.random.randint(
        #     0, self.utility_max + 1, size=(self.nb_agents, self.nb_items)
        # )

        # Ensures high cosine similarity by sampling utilities from the positive part of R^3 - cone
        min_angle = 15
        max_angle = 30
        thetas_1 = np.random.uniform(min_angle, max_angle, 2)
        thetas_2 = np.random.uniform(90 - (max_angle - min_angle), 90, 2)

        # utilities_1 = unit_vector_from_angles(thetas_1[0], thetas_1[1])
        # utilities_2 = unit_vector_from_angles(thetas_2[0], thetas_2[1])
        utilities_1 = np.array([1, 0.3, 0.2])
        utilities_2 = np.array([0.3, 1, 0.2])

        self.utilities = np.vstack((utilities_1, utilities_2))

        # Ensures "fair" distribution of the available resources
        # self.item_pool = np.sum(self.utilities, axis=0)
        self.item_pool = np.array([1, 1, 1])

        self.turn = 0
        self.rewards = [0] * self.nb_agents
        self.dones = [False] * self.nb_agents
        self.done = False

        # self.old_prop = torch.round(torch.from_numpy(self.item_pool / 2)).type(
        #     torch.float32
        # )
        self.old_prop = np.round(self.item_pool / 2).astype(np.float32)
        self.old_utte = torch.zeros(self.utte_max_len, dtype=torch.int64)
        self.utte_dummy = torch.zeros(self.utte_max_len, dtype=torch.int64)
        self.prop_dummy = torch.round(torch.from_numpy(self.item_pool / 2)).type(
            torch.int
        )
        self.agent_in_turn = 0

        if isinstance(self._max_turns, int):
            self.max_turns = self._max_turns
        elif isinstance(self._max_turns, (list, tuple)):
            lower, mean, upper = self._max_turns
            self.max_turns = np.random.poisson(mean, size=1)[0]
            self.max_turns = np.clip(self.max_turns, lower, upper)
        elif isinstance(self._max_turns, dict):
            lower, mean, upper = (
                self._max_turns["lower"],
                self._max_turns["mean"],
                self._max_turns["upper"],
            )
            self.max_turns = np.random.poisson(mean, size=1)[0]
            self.max_turns = np.clip(self.max_turns, lower, upper)
        self.baseline_reward = np.einsum("j,ij->i", self.item_pool, self.utilities) / 2
        self.info = {
            "reward_if_agree_prev": np.array([0, 0]),
            "agent_in_turn": 0,
            "turn": -1,
            "action_term": torch.tensor(0).to(self.device),
            "action_utte": self.old_utte.to(self.device),
            "action_prop": torch.tensor(self.old_prop).to(self.device),
            "actual_reward": -1,
            "item_pool": self.item_pool,
            "utilities": self.utilities,
            "max_turns": self.max_turns,
            "utte_channel": self.utte_channel,
            "prop_channel": self.prop_channel,
            "alpha": self.alpha,
            "terminate_condition": self.terminate_condition,
            "nb_agents": self.nb_agents,
            "baseline_reward": self.baseline_reward,
            "nb_items": self.nb_items,
            "item_max_quantity": self.item_max_quantity,
            "utility_max": self.utility_max,
            "done": False,
            "utility_cos_sim": get_cosine_similarity(self.utilities[0], self.utilities[1]),
            "sum_rewards": 0
        }
        self.history = [self.info]
        self.obs = self._get_obs()
        return self.obs, self.info

    def step(self, action):
        if self.done:
            return self.obs, 0, self.done, None, self.info
        assert isinstance(action, dict), "action should be a dict"
        if "term" in action:
            assert int(action["term"]) in [0, 1], "termination should be 0 or 1"
        assert (
            len(action["utte"]) == self.utte_max_len
            or len(action["utte"][0]) == self.utte_max_len
        ), f"utte length should be {self.utte_max_len}"
        assert (
            len(action["prop"]) == self.nb_items
            or len(action["prop"][0]) == self.nb_items
        ), f"prop length should be {self.nb_items}"
        self.agent_in_turn = int(self.turn % self.nb_agents)
        self.info["agent_in_turn"] = self.agent_in_turn
        self.info["turn"] = self.turn
        sum_rewards = 0
        if self.terminate_condition.lower() == "by_agent":
            term = action["term"]
        utte, prop = (
            action["utte"],
            action["prop"],
        )
        for k in action:
            self.info[f"action_{k}"] = a2t_in_nested_dict(action, self.device)[k]
        if not self.utte_channel:
            utte = self.utte_dummy
        if self.terminate_condition.lower() == "by_agent":
            term_flag = term
        elif self.terminate_condition.lower() == "equal_prop":
            term_flag = compare(self.old_prop, prop, "==", allany="all")
        elif self.terminate_condition.lower() == "equal_eps":
            term_flag = compare_eps(self.old_prop, self.item_pool - prop)
        elif self.terminate_condition.lower() == "satisfied_prop":
            if self.agent_in_turn == 0:
                # the old_prop is done by agent 1,
                # prop is done by agent 0, larger prop means higher utility for agent 0
                term_flag = compare(prop, self.old_prop, "<=", allany="all")
            elif self.agent_in_turn == 1:
                # the old_prop is done by agent 0,
                # prop is done by agent 1, smaller prop means higher utility for agent 1
                term_flag = compare(
                    prop, self.old_prop, ">=", allany="all"
                )  # 1 - prop <= 1 - self.old_prop
            else:
                raise ValueError(f"Not supported agent_in_turn: {self.agent_in_turn}")
        if not np.any(
            self.dones
        ):  # keep track of the rewards until a done flag appears
            self.info["reward_if_agree_prev"] = self._calculate_rewards(self.old_prop)
        else:
            # if any agent has terminated,
            # the game transfers into settlement period
            # the agent shall get reward based on
            # the previous calculated rewards
            # self.info["reward_if_agree_prev"] = self.info["reward_if_agree_prev"][-1]
            pass

        self.obs = self._get_obs()
        if np.any(self.dones):
            # if any agent has terminated, the game transfers into settlement period
            # the agent shall get reward based on the previous calculated rewards
            # and mark itself as done.
            # When all the agents mark themselves as done,
            # the game is over (self.done = True)
            reward = self.rewards[self.agent_in_turn]
            self.dones[self.agent_in_turn] = True
            self.done = np.all(self.dones)
            sum_rewards = np.sum(self.rewards)
        elif term_flag:
            if self.terminate_condition.lower() == "satisfied_prop":
                self.rewards = self._calculate_rewards((prop + self.old_prop) / 2)
            else:
                self.rewards = self.info["reward_if_agree_prev"]
            reward = self.rewards[self.agent_in_turn]
            self.dones[self.agent_in_turn] = True
            self.done = np.all(self.dones)
            sum_rewards = np.sum(self.rewards)
        else:
            self.rewards = [0] * self.nb_agents
            reward = self.rewards[self.agent_in_turn]

        self.info["actual_reward"] = reward
        truncation = None
        if self.turn >= self.max_turns and self.agent_in_turn == self.nb_agents - 1:
            self.done = True
            self.info["done"] = True
            return self.obs, reward, self.done, truncation, self.info

        # update for the next round
        if not self.done:
            self.turn += 1
        
        self.info["utility_cos_sim"] = get_cosine_similarity(self.utilities[0], self.utilities[1])
        self.info["sum_rewards"] = sum_rewards

        self.old_prop = deepcopy(prop)
        self.old_utte = deepcopy(utte)
        self.history.append(self.info)
        self.info["done"] = self.done
        return self.obs, reward, self.done, truncation, self.info

    def _get_obs(self):
        item_context = np.concatenate(
            (self.item_pool, self.utilities[self.agent_in_turn])
        )
        if self.utte_channel:
            utte_context = self.old_utte

        else:
            utte_context = self.utte_dummy
        if self.prop_channel:
            prop_context = self.old_prop
        else:
            prop_context = self.prop_dummy
        # return np.concatenate([item_context, utte_context, prop_context])
        return {
            "item_and_utility": item_context,  # shape: item_pool(=3) + utility(=3) = 6
            "utte": utte_context,  # shape: utterance_max_len = 6
            "prop": prop_context,  # shape: 3
        }

    def _calculate_rewards(self, prop):
        assert self.nb_agents == 2, "Only support 2 agents for now"
        if self._is_valid(prop):
            # import pdb; pdb.set_trace()
            # if self.agent_in_turn == 0:
            #     prop = np.array([1, 0, 0.5])
            # else:
            #     prop = np.array([0, 1, 0.5])
            # prop = np.array([1, 1, 0.99])

            if self.agent_in_turn == 1:
                reward_A = np.dot(self.utilities[0], self.item_pool - t2a(prop))
                reward_B = np.dot(self.utilities[1], t2a(prop))
            else:
                reward_A = np.dot(self.utilities[0], t2a(prop))
                reward_B = np.dot(self.utilities[1], self.item_pool - t2a(prop))
        else:
            reward_A = reward_B = 0

        joint_reward = [
            np.dot(np.roll(self.alpha, shift=-i), np.array([reward_A, reward_B]))
            for i in range(self.nb_agents)
        ]
        # import pdb; pdb.set_trace()
        scaled_reward = self._scale_reward(joint_reward)

        return scaled_reward

    def _is_valid(self, prop):
        # Check if the proposal is valid (i.e., it doesn't request more items than available in the item pool)
        return np.all(t2a(prop) <= self.item_pool)

    def _scale_reward(self, joint_reward):
        joint_reward = (
            np.array(joint_reward)
            if not isinstance(joint_reward, np.ndarray)
            else joint_reward
        )
        unique_elements, counts = np.unique(self.alpha, return_counts=True)
        if (
            np.all(unique_elements == [0, 1]) and counts[1] == 1
        ):  # n-1 times of 0 and 1 time of 1 => selfish agent
            max_possible_reward = np.einsum("j,ij->i", self.item_pool, self.utilities)
        else:
            max_possible_reward = np.array(
                [
                    np.dot(
                        self.item_pool,
                        np.maximum(
                            *(
                                np.roll(self.alpha, shift=-i).reshape(-1, 1)
                                * self.utilities
                            )
                        ),
                    )
                    for i in range(self.nb_agents)
                ]
            )
            # compare np.maximum(alpha[0]*Ua and alpha[1]*Ub) for agent a
            # and compare np.maximum(alpha[1]*Ua and alpha[0]*Ub) for agent b
            # P.S. alpha[0] = alpha[1] = 1 for prosocial agents
        return np.where(
            max_possible_reward != 0, joint_reward / max_possible_reward, 0
        )  # only divide when max_possible_reward != 0, otherwise return 0
        # return np.nan_to_num(joint_reward / max_possible_reward)

    def render(self, mode="detailed"):
        if mode == "detailed":
            print("item pool: ", self.item_pool)
            print("utilities: ", self.utilities)
            for i in range(0, len(self.info["reward_if_agree_prev"])):
                print("step: ", i)
                print("agent in turn: ", self.info["agent_in_turn"][i])
                print("turn: ", self.info["turn"][i])
                for k in self.info:
                    if k.startswith("action"):
                        print(k, ": ", self.info[k][i])
                print(
                    "reward_if_agree_prev: ",
                    self.info["reward_if_agree_prev"][i],
                )
                print("actual_reward: ", self.info["actual_reward"][i])
                print("")
        return


from gym.vector import SyncVectorEnv


def make_vectorized_env(nb_env, *args, **kwargs):
    def make_exchange_env():
        return ExchangeEnv(*args, **kwargs)

    envs = SyncVectorEnv([make_exchange_env for _ in range(nb_env)])
    return envs


if __name__ == "__main__":
    # set random seed
    np.random.seed(2)
    torch.manual_seed(0)
    random.seed(0)
    # initialize the environment
    env = ExchangeEnv(
        nb_agents=2,
        max_turns=10,
        alpha=[1, 0],  # selfish agent, alpha = [1, 0]; prosocial agent, alpha = [1, 1]
        utte_channel=True,
        prop_channel=True,
    )
    env.reset()
    # test the environment
    print("=====================test_single_env=====================")
    action_A_1 = {
        "term": False,
        "utte": np.zeros(6),
        "prop": np.array([1.1, 2.1, 6.1]),
    }  # [False, np.zeros(6), [1, 2, 6]]
    action_B_1 = {
        "term": False,
        "utte": np.zeros(6),
        "prop": np.array([2, 3, 3]),
    }  # [False, np.zeros(6), [2, 3, 3]]
    action_A_2 = {
        "term": False,
        "utte": np.zeros(6),
        "prop": np.array([2, 2, 1]),
    }  # [False, np.zeros(6), [2, 2, 1]]
    action_B_2 = {
        "term": False,
        "utte": np.zeros(6),
        "prop": np.array([1, 2, 3]),
    }  # [False, np.zeros(6), [1, 2, 3]]
    action_A_3 = {
        "term": True,
        "utte": np.zeros(6),
        "prop": np.array([1, 1, 1]),
    }  # [True, np.zeros(6), [1, 1, 1]]
    action_B_3 = {
        "term": False,
        "utte": np.zeros(6),
        "prop": np.array([1, 1, 1]),
    }  # [False, np.zeros(6), [1, 1, 1]]
    actions = [action_A_1, action_B_1, action_A_2, action_B_2, action_A_3, action_B_3]
    print("item pool: ", env.item_pool)
    print("utilities: ", env.utilities)
    for i, action in enumerate(actions):
        obs, reward, done, _, info = env.step(action)
        print("step: ", i)
        print("obs: ", obs)
        print("agent in turn: ", env.agent_in_turn)
        print("reward: ", reward)
        print("dones: ", env.dones)
        print("done: ", done)
        print("info: ", info)
        print("")
    print("=====================end_of_test=====================")
    # test the vectorized environment
    print("=====================test_vectorized_env=====================")
    envs = make_vectorized_env(10)
    # Reset the environments
    envs.reset()

    # Example actions for each environment
    actions = envs.action_space.sample()

    # Step the environments with the actions
    next_states, rewards, dones, truncation, infos = envs.step(actions)
    print("=====================end_of_test=====================")