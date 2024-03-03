import gym
import os
import sys
import torch

grandparent_folder = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(grandparent_folder)
from src.utils.utils import get_cosine_similarity, unit_vector_from_angles

class DiscreteEG(gym.Env):
    def __init__(
        self,
        max_turns={
            "lower": 2,
            "mean": 4,
            "upper": 6,
        },
        utte_channel=False,
        utte_max_len=6,
        nb_items=3,
        item_max_quantity=5,
        utility_max=10,
        device=None,
        batch_size=128,
        sampling_type="no_sampling"
        ):
        super(DiscreteEG, self).__init__()

        self._max_turns = max_turns
        self.utte_channel = utte_channel
        self.utte_max_len = utte_max_len
        self.nb_items = nb_items
        self.item_max_quantity = item_max_quantity
        self.utility_max = utility_max
        self.device = device
        self.batch_size = batch_size
        self.sampling_type = sampling_type
        self.info = {}

    def _get_obs(self, props):
        prop_1, prop_2 = props
        item_context_1 = torch.cat([self.item_pool, self.utilities_1], dim=1)
        item_context_2 = torch.cat([self.item_pool, self.utilities_2], dim=1)

        # TODO: Add utterance support
        return ({
            "item_and_utility": item_context_1,
            "utte": self.utte_dummy_1,
            "prop": prop_1,
        },
        {
            "item_and_utility": item_context_2,
            "utte": self.utte_dummy_2,
            "prop": prop_2,
        })


    def _sample_utilities(self):
        if self.sampling_type == "uniform":
            utilities_1 = torch.randint(
                0, 
                self.utility_max + 1, 
                (self.batch_size, self.nb_items)
            ).to(self.device)
            utilities_2 = torch.randint(
                0, 
                self.utility_max + 1, 
                (self.batch_size, self.nb_items)
            ).to(self.device)
            item_pool = torch.randint(
                0, 
                self.item_max_quantity + 1, 
                (self.batch_size, self.nb_items)
            ).to(self.device)
        elif self.sampling_type == "no_sampling":
            # For debugging purposes
            utilities_1 = torch.tensor([10, 3, 2]).repeat(self.batch_size, 1).to(self.device)
            utilities_2 = torch.tensor([3, 10, 2]).repeat(self.batch_size, 1).to(self.device)
            item_pool =  torch.tensor([4, 4, 4]).repeat(self.batch_size, 1).to(self.device)
        else:
            raise Exception(f"Unsupported sampling type: '{self.sampling_type}'.")

        return utilities_1, utilities_2, item_pool

    def reset(self):
        self.a1_is_done = torch.torch.full(
            (self.batch_size,), 
            False, 
            dtype=torch.bool
        ).to(self.device)
        self.a2_is_done = torch.torch.full(
            (self.batch_size,), 
            False, 
            dtype=torch.bool
        ).to(self.device)
        self.env_is_done = torch.torch.full(
            (self.batch_size,), 
            False, 
            dtype=torch.bool
        ).to(self.device)

        self.turn = torch.zeros((self.batch_size,)).to(self.device)

        self.utilities_1, self.utilities_2, self.item_pool = self._sample_utilities()

        max_turns = torch.poisson(
            torch.ones(self.batch_size) * self._max_turns["mean"]
        ).to(self.device)
        self.max_turns = torch.clamp(
            max_turns, 
            self._max_turns["lower"], 
            self._max_turns["upper"]
        )

        self.prop_dummy = torch.zeros((self.batch_size, self.nb_items)).to(self.device)
        self.utte_dummy_1 = torch.zeros((self.batch_size, self.utte_max_len)).to(self.device)
        self.utte_dummy_2 = torch.ones((self.batch_size, self.utte_max_len)).to(self.device)
        obs = self._get_obs((self.prop_dummy, self.prop_dummy))

        return obs, self.info

    def step(self, actions):
        self.turn += 1
        actions_1, actions_2 = actions
        prop_1 = actions_1["prop"]
        prop_2 = actions_2["prop"]

        obs = self._get_obs((prop_2, prop_1))

        rewards_1 = torch.bmm(
            prop_1.float().unsqueeze(1), 
            self.utilities_1.float().unsqueeze(2)
        ).flatten()
        rewards_2 = torch.bmm(
            prop_2.float().unsqueeze(1), 
            self.utilities_2.float().unsqueeze(2)
        ).flatten()
        max_reward_1 = torch.bmm(
            self.item_pool.float().unsqueeze(1), 
            self.utilities_1.float().unsqueeze(2)
        ).flatten()
        max_reward_2 = torch.bmm(
            self.item_pool.float().unsqueeze(1), 
            self.utilities_2.float().unsqueeze(2)
        ).flatten()
        norm_rewards_1 = rewards_1/max_reward_1
        norm_rewards_2 = rewards_2/max_reward_2

        dones = (self.turn >= self.max_turns)
        valid_props = (((prop_1 + prop_2) <= self.item_pool).sum(dim=1) == 3)
        done_and_valid = torch.logical_and(dones, valid_props)

        utilities_1, utilities_2, item_pool = self._sample_utilities()

        rewards_1 = torch.where(done_and_valid, norm_rewards_1, 0)
        rewards_2 = torch.where(done_and_valid, norm_rewards_2, 0)

        reset = dones.unsqueeze(1).repeat(1, 3)
        self.utilities_1 = torch.where(reset, utilities_1, self.utilities_1)
        self.utilities_2 = torch.where(reset, utilities_2, self.utilities_2)
        self.item_pool = torch.where(reset, item_pool, self.item_pool)
        self.turn = torch.where(dones, 0, self.turn)

        return obs, (rewards_1, rewards_2), dones, None, self.info

if __name__ == "__main__":
    env = DiscreteEG()
    env.reset()
    prop_1 = torch.randint(0, 3, (128, 3))
    prop_2 = torch.randint(0, 3, (128, 3))
    actions_1 = {
        "term": torch.zeros((128)).to(env.device),
        "utte": torch.zeros((128, 6)).to(env.device),
        "prop": prop_1
    }
    actions_2 = {
        "term": torch.zeros((128)).to(env.device),
        "utte": torch.zeros((128, 6)).to(env.device),
        "prop": prop_2
    }
    env.step((actions_1, actions_2))





        
