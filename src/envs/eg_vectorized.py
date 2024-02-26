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
            "lower": 4,
            "mean": 7,
            "upper": 10,
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

    def _get_obs(self):
        item_context = torch.cat([
            self.item_pool,
            self.utilities[self.agent_in_turn, torch.arange(self.batch_size)]
        ], dim=1)
        import pdb; pdb.set_trace()


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
            item_pool =  torch.tensor([2, 2, 2]).repeat(self.batch_size, 1).to(self.device)
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
        self.agent_in_turn = torch.zeros((self.batch_size,), dtype=torch.long).to(self.device)

        self.utilities_1, self.utilities_2, self.item_pool = self._sample_utilities()
        self.utilities = torch.stack([self.utilities_1, self.utilities_2])

        max_turns = torch.poisson(
            torch.ones(self.batch_size) * self._max_turns["mean"]
        ).to(self.device)
        self.max_turns = torch.clamp(
            max_turns, 
            self._max_turns["lower"], 
            self._max_turns["upper"]
        )

        self.utte_dummy = torch.zeros((self.batch_size, self.utte_max_len)).to(self.device)
        self.self.old_prop = torch.zeros((self.batch_size, self.nb_items)).to(self.device)
        obs = self._get_obs()

        
if __name__ == "__main__":
    env = DiscreteEG()
    env.reset()



        
