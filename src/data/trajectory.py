import torch

class TrajectoryBatch():
    def __init__(self, batch_size: int, max_traj_len: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.max_traj_len = max_traj_len
        self.device = device
        self.data = {
            "obs_0": torch.ones(
                (batch_size, max_traj_len, 25),
                dtype=torch.float32,
                device=device,
            ),
            "obs_1": torch.ones(
                (batch_size, max_traj_len, 25),
                dtype=torch.float32,
                device=device,
            ),
            "terms_0": torch.ones(
                (batch_size, max_traj_len), device=device, dtype=torch.int32
            ),
            "terms_1": torch.ones(
                (batch_size, max_traj_len), device=device, dtype=torch.int32
            ),
            "props_0": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "props_1": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "a_props_0": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "a_props_1": 1e7 * torch.ones(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "i_and_u_0": 1e7 * torch.ones(
                (batch_size, max_traj_len, 12), device=device, dtype=torch.float32
            ),
            "i_and_u_1": 1e7 * torch.ones(
                (batch_size, max_traj_len, 12), device=device, dtype=torch.float32
            ),
            "actions_0": torch.ones(  # todo: remove these for the eg game, or make it the correct actions
                (batch_size, max_traj_len, 2), device=device, dtype=torch.float32
            ),
            "actions_1": torch.ones(
                (batch_size, max_traj_len, 2), device=device, dtype=torch.float32
            ),
            "rewards_0": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "rewards_1": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_0": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_1": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "reward_sums_0": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "reward_sums_1": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "dones": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.bool,
            ),
            "cos_sims": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "max_sum_rewards": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "max_sum_reward_ratio": torch.ones(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
        }
    
    def add_step_sim(self, action, observations, rewards, log_ps, done, info, t):
        self.data[f'props_{0}'][:, t, :] = observations[0]['prop']
        self.data[f'props_{1}'][:, t, :] = observations[1]['prop']
        self.data[f'a_props_{0}'][:, t, :] = action[0]['prop']
        self.data[f'a_props_{1}'][:, t, :] = action[1]['prop']
        self.data[f'i_and_u_{0}'][:, t, :] = observations[0]['item_and_utility']
        self.data[f'i_and_u_{1}'][:, t, :] = observations[1]['item_and_utility']
        self.data[f'rewards_{0}'][:, t] = rewards[0]
        self.data[f'rewards_{1}'][:, t] = rewards[1]
        self.data[f'log_ps_{0}'][:, t] = log_ps[0]
        self.data[f'log_ps_{1}'][:, t] = log_ps[1]
        self.data[f'reward_sums_{0}'][:, t] = rewards[0] + rewards[1]
        self.data[f'reward_sums_{1}'][:, t] = rewards[0] + rewards[1]
        self.data['max_sum_rewards'][:, t] = info['max_sum_rewards']
        self.data['max_sum_reward_ratio'][:, t] = info['max_sum_reward_ratio']
        self.data['cos_sims'][:, t] = info['cos_sims']
        self.data['dones'][:, t] = done

    def add_step_f1(self, action, observations, rewards, log_ps, done, info, t):
        self.data['obs_0'][:, t, :] = observations[0]
        self.data['obs_1'][:, t, :] = observations[1]
        self.data['actions_0'][:, t, :] = action[0]
        self.data['actions_1'][:, t, :] = action[1]
        self.data['rewards_0'][:, t] = rewards[:, 0]
        self.data['rewards_1'][:, t] = rewards[:, 1]
        self.data['log_ps_0'][:, t] = log_ps[0]
        self.data['log_ps_1'][:, t] = log_ps[1]
        self.data['dones'][:, t] = done

    def detach_and_move_to_cpu(self):
        """Detaches all tensors in the data dictionary and moves them to CPU."""
        for key in self.data:
            if self.data[key].requires_grad:
                self.data[key] = self.data[key].detach()  # Detach from current graph
            self.data[key] = self.data[key].cpu().numpy().tolist()  # Move to CPU