import torch

class TrajectoryBatch():
    def __init__(self, batch_size: int, max_traj_len: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.max_traj_len = max_traj_len
        self.device = device
        self.data = {
            "obs_0": torch.empty(
                (batch_size, max_traj_len, 9),
                dtype=torch.float32,
                device=device,
            ),
            "obs_1": torch.empty(
                (batch_size, max_traj_len, 9),
                dtype=torch.float32,
                device=device,
            ),
            "terms_0": torch.empty(
                (batch_size, max_traj_len), device=device, dtype=torch.int32
            ),
            "terms_1": torch.empty(
                (batch_size, max_traj_len), device=device, dtype=torch.int32
            ),
            "props_0": torch.empty(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "props_1": torch.empty(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.float32
            ),
            "uttes_0": torch.empty(
                (batch_size, max_traj_len, 6), device=device, dtype=torch.float32
            ),
            "uttes_1": torch.empty(
                (batch_size, max_traj_len, 6), device=device, dtype=torch.float32
            ),
            "rewards_0": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "rewards_1": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_0": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "log_ps_1": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "reward_sums_0": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "reward_sums_1": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "dones": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.bool,
            ),
            "cos_sims": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
            "max_sum_rewards": torch.empty(
                (batch_size, max_traj_len),
                device=device,
                dtype=torch.float32,
            ),
        }
    
    def add_step(self, action, observations, rewards, done, info, t):
        self.data[f'obs_{t%2}'][:, t//2, :] = observations
        self.data[f'terms_{t%2}'][:, t//2] = action['term']
        self.data[f'props_{t%2}'][:, t//2, :] = action['prop']
        self.data[f'uttes_{t%2}'][:, t//2, :] = action['utte']
        self.data[f'rewards_{t%2}'][:, t//2] = rewards
        self.data[f'reward_sums_{t%2}'][:, t//2] = torch.tensor(info['sum_rewards'], device=self.device)
        self.data[f'cos_sims'][:, t] = torch.tensor(info['utility_cos_sim'], device=self.device)
        self.data[f'dones'][:, t] = done

    def add_step_sim(self, action, observations, rewards, log_ps, done, info, t):
        self.data[f'obs_{0}'][:, t, :] = observations[0]
        self.data[f'obs_{1}'][:, t, :] = observations[1]
        # self.data[f'terms_{0}'][:, t] = action[0]['term']
        # self.data[f'terms_{1}'][:, t] = action[1]['term']
        self.data[f'props_{0}'][:, t, :] = action[0]['prop']
        self.data[f'props_{1}'][:, t, :] = action[1]['prop']
        # self.data[f'uttes_{0}'][:, t, :] = action[0]['utte']
        # self.data[f'uttes_{1}'][:, t, :] = action[1]['utte']
        self.data[f'rewards_{0}'][:, t] = rewards[0]
        self.data[f'rewards_{1}'][:, t] = rewards[1]
        self.data[f'log_ps_{0}'][:, t] = log_ps[0]
        self.data[f'log_ps_{1}'][:, t] = log_ps[1]
        self.data[f'reward_sums_{0}'][:, t] = rewards[0] + rewards[1]
        self.data[f'reward_sums_{1}'][:, t] = rewards[0] + rewards[1]
        self.data['max_sum_rewards'][:, t] = info['max_sum_rewards']
        self.data['dones'][:, t] = done