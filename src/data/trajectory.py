import torch

class TrajectoryBatch():
    def __init__(self, batch_size: int, max_traj_len: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.max_traj_len = max_traj_len
        self.device = device
        self.data = {
            "obs_0": torch.empty(
                (batch_size, max_traj_len, 15),
                dtype=torch.float32,
                device=device,
            ),
            "obs_1": torch.empty(
                (batch_size, max_traj_len, 15),
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
                (batch_size, max_traj_len, 3), device=device, dtype=torch.int32
            ),
            "props_1": torch.empty(
                (batch_size, max_traj_len, 3), device=device, dtype=torch.int32
            ),
            "uttes_0": torch.empty(
                (batch_size, max_traj_len, 6), device=device, dtype=torch.int32
            ),
            "uttes_1": torch.empty(
                (batch_size, max_traj_len, 6), device=device, dtype=torch.int32
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
        }
    
    def add_step(self, action, observations, rewards, t):
        self.data[f'obs_{t%2}'][:, t//2, :] = observations
        self.data[f'terms_{t%2}'][:, t//2] = action['term']
        self.data[f'props_{t%2}'][:, t//2, :] = action['prop']
        self.data[f'uttes_{t%2}'][:, t//2, :] = action['utte']
        self.data[f'rewards_{t%2}'][:, t//2] = rewards