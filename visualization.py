import cv2
import hydra
import os

import numpy as np
import torch as t
 
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.meltingpot import MeltingpotEnv

from train import (
    _gen_sim,
    ReplayBuffer,    
)
from utils import (
    instantiate_agent
)

num_frames = 1000
model_folder = 'experiments'
model_name = 'x3m4fr9o/model_1000.pt'
output_folder = 'videos'


def create_video_from_frames(frames, output_folder, video_name='output.mp4', frame_rate=3):
    """
    Create an MP4 video from a numpy array of frames.

    Args:
    - frames (np.array): 4D numpy array with shape (time, height, width, channels).
    - output_folder (str): Folder to save the output video.
    - video_name (str): Name of the output video file.
    - frame_rate (int): Frame rate of the output video.

    Returns:
    - None
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if frames.dtype != np.uint8:
        frames = (frames).astype(np.uint8)

    # Get the shape of the first frame to set the video properties
    height, width, channels = frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 (H264 is widely supported)
    video_path = os.path.join(output_folder, video_name)
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video file
    for i in range(frames.shape[0]):
        video_writer.write(frames[i])

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved to {video_path}")

@hydra.main(version_base="1.3", config_path="configs", config_name="meltingpot.yaml")
def main(cfg: DictConfig) -> None:
    cxt_len = cfg['max_cxt_len']
    scenario = cfg['env']['scenario']

    env = ParallelEnv(1, lambda: MeltingpotEnv(scenario))

    agents = []
    images = []
    rewards = []
    agent = instantiate_agent(cfg)

    model_path = os.path.join(model_folder, model_name)
    model_dict = t.load(model_path)
    agent.actor.load_state_dict(model_dict['actor_state_dict'])

    for i in range(cfg['env']['num_agents']):
        agents.append(agent)

    state = env.reset()

    replay_buffer = ReplayBuffer(
        env=env,
        replay_buffer_size=cfg['rb_size'],
        n_agents=len(agents),
        cxt_len=cxt_len,
        device=cfg['device']
    )

    for i in range(num_frames // cxt_len):
        trajectory, state = _gen_sim(state, env, 1, cxt_len, agents, replay_buffer, True)
        images.append(trajectory.data['full_maps'][0].cpu().numpy())
        rewards.append(trajectory.data['rewards'].cpu().numpy())
    
    frames = np.concatenate(images)
    create_video_from_frames(frames, output_folder=output_folder, video_name='output.mp4', frame_rate=3)

    

if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    main()