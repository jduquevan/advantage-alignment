import cv2
import hydra
import os
import time

import numpy as np
import torch as t
 
from omegaconf import DictConfig, OmegaConf
from torchrl.envs import ParallelEnv
from torchrl.envs.libs.meltingpot import MeltingpotEnv
from typing import Callable, Any, Tuple, List, Dict

from train import (
    _gen_sim,
    ReplayBuffer,    
)
from utils import (
    instantiate_agent
)

num_frames = 200
model_folder = 'experiments'
model_name = '7h4b95f9/model_0.pt'
output_folder = 'videos'

def create_video_from_images(image_sequence, output_folder, video_name='output.mp4', frame_rate=3):
    """
    Create an MP4 video from a sequence of images.

    Args:
    - image_sequence (list of np.array): List of images in numpy array format.
    - output_folder (str): Folder to save the output video.
    - video_name (str): Name of the output video file.
    - frame_rate (int): Frame rate of the output video.

    Returns:
    - None
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the shape of the first image to set the video properties
    height, width, layers = image_sequence[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_path = os.path.join(output_folder, video_name)
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video file
    for img in image_sequence:
        video_writer.write(img)

    # Release the VideoWriter object
    video_writer.release()
    print(f"Video saved to {video_path}")

def make_reward_screen(past_rewards):
    """
    given past_rewards of shape (n_timesteps, n_agents)
    """
    rgb_list = []

    for x in past_rewards:
        z = make_rgb_rewards_line(x)
        rgb_list.append(z)
        rgb_list.append(np.zeros([3, z.shape[1], 3]))

    return np.concatenate(rgb_list, axis=0).astype(np.uint8)

def glue_rewards_to_images(image_seq, rew_seq):

    rew_screen = make_reward_screen(rew_seq)

    size_y = image_seq[0].shape[0]

    # pixel size in y of the numbers, we jump by this number every image
    numbers_size = 18

    new_images = []

    for i, im in enumerate(image_seq):

        x = rew_screen[:int(numbers_size*(i+1))]

        if x.shape[0] > size_y:
            x = x[int(x.shape[0]-size_y):]
        elif x.shape[0] < size_y:
            x = np.concatenate([np.zeros([size_y-x.shape[0], x.shape[1], 3]),x], axis=0)

        new_images.append(np.concatenate([im, x], axis=1).astype(np.uint8))

    return new_images

@hydra.main(version_base="1.3", config_path="configs", config_name="meltingpot.yaml")
def main(cfg: DictConfig) -> None:
    cxt_len = cfg['max_cxt_len']
    scenario = cfg['env']['scenario']

    env = ParallelEnv(1, lambda: MeltingpotEnv(scenario))

    agents = []
    images = []
    agent = instantiate_agent(cfg)

    model_path = os.path.join(model_folder, model_name)
    model_dict = t.load(model_path)
    agent.actor.load_state_dict(model_dict['actor_state_dict'])

    for i in range(cfg['env']['num_agents']):
        agents.append(agent)

    state = env.reset()

    for i in range(num_frames // cxt_len):

        replay_buffer = ReplayBuffer(
            env=env,
            replay_buffer_size=cfg['rb_size'],
            n_agents=len(agents),
            cxt_len=cxt_len,
            device=cfg['device']
        )

        trajectory, state = _gen_sim(state, env, 1, cxt_len, agents, replay_buffer, False, False)
        images.append(trajectory.data['full_maps'][0].cpu().numpy())
        import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    main()