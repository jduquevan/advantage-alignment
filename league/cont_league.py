from typing import List

import fire
import os

# add parent to path
import sys

import torch
from tqdm import tqdm

# Add the directory containing the 'utils' module to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import instantiate_agent
from train import ObligatedRatioDiscreteEG, AdvantageAlignment, AlwaysCooperateAgent, AlwaysDefectAgent


def train_gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load_env.sh\n"
    script += run_command
    return script


def train_run_job(command, fake_submit: bool = True, partition: str = 'long'):
    script = train_gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = f'sbatch --time=2:00:0 --partition={partition} --gres=gpu:a100l:1 --mem=32Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def train_main(seeds: List[int], agent_type: str, fake_submit: bool = True, partition='long', test=True):
    commands = []

    command_base_str = "python train.py training.vanilla={vanilla} " \
    "seed={seed} " \
    "wandb_tags=[cont_exchange_game_league_v1] " \
    "sum_rewards={sum_rewards} " \
    "training.proximal=False " \
    "training.max_traj_len=50 " \
    "env_conf.obs_mode=raw " \
    "encoder.num_layers=2 " \
    "mlp_model.num_layers=2 " \
    "replay_buffer.mode=enabled " \
    "replay_buffer.size=100000 " \
    "replay_buffer.update_size=500 " \
    "replay_buffer.off_policy_ratio=0.05 " \
    "optimizer_actor.lr=0.001 " \
    "optimizer_critic.lr=0.001 " \
    "training.entropy_beta=0.005 " \
    "training.aa_weight={aa_weight} " \
    "self_play=True " \
    "batch_size=16384 " \
    "training.clip_grad_norm=1.0 " \
    "training.checkpoint_dir=./experiments/cont_league/{agent_type}/seed_{seed} " \
    "--config-name milad_mila_cont.yaml"

    if agent_type == 'advantage_alignment':
        sum_rewards = False
        vanilla = False   # this enables the advantage alignment, wish it was more explicit in what it does, but it's fine
        aa_weight = 3.0
    elif agent_type == 'ppo':
        sum_rewards = False
        vanilla = True
        aa_weight = 0.0
    elif agent_type == 'ppo-sum-rewards':
        sum_rewards = True
        vanilla = True
        aa_weight = 0.0
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    for seed in seeds:
        commands.append(command_base_str.format(seed=seed, vanilla=vanilla, sum_rewards=sum_rewards, aa_weight=aa_weight, agent_type=agent_type))

    if test:
        commands = commands[:1]

    for command in commands:
        train_run_job(command=command, fake_submit=fake_submit, partition=partition)

def submit_training_agents_for_league(fake_submit: bool = True, partition: str = 'long', test: bool = True):
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    agent_types = ['advantage_alignment', 'ppo', 'ppo-sum-rewards']
    for agent_type in agent_types:
        train_main(seeds, agent_type, fake_submit=fake_submit, partition=partition, test=test)

def load_cfg():
    # load omegaconf
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="../configs", job_name=""):
        cfg = compose(config_name="milad_mila_cont", overrides=['device=cpu',
                                                                'batch_size=32',
                                                                'env_conf.obs_mode=raw',
                                                                'encoder.num_layers=2',
                                                                'mlp_model.num_layers=2',
                                                                'training.max_traj_len=50'],
                      )
    # print('config is', cfg)
    return cfg

def load_env(cfg):
    if cfg['env_conf']['type'] == 'eg-obligated_ratio':
        env = ObligatedRatioDiscreteEG(
            obs_mode=cfg['env_conf']['obs_mode'],
            max_turns={
                'lower': cfg['env_conf']['max_turns_lower'],
                'mean': cfg['env_conf']['max_turns_mean'],
                'upper': cfg['env_conf']['max_turns_upper']
            },
            sampling_type=cfg['env_conf']['sampling_type'],
            item_max_quantity=cfg['env_conf']['item_max_quantity'],
            batch_size=cfg['batch_size'],
            device=cfg['env_conf']['device'],
            prop_mode=cfg['env_conf']['prop_mode'],
        )
    else:
        raise ValueError(f"Environment type {cfg['env_conf']['type']} not supported.")

    return env

def load_agent(agent_type: str, checkpoint_path: str = None):
    cfg = load_cfg()
    env = load_env(cfg)
    always_cooperate_agent = AlwaysCooperateAgent(max_possible_prop=env.item_max_quantity, device='cpu')
    always_defect_agent = AlwaysDefectAgent(max_possible_prop=env.item_max_quantity, device='cpu')

    if agent_type == 'ckpt':
        agent = instantiate_agent(cfg)
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        agent.critic.load_state_dict(ckpt['critic_state_dict'])
        return agent
    elif agent_type == 'Always Cooperate':
        return always_cooperate_agent
    elif agent_type == 'Always Defect':
        return always_defect_agent


def run_league_between_two_agents(agent_1, agent_2):
    cfg = load_cfg()
    env = load_env(cfg)
    return AdvantageAlignment._gen_sim(env, batch_size=32, trajectory_len=cfg['training']['max_traj_len'], agent_1=agent_1, agent_2=agent_2)

def dirty_loading_of_agents():
    # this method is very nasty, we should have made better conventions for the file structure, but here is a technical debt!
    experiments_root = './league/cont_league'
    agents = {}
    for agent_type in ['advantage_alignment', 'ppo', 'ppo-sum-rewards']:
        agent_type_root = os.path.join(experiments_root, agent_type)
        print(f'**********loading agents for {agent_type}*********')
        for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
            print(f'*****loading agent for seed {seed}*****')
            ckpt_path = os.path.join(agent_type_root, f'seed_{seed}')
            # get a list of available directories
            dirs = os.listdir(ckpt_path)
            # sort dirs from recent to old based on their creation time
            dirs.sort(key=lambda x: os.path.getctime(os.path.join(ckpt_path, x)), reverse=True)
            # iterate over the directories
            for d in dirs:
                # get the checkpoint path
                ckpt = os.path.join(ckpt_path, d, 'model_800.pt')
                print(f'loading agent from {ckpt}')
                if not os.path.exists(ckpt):
                    print(f'checkpoint {ckpt} does not exist, skipping')
                    continue
                print(f'checkpoint exists at {ckpt}')
                agent = load_agent('ckpt', ckpt)
                agents.setdefault(agent_type, {})[seed] = agent

    env = load_env(load_cfg())
    agents.setdefault('Always Cooperate', {})[42] = AlwaysCooperateAgent(max_possible_prop=env.item_max_quantity, device='cpu')
    agents.setdefault('Always Defect', {})[42] = AlwaysDefectAgent(max_possible_prop=env.item_max_quantity, device='cpu')
    return agents

def run_league_runs():
    agents = dirty_loading_of_agents()

    # flatten the agents
    flat_agents = {}
    for agent_type, seeds in agents.items():
        for seed, agent in seeds.items():
            flat_agents[f'{agent_type}_{seed}'] = agent

    # move agents to device, detect device if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for agent in flat_agents.values():
        agent.to(device)

    # make all the pairs
    pairs = []
    for agent_1_name, agent_1 in flat_agents.items():
        for agent_2_name, agent_2 in flat_agents.items():
            pairs.append((
                {'agent_name': agent_1_name, 'agent': agent_1},
                {'agent_name': agent_2_name, 'agent': agent_2}
            ))

    # run the league
    league_results = {}
    for pair in tqdm(pairs):
        agent_1, agent_2 = pair
        agent_1_name = agent_1['agent_name']
        agent_2_name = agent_2['agent_name']
        agent_1 = agent_1['agent']
        agent_2 = agent_2['agent']
        print(f'running league between {agent_1_name} and {agent_2_name}:')
        result = run_league_between_two_agents(agent_1, agent_2)
        league_results[f'{agent_1_name}___{agent_2_name}'] = result
        print(f"rew {agent_1_name}: {result.data['rewards_0'].mean()} rew {agent_2_name}: {result.data['rewards_1'].mean()}")

        # write the results to a pickle file
    import pickle
    with open('./league/league_results_eg_cont_quick.pkl', 'wb') as f:
        pickle.dump(league_results, f)



if __name__ == '__main__':
    fire.Fire()
