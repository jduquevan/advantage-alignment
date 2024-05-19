from typing import List

import fire
import os

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


if __name__ == '__main__':
    fire.Fire()
