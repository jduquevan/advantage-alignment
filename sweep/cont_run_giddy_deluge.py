import os
import random


def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load_env.sh\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True, partition: str = 'long'):
    script = gen_script(command)
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

def main(fake_submit: bool = True, partition='long', test=True):
    commands = []
    actor_lr = random.choice([1e-3])  #
    critic_lr = random.choice([1e-3])  #
    entropy_beta = random.choice([0.005])
    aa_weight = random.choice([3.])
    off_policy_ratio = random.choice([0.05])
    clip_grad_norm = random.choice([1.0])  #

    commands.append(
        f"python train.py training.vanilla=False"
        f" wandb_tags=[exchange_game_sweep_v3]"
        f" sum_rewards=False "
        f" training.proximal=False"
        f" training.max_traj_len=50"
        f" env_conf.obs_mode=raw"
        f" encoder.num_layers=2"
        f" mlp_model.num_layers=2"
        f" replay_buffer.mode=enabled"
        f" replay_buffer.size=100000"
        f" replay_buffer.update_size=500"
        f" replay_buffer.off_policy_ratio={off_policy_ratio}"
        f" optimizer_actor.lr={actor_lr}"
        f" optimizer_critic.lr={critic_lr}"
        f" training.entropy_beta={entropy_beta}"
        f" training.aa_weight={aa_weight}"
        f" self_play=True"
        f" batch_size=16384"
        f" training.clip_grad_norm={clip_grad_norm}"
        f" --config-name milad_mila_cont.yaml"
    )

    if test:
        commands = commands[:1]
    for command in commands:
        run_job(command=command, fake_submit=fake_submit, partition=partition)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









