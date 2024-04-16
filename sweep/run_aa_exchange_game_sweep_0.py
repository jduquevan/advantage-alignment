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
        command = f'sbatch --time=8:00:0 --partition={partition} --gres=gpu:a100l:1 --mem=20Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def main(fake_submit: bool = True, partition='long', test=True):
    commands = []
    for aa_weight in [1., 10., 30., 100.]:
        for entropy_beta in [0.1, 0.2, 1.0, 3.0]:
            for actor_lr in [1e-3, 1e-4]:
                commands.append(
                    f"python train.py training.vanilla=False"
                    f" wandb_tags=[exchange_game_aa_0]"
                    f" sum_rewards=False "
                    f" optimizer_actor.lr={actor_lr}"
                    f" training.entropy_beta={entropy_beta}"
                    f" training.aa_weight={aa_weight}"
                    f" self_play=True"
                    f" --config-name milad_mila.yaml"
                )

    if test:
        commands = commands[:1]
    for command in commands:
        run_job(command=command, fake_submit=fake_submit, partition=partition)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









