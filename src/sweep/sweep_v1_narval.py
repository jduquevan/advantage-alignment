import os
import random

def gen_command(config):
    command = "sbatch src/sweep/run_job_v1_narval.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [5e-6, 1e-05, 3e-05],
        'training.entropy_beta': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'training.clip_range': [0.2, 0.3, 0.4, 0.5, 1],
        'training.updates_per_batch': [1, 2, 3],
        'd_model': [256, 512, 1024],
        'num_layers': [4, 5],
        'optimizer_critic.lr': [5e-5, 1e-4, 5e-4, 1e-3],
        'training.clip_grad_norm': [10, 100],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    config['linear_model.num_hidden'] = config['num_layers'] - 1

    # submit this job using slurm
    command = gen_command(config)
    if fake_submit:
        print('fake submit')
    else:
        os.system(command)
    print(command)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_job(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()