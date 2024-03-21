import os
import random

def gen_command(config):
    command = "sbatch src/sweep/run_job_v1_narval.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [1e-6, 1e-5, 1e-4, 1e-3],
        'training.entropy_beta': [0.007, 0.05, 1, 1.5, 3],
        'training.clip_range': [0.05, 0.15, 0.3],
        'training.updates_per_batch': [1, 2, 3],
        'd_model': [32, 64, 128, 256],
        'num_layers': [2, 3, 4, 5],
        'optimizer_critic.lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'training.critic_loss_mode': ['td-1', 'MC']
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    config['model_in_size'] = config['d_model'] + 12
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