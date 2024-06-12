import os
import random

def gen_command(config):
    command = "sbatch src/sweep/run_job_f1_v0.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [1e-05],
        'optimizer_critic.lr': [0.0005],
        'training.entropy_beta': [0.0001],
        'training.clip_range': [0.15],
        'training.updates_per_batch': [2],
        'gru_model.hidden_size': [128, 256, 512, 1024, 2048],
        'mlp_model.hidden_size': [128, 256, 512, 1024, 2048],
        'linear_model.hidden_size': [128, 256, 512, 1024, 2048],
        'gru_model.num_layers': [1, 2, 3, 4, 5],
        'mlp_model.num_layers': [1, 2, 3, 4, 5],
        'linear_model.num_hidden': [1, 2, 3, 4, 5],
        # 'hidden_size': [32, 64, 128, 256],
        # 'num_layers': [1, 2, 4]
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # config['in_size'] = config['hidden_size'] + 50

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