import os
import random

def gen_command(config):
    command = "sbatch src/sweep/run_job_v2_narval.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [0.00005, 0.0001, 0.0003, 0.001],
        'optimizer_critic.lr': [0.0001, 0.0003, 0.0005, 0.001, 0.003],
        'training.entropy_beta': [0.0001, 0.001, 0.01, 0.1],
        'training.clip_range': [0.3],
        'training.updates_per_batch': [1],
        'gru_model.hidden_size': [512, 768],
        'mlp_model.hidden_size': [512],
        'linear_model.hidden_size': [512],
        'gru_model.num_layers': [5, 6],
        'mlp_model.num_layers': [3, 4],
        'linear_model.num_hidden': [3, 4],
        'training.clip_grad_norm': [100],
        'optimizer_actor.weight_decay': [0, 0.00005, 0.0001, 0.0005],
        'training.aa_weight': [0.5, 1, 3, 5],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    config['mlp_model.in_size'] = config['gru_model.hidden_size']
    config['linear_model.in_size'] = config['gru_model.hidden_size']

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