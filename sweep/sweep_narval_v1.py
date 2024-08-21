import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [1e-7, 1e-6, 1e-5, 1e-4],
        'optimizer_critic.lr': [1e-7, 1e-6, 1e-5, 1e-4],
        'training.entropy_beta': [0.005, 0.01, 0.05, 0.1, 0.3],
        'training.clip_range': [0.1, 0.15, 0.2, 0.3, 0.5],
        'training.updates_per_batch': [1, 2, 3, 4],
        'training.kl_threshold': [0.01, 0.05, 0.1],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

    # if config['hp.agent_replay_buffer.mode'] == 'disabled':
    #     config['hp.agent_replay_buffer.capacity'] = 1
    #     config['hp.agent_replay_buffer.update_freq'] = 1

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