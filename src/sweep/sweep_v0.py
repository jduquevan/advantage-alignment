import os
import random

def gen_command(config):
    command = "sbatch src/sweep/run_job.slurm 1"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [5e-6, 7e-6, 1e-5, 3e-5],
        'optimizer_critic.lr': [1e-4, 5e-4, 1e-3, 5e-3],
        'training.entropy_beta': [0, 0.007, 0.01, 0.03, 0.05],
        'training.clip_range': [0.05, 0.15, 0.3],
        'training.updates_per_batch': [1, 2],
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