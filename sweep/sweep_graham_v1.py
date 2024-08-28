import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [1e-6, 3e-6, 7e-6, 3e-5, 1e-5, 3e-5, 1e-4, 5e-4],
        'optimizer_critic.lr': [ 1e-6, 3e-6, 7e-6, 3e-5, 1e-5, 3e-5, 1e-4, 5e-4],
        'training.entropy_beta': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        'training.clip_range': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
        'training.updates_per_batch': [1, 2, 3],
        'training.kl_threshold': [0.01, 0.05],
        'hidden_size': [128, 256, 512],
        'encoder.num_layers': [3, 4, 5, 6, 7],
        'training.critic_loss_mode': ['MC', 'td-1', 'interpolate'],
        'max_cxt_len': [30, 50, 70],
        'optimizer_ss.lr': [1e-6, 3e-6, 7e-6, 3e-5, 1e-5, 3e-5],
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