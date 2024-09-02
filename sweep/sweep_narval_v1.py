import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        'optimizer_critic.lr': [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        'training.entropy_beta': [0, 0.00005, 0.0001, 0.0005],
        'training.clip_range': [0.1, 0.2, 0.3],
        'training.updates_per_batch': [1, 2, 3],
        'training.kl_threshold': [0.01, 0.05],
        'hidden_size': [32, 64, 128],
        'encoder.num_layers': [2, 3, 4, 5],
        'training.critic_loss_mode': ['MC', 'td-1', 'interpolate'],
        'max_cxt_len': [50, 60, 70],
        'optimizer_ss.lr': [1e-5, 3e-5, 1e-4, 3e-4,],
        'use_ss_loss': [True, False],
        'training.actor_loss_mode': ['integrated_aa'],
        'training.aa_weight': [1, 2, 3, 4],
        'env.batch': [6, 12, 18, 24, 30],
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