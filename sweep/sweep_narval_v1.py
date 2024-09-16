import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [0.00003, 0.0001, 0.0003, 0.001, 0.003],
        'optimizer_critic.lr': [0.00003, 0.0001, 0.0003, 0.001, 0.003],
        'training.entropy_beta': [0.0001, 0.001, 0.003, 0.01],
        'training.clip_range': [0.1, 0.2, 0.3],
        'training.updates_per_batch': [2, 3],
        'training.kl_threshold': [0.01, 0.05],
        'hidden_size': [128],
        'encoder.num_layers': [2, 3, 4],
        'training.critic_loss_mode': ['td-1', 'MC'],
        'max_cxt_len': [60],
        'optimizer_ss.lr': [0],
        'use_ss_loss': [False],
        'training.actor_loss_mode': ['integrated_aa'],
        'training.aa_weight': [0.3, 0.5, 0.7, 1, 1.5, 2, 2.5],
        'env.batch': [6],
        'agent_rb_size': [200],
        'training.add_to_agent_replay_buffer_every': [20],
        'training.id_weight': [0],
    }

    # sample a random config
    config = {}
    for key, values in hparams.items():
        config[key] = random.choice(values)

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