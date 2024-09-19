import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [0.00001, 0.00003, 0.0001],
        'optimizer_critic.lr': [0.0001, 0.0003, 0.001],
        'training.entropy_beta': [0.0001,0.0005, 0.001],
        'training.clip_range': [0.1, 0.2, 0.3],
        'training.updates_per_batch': [2, 3],
        'training.kl_threshold': [0.01, 0.05],
        'hidden_size': [192],
        'encoder.num_layers': [3],
        'training.critic_loss_mode': ['td-1'],
        'max_cxt_len': [30],
        'optimizer_ss.lr': [0.00001, 0.00003],
        'use_ss_loss': [True],
        'training.actor_loss_mode': ['integrated_aa'],
        'training.aa_weight': [0.3, 0.5, 0.7, 1, 1.5, 2, 2.5],
        'env.batch': [6],
        'agent_rb_size': [200, 400],
        'training.add_to_agent_replay_buffer_every': [10, 20],
        'training.id_weight': [0, 0.5, 1],
        'on_policy_only': [True, False],
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