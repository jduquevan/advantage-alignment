import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [0.00005],
        'optimizer_critic.lr': [0.0001],
        'training.entropy_beta': [0.01, 0.03, 0.05, 0.07, 0.1],
        'training.clip_range': [0.1, 0.2, 0.3],
        'training.updates_per_batch': [2, 3],
        'training.kl_threshold': [0.05],
        'hidden_size': [192],
        'encoder.num_layers': [3],
        'training.critic_loss_mode': ['td-1'],
        'max_cxt_len': [10, 15, 20, 25, 30],
        'optimizer_ss.lr': [0],
        'use_ss_loss': [False],
        'training.actor_loss_mode': ['integrated_aa'],
        'training.aa_weight': [0.5, 1, 1.5, 2],
        'env.batch': [6],
        'agent_rb_size': [100, 300, 500],
        'training.add_to_agent_replay_buffer_every': [1, 10, 20],
        'training.id_weight': [1],
        'on_policy_only': [True, False],
        'encoder.init_weights': [False],
        'mlp_model.init_weights': [False],
        'training.center_rewards': [False],
        'ss_module.mask_p': [0.15],
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