import os
import random

def gen_command(config):
    command = "sbatch sweep/run_job_narval_v1.slurm 42"
    for key, value in config.items():
        command += " {}".format(value)
    return command


def run_random_job(fake_submit: bool = True):
    hparams = {
        'optimizer_actor.lr': [0.00003, 0.0001, 0.0003],
        'optimizer_critic.lr': [0.0001],
        'training.entropy_beta': [0.0001, 0.001, 0.003],
        'training.clip_range': [0.2, 0.3],
        'training.updates_per_batch': [2],
        'training.kl_threshold': [0.01],
        'hidden_size': [192],
        'encoder.num_layers': [3],
        'training.critic_loss_mode': ['td-1'],
        'max_cxt_len': [20, 30, 40],
        'optimizer_ss.lr': [0.00001, 0.00003, 0.0001, 0.0003],
        'use_ss_loss': [True],
        'training.actor_loss_mode': ['integrated_aa'],
        'training.aa_weight': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'env.batch': [6],
        'agent_rb_size': [400, 500],
        'training.add_to_agent_replay_buffer_every': [20],
        'training.id_weight': [1],
        'on_policy_only': [True, False],
        'encoder.init_weights': [False],
        'mlp_model.init_weights': [False],
        'training.center_rewards': [False],
        'ss_module.mask_p': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
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