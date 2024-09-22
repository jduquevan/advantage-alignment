import os
import random


def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True):
    script = gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = f'sbatch --time=12:0:0  --gres=gpu:a100:1 --account=rrg-bengioy-ad --cpus-per-task=6 --mem=40G --gres=gpu:1 -c 4 sweep_temp_job.sh  '
        os.system(command)


def run_random_jobs(fake_submit: bool = True):
    command = ("python train.py seed=42 "
               "optimizer_actor.lr={optimizer_actor__lr} "
               "optimizer_critic.lr={optimizer_critic__lr} "
               "optimizer_ss.lr={optimizer_ss__lr} "
               "training.entropy_beta={training__entropy_beta} "
               "training.clip_range=0.1 "
               "training.updates_per_batch=2 "
               "training.kl_threshold={training__kl_threshold} "
               "hidden_size=192 "
               "encoder.num_layers=3 "
               "training.critic_loss_mode=td-1 "
               "max_cxt_len={max_cxt_len} "
               "optimizer_ss.lr=0 "
               "use_ss_loss={use_ss_loss} "
               "ss_module.mask_p=0.15 "
               "training.actor_loss_mode={actor_loss_mode} "
               "training.aa_weight={training__aa_weight} "
               "env.batch=6 "
               "agent_rb_size=200 "
               "training.add_to_agent_replay_buffer_every=20 "
               "training.id_weight=0 "
               "debug_mode=False "
               "sum_rewards={sum_rewards} "
               "wandb_dir=/home/aghajohm/scratch/adalign/wandb_dir "
               "hydra.run.dir=/home/aghajohm/scratch/adalign/hydra_dir "
               "training.checkpoint_dir=/home/aghajohm/scratch/adalign/experiments "
               "training.video_dir=/home/aghajohm/scratch/adalign/videos/ "
               "wandb_tags=[commons_harvest,v102] "
               "--config-name=meltingpot.yaml ")

    hparams = {
        'optimizer_actor__lr': [1e-5, 3e-5, 1e-4],
        # 'optimizer_critic__lr': [1e-5, 3e-5, 1e-4],
        'training__entropy_beta': [0, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
        'training__kl_threshold': [1e-2, 5e-2],
        'max_cxt_len': [15, 20, 25, 30],
        'training__aa_weight': [0.5, 0.7, 0.8, 1.0, 2.0],
        'use_ss_loss': [True],
    }

    # sample a random config
    cfg = {}
    for key, values in hparams.items():
        cfg[key] = random.choice(values)

    cfg['optimizer_critic__lr'] = cfg['optimizer_actor__lr']
    cfg['optimizer_ss__lr'] = cfg['optimizer_actor__lr']

    # submit this job using slurm
    cfg['actor_loss_mode'] = 'integrated_aa'
    cfg['sum_rewards'] = False
    aa_command = command.format(**cfg)

    cfg['actor_loss_mode'] = 'naive'
    cfg['training.aa_weight'] = 0.0
    cfg['sum_rewards'] = False
    naive_command = command.format(**cfg)

    cfg['sum_rewards'] = True
    sum_reward_command = command.format(**cfg)

    # run them
    run_job(aa_command, fake_submit=fake_submit)
    run_job(naive_command, fake_submit=fake_submit)
    run_job(sum_reward_command, fake_submit=fake_submit)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_jobs(fake_submit=fake_submit)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire

    fire.Fire()