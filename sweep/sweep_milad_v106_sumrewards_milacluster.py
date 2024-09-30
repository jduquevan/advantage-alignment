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
        command = f'sbatch --time=24:0:0  --gres=gpu:l40s:1 --partition=long --cpus-per-task=6 --mem=64G  sweep_temp_job.sh  '
        os.system(command)


def run_random_jobs(fake_submit: bool = True):
    command = ("python train.py run_id={run_id} seed={seed} "
               "optimizer_actor.lr=1e-05 "
               "optimizer_critic.lr=1e-05 "
               "optimizer_ss.lr=1e-05 "
               "training.entropy_beta=0.1 "
               "training.clip_range=0.1 "
               "training.updates_per_batch=2 "
               "training.kl_threshold=0.05 "
               "hidden_size=192 "
               "encoder.num_layers=3 "
               "training.critic_loss_mode=td-1 "
               "max_cxt_len=15 "
               "optimizer_ss.lr=0 "
               "use_ss_loss=True "
               "ss_module.mask_p=0.15 "
               "training.actor_loss_mode=naive "
               "training.aa_weight=0.0 "
               "env.batch=6 "
               "agent_rb_size=200 "
               "training.add_to_agent_replay_buffer_every=20 "
               "training.id_weight=0 "
               "debug_mode=False "
               "sum_rewards=True "
               "wandb_dir=/home/mila/a/aghajohm/scratch/adalign/wandb_dir "
               "hydra.run.dir=/home/mila/a/aghajohm/scratch/adalign/hydra_dir "
               "training.checkpoint_dir=/home/mila/a/aghajohm/scratch/adalign/experiments "
               "training.video_dir=/home/mila/a/aghajohm/scratch/adalign/videos/ "
               "wandb_tags=[commons_harvest,v106,sumrewards] --config-name=meltingpot.yaml")

    for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
        run_job(command.format(seed=seed, run_id=f"iclr2025_sumrewards_{seed}"), fake_submit=fake_submit)

def main(num_jobs: int, fake_submit: bool = True):
    for i in range(num_jobs):
        run_random_jobs(fake_submit=fake_submit)

if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire

    fire.Fire()