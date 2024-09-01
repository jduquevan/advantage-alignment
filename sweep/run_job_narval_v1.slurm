#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=meltingpot

# Remove one # to uncommment
#SBATCH --output=/scratch/jduque/slurm_output/slurm-%j.out
#SBATCH --error=/scratch/jduque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=0-11:59:00   
#SBATCH --gres=gpu:1 

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=juan.duque@mila.quebec


# Submit jobs.
version=4
export WANDB_ENTITY="jduque"

module purge
eval "$(conda shell.bash hook)"
source /home/jduque/environments/meltingpot/bin/activate
module load cuda/12.2
export WANDB_MODE=offline

python train.py \
    seed=${1} \
    optimizer_actor.lr=${2} \
    optimizer_critic.lr=${3} \
    training.entropy_beta=${4} \
    training.clip_range=${5} \
    training.updates_per_batch=${6} \
    training.kl_threshold=${7} \
    hidden_size=${8} \
    encoder.num_layers=${9} \
    training.critic_loss_mode=${10} \
    max_cxt_len=${11} \
    optimizer_ss.lr=${12} \
    use_ss_loss=${13} \
    training.actor_loss_mode=${14} \
    training.aa_weight=${15} \
    env.batch=${15} \
    debug_mode=False \
    wandb_dir=/home/jduque/scratch/wandb \
    hydra.run.dir=/home/jduque/scratch/hydra \
    --config-name='meltingpot.yaml' \
