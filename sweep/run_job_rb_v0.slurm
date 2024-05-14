#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=aa_v0

# Remove one # to uncommment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-00:59:00
#SBATCH --gres=gpu:a100l:1


# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate cogames
export WANDB_ENTITY="jduque"
export HYDRA_FULL_ERROR=1

python train.py \
    seed=${1} \
    agent_replay_buffer.size=${2} \
    agent_replay_buffer.update_every_step=${3} \
    agent_replay_buffer.off_policy_ratio=${4} \
    replay_buffer.mode=disabled \
    --config-name='eg.yaml' \