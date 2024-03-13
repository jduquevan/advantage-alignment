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
#SBATCH --gres=gpu:a100l


# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate cogames
export WANDB_ENTITY="jduque"
export HYDRA_FULL_ERROR=1

python train.py \
    seed=${1} \
    optimizer_actor.lr=${2} \
    optimizer_critic.lr=${2} \
    optimizer_spr.lr=${2} \
    training.entropy_beta=${3} \
    training.clip_range=${4} \
    training.updates_per_batch=${5} \
    transformer_model.d_model=${6} \
    mlp_model.in_size=${6} \
    mlp_model.hidden_size=${6} \
    linear_model.in_size=${6} \
    spr_model.in_size=${6} \
    spr_model.out_size=${6} \
    transformer_model.num_layers=${7}
