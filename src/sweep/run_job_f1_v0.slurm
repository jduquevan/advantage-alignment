#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=aa_f1_v0

# Remove one # to uncommment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-00:29:00
#SBATCH --gres=gpu:a100l
#SBATCH --cpus-per-task=6

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate cogames
export WANDB_ENTITY="jduque"
export HYDRA_FULL_ERROR=1

python train.py \
    --config-name='f1.yaml'\
    seed=${1} \
    optimizer_actor.lr=${2} \
    optimizer_critic.lr=${3} \
    training.entropy_beta=${4} \
    training.clip_range=${5} \
    training.updates_per_batch=${6} \
    gru_model.hidden_size=${7} \
    mlp_model.in_size=${7} \
    linear_model.in_size=${7} \
    mlp_model.hidden_size=${8} \
    linear_model.hidden_size=${9} \
    gru_model.num_layers=${10} \
    mlp_model.num_layers=${11} \
    linear_model.num_hidden=${12} \

