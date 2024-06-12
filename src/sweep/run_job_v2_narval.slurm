#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=aa_v0

# Remove one # to uncommment
#SBATCH --output=/scratch/jduque/slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --account=rrg-bengioy-ad
#SBATCH --mem=40G
#SBATCH --time=0-02:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
source /home/jduque/environments/f1/bin/activate
module load cuda/12.2
export WANDB_MODE=offline
export WANDB_ENTITY="jduque"
export HYDRA_FULL_ERROR=1

python train.py \
    wandb='offline' \
    wandb_dir='/home/jduque/scratch/wandb' \
    hydra.run.dir='/home/jduque/scratch/hydra' \
    seed=${1} \
    optimizer_actor.lr=${2} \
    optimizer_critic.lr=${3} \
    training.entropy_beta=${4} \
    training.clip_range=${5} \
    training.updates_per_batch=${6} \
    gru_model.hidden_size=${7} \
    mlp_model.hidden_size=${8} \
    linear_model.hidden_size=${9} \
    gru_model.num_layers=${10} \
    mlp_model.num_layers=${11} \
    linear_model.num_hidden=${12} \
    training.clip_grad_norm=${13} \
    optimizer_actor.weight_decay=${14} \
    training.aa_weight=${15} \
    mlp_model.in_size=${16} \
    linear_model.in_size=${17} \
    
