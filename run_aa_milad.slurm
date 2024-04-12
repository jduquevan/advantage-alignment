#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=adv_algmt

# Remove one # to uncommment
#SBATCH --output=/home/mila/a/aghajohm/scratch/advantage_alignment/slurm_output/slurm-%j.out
#SBATCH --error=/home/mila/a/aghajohm/scratch/advantage_alignment/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-01:00:00
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
#SBATCH --mail-type=NONE
# Remember to set your email address here instead of nobody
#SBATCH --mail-user=aghajohm@mila.quebec


# Submit jobs.
export WANDB_ENTITY="miladink"

module purge
eval "$(conda shell.bash hook)"
conda activate aa_env

python train.py training.vanilla=False sum_rewards=False training.entropy_beta=0.2 training.aa_weight=100. self_play=True --config-name milad_mila.yaml 
