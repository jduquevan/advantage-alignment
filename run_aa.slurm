#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=adv_algmt

# Remove one # to uncommment
#SBATCH --output=/home/mila/j/juan.duque/scratch/slurm_output/slurm-%j.out
#SBATCH --error=/home/mila/j/juan.duque/scratch/slurm_output/job-%j.out

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-01:59:00   
#SBATCH --gres=gpu:a100l:1

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
conda activate cogames
module load python/3.9

python train.py\
    seed=${1}\
    --config-name='eg.yaml'