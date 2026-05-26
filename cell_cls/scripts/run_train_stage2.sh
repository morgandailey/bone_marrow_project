#!/bin/bash
#SBATCH --job-name=train_s2v3
#SBATCH --output=/work/u4001296/project1/cell_cls/logs/train_s2v3_%j.out
#SBATCH --error=/work/u4001296/project1/cell_cls/logs/train_s2v3_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --account=MST114560
#SBATCH --partition=normal

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate project1

cd /work/u4001296/project1/cell_cls

python train/train_stage2.py

echo "End: $(date)"