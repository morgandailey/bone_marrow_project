#!/bin/bash
#SBATCH --job-name=wsi_s23_bas
#SBATCH --output=/work/u4001296/project1/cell_cls/logs/wsi_s23_bas_%j.out
#SBATCH --error=/work/u4001296/project1/cell_cls/logs/wsi_s23_bas_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --account=MST114560
#SBATCH --partition=normal

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate project1

cd /work/u4001296/project1/cell_cls

python infer/wsi_detect_classify.py \
    --wsi       /work/u4001296/project1/data/wsi/slide-2025-10-01T13-44-33-R1-S23.mrxs \
    --ckpt      /work/u4001296/project1/cell_cls/runs/stage2_v3/best.pt \
    --output_dir /work/u4001296/project1/cell_cls/runs/wsi_scan_s23_s2v3 \
    --conf      0.5 \
    --cellpose_model cyto2

echo "End: $(date)"