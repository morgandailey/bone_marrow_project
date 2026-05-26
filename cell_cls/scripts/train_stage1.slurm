#!/bin/bash
#SBATCH --job-name=cell_cls_stage1
#SBATCH --partition=normal
#SBATCH --account=mst114560
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/u4001296/project1/cell_cls/logs/%j.out
#SBATCH --error=/work/u4001296/project1/cell_cls/logs/%j.err
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=morgan.dailey0326@gmail.com

ml nvhpc-hpcx-cuda12/24.7

mkdir -p /work/u4001296/project1/cell_cls/logs
mkdir -p /work/u4001296/project1/cell_cls/runs/stage1_v3

cd /work/u4001296/project1/cell_cls

/home/u4001296/.conda/envs/project1/bin/python train/train_stage1.py