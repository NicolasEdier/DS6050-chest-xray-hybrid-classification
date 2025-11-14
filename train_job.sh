#!/bin/bash
#SBATCH --job-name=chest_xray
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8

# Load modules
module purge
module load anaconda

# Activate environment
source activate chest_xray

# Print info
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run training
python training/train.py \
    --model resnet50 \
    --data_dir ./processed_data \
    --image_dir ./NIH_ChestXray \
    --batch_size 32 \
    --epochs 25 \
    --lr 1e-4 \
    --num_workers 4

echo "Job finished: $(date)"
