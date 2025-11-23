#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --job-name=phase1_solver

# Phase 1: BS-PGD Attack with Hybrid Ensemble
# This script runs the main solver on the A100 GPU

echo "=================================================================="
echo "Phase 1: Local BS-PGD Solver"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "=================================================================="

# Load modules
module load GCC CUDA PyTorch torchvision

# Navigate to version3 directory
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

# Check GPU
echo ""
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Run solver with FAST parameters (optimized for speed)
# Original (slow): bs-steps=8, restarts=15, pgd-steps=150 → ~60-90 min
# Fast config: bs-steps=4, restarts=5, pgd-steps=80 → ~6-9 min (10x faster)
python -u main_solver.py \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --epsilon-min 1.0 \
    --epsilon-max 10.0 \
    --bs-steps 4 \
    --pgd-steps 80 \
    --restarts 5 \
    --alpha-factor 2.5 \
    --kappa 0.0 \
    --momentum 0.9 \
    --device cuda

exit_code=$?

echo ""
echo "=================================================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

exit $exit_code

