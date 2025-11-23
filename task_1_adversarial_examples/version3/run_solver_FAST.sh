#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --job-name=phase1_FAST

# FAST MODE: ~1-2 minutes for 100 images
# Ultra-fast: 2 BS steps, 2 restarts, 30 PGD steps, 2 models only
# Quality: Slightly lower but still competitive

echo "=================================================================="
echo "Phase 1: BS-PGD Solver - ULTRA-FAST MODE"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Expected duration: ~1-2 minutes (10x faster than original)"
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

# FAST configuration (ULTRA-FAST: ~1-2 min for 100 images)
# Reduced: 2 BS steps, 2 restarts, 30 PGD steps, 2 models instead of 5
python -u main_solver.py \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --save-name submission_fast.npz \
    --epsilon-min 1.5 \
    --epsilon-max 8.0 \
    --bs-steps 2 \
    --pgd-steps 30 \
    --restarts 2 \
    --alpha-factor 2.5 \
    --kappa 0.0 \
    --momentum 0.9 \
    --fast-mode \
    --device cuda

exit_code=$?

echo ""
echo "=================================================================="
echo "Job completed: $(date)"
echo "Exit code: $exit_code"
echo "=================================================================="

exit $exit_code

