#!/bin/bash
#SBATCH --account=training2557
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --job-name=phase1_QUALITY

# QUALITY MODE: ~60-90 minutes for 100 images
# Maximum quality: More binary search steps, restarts, iterations

echo "=================================================================="
echo "Phase 1: BS-PGD Solver - QUALITY MODE"
echo "=================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Expected duration: ~60-90 minutes"
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

# QUALITY configuration (original, slower but better)
python -u main_solver.py \
    --dataset ../natural_images.pt \
    --output-dir ./output \
    --log-dir ./logs \
    --save-name submission_quality.npz \
    --epsilon-min 0.5 \
    --epsilon-max 12.0 \
    --bs-steps 8 \
    --pgd-steps 150 \
    --restarts 15 \
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

