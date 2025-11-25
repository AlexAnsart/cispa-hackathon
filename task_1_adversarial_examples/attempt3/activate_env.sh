#!/bin/bash
# Activation script for version3 environment
# Usage: source activate_env.sh

VENV_PATH="/p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated venv: $(which python)"
    echo "✓ Python version: $(python --version)"
    echo ""
    echo "You can now run:"
    echo "  python analyze.py output/submission_fast.npz --mode local"
    echo "  python analyze.py output/submission_fast.npz --mode api"
    echo "  python submit.py output/submission_fast.npz --action submit"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

