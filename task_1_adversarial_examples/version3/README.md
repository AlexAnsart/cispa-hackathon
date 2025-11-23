# Phase 1: Local BS-PGD Solver

PhD-level implementation of Binary Search PGD with Hybrid Ensemble for adversarial example generation.

## Architecture Overview

### Core Algorithm: BS-PGD (Binary Search PGD)

For each image, we perform binary search on epsilon to find the **minimal L2 perturbation** that satisfies:

```
logit_max_wrong - logit_true > κ_i
```

Where `κ_i` is a per-image confidence margin (initially 0, calibrated in Phase 2).

### Key Features

1. **Adaptive Step Size**: `α = 2.5 × ε / N_steps`
   - Ensures fine-grained optimization regardless of epsilon magnitude

2. **Random Restarts** (15 parallel attempts)
   - Escapes local minima in non-convex loss landscape
   - Keeps best candidate across all restarts

3. **Hybrid Ensemble Strategy**:
   - **Group A**: ImageNet models (ResNet50, DenseNet121, VGG16_BN, EfficientNet_B0) with 28→224 upsampling
   - **Group B**: Smaller models (ResNet18) with 28→32 upsampling
   - **Rationale**: We don't know if black-box uses large pretrained models or native small models

4. **Input Diversity**:
   - Random scaling/padding adapted for 28×28 images
   - Improves transferability without destroying information

5. **MI-FGSM Momentum** (0.9):
   - Stabilizes gradient direction across iterations
   - Better transferability than vanilla PGD

6. **Best Candidate Tracking**:
   - Keeps lowest L2 that satisfies success condition
   - Prevents oscillation artifacts

### State Management (JSON Logging)

All progress is logged for Phase 2 integration:

- **`local_state.json`**: Best results per image (updated incrementally)
- **`run_history.json`**: Complete history of all runs
- **`stats_summary.json`**: Quick stats for monitoring

## File Structure

```
version3/
├── models.py              # Hybrid ensemble (ImageNet + CIFAR models)
├── attack.py              # BS-PGD implementation with random restarts
├── main_solver.py         # Main orchestrator with JSON logging
├── submit.py              # API submission utility
├── analyze.py             # Result analysis (local/API modes)
├── run_solver.sh          # SLURM batch script for A100 GPU
├── README.md              # This file
├── output/                # Generated submission files (.npz)
└── logs/
    ├── local_state.json      # Per-image best results
    ├── run_history.json      # Complete run history
    ├── stats_summary.json    # Quick stats
    ├── slurm_*.out           # SLURM job output
    └── api/                  # API responses (logits, submissions)
```

## Usage Guide

### 1. Launch Attack (SLURM Batch)

**Recommended**: Use SLURM to run on A100 GPU.

```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3
sbatch run_solver.sh
```

**Monitor progress**:
```bash
# Check job status
squeue -u $USER

# Watch output in real-time (once job starts)
tail -f logs/slurm_<job_id>.out

# Check for errors
tail -f logs/slurm_<job_id>.err
```

### 2. Monitor Progress

While the job is running, check quick stats:

```bash
cat logs/stats_summary.json
```

This shows:
- Success rate on local ensemble
- Average L2 distances
- Confidence margins

### 3. Analyze Results (Local Mode)

Fast local analysis (no API call):

```bash
python analyze.py output/submission_run1.npz --mode local
```

Output:
- L2 distance statistics
- **LOWER BOUND** on leaderboard score (assumes all succeed)

### 4. Analyze Results (API Mode)

True score from black-box classifier:

```bash
python analyze.py output/submission_run1.npz --mode api
```

Output:
- Per-image predictions and L2 distances
- Success rate (actual misclassifications)
- **TRUE LEADERBOARD SCORE**

⚠️ **Rate limit**: Can only query logits once per 15 minutes.

### 5. Submit to Leaderboard

```bash
python submit.py output/submission_run1.npz --action submit
```

⚠️ **Rate limit**: Can only submit once per 5 minutes.

**Combined workflow** (get logits + submit):
```bash
python submit.py output/submission_run1.npz --action both
```

## Interpreting Results

### Local State JSON

`logs/local_state.json` structure:

```json
{
  "num_runs": 3,
  "last_update": "2025-11-23T14:30:00",
  "images": {
    "0": {
      "best_l2": 4.2314,
      "kappa": 0.0,
      "epsilon": 6.5,
      "success": true,
      "margin": 5.3,
      "num_updates": 2,
      "last_update": "2025-11-23T14:30:00"
    },
    ...
  }
}
```

**Key fields**:
- `best_l2`: Lowest L2 distance achieving local success
- `kappa`: Confidence margin used (Phase 2 will calibrate this)
- `success`: Whether local ensemble was fooled
- `margin`: `logit_max_wrong - logit_true`

### Analysis Output

**Local mode** (`--mode local`):
```
L2 Distance Statistics:
  Average (normalized): 0.1234  ← Lower bound on score
  Min:                  0.0543
  Max:                  0.9821
```

**API mode** (`--mode api`):
```
Success Rate: 87/100 (87.0%)
Leaderboard Score: 0.1876  ← True score
  Successful only:   0.1234
  Failed (all 1.0):  1.0000
```

**Interpretation**:
- If `success_rate < 100%`, some attacks failed to transfer
- Failed images get score = 1.0 (maximum penalty)
- Goal: Maximize success rate, minimize L2 for successes

### Understanding Success vs. Failure

**Local success but API failure** → Attack didn't transfer
- **Diagnosis**: Black-box boundary is farther than surrogate
- **Phase 2 action**: Increase `kappa` for this image (push attack stronger)

**Local failure** → Need larger epsilon or more restarts
- Rare with current config (15 restarts, ε_max=12)
- Check `binary_search_path` in `run_history.json`

## Customization

### Adjust Attack Strength

Edit `run_solver.sh` parameters:

```bash
# More aggressive search (wider epsilon range)
--epsilon-min 1.0 \
--epsilon-max 15.0 \

# More thorough optimization
--bs-steps 10 \        # More binary search iterations
--pgd-steps 200 \      # More PGD steps per trial
--restarts 20 \        # More random restarts

# Faster testing (lower quality)
--bs-steps 5 \
--pgd-steps 100 \
--restarts 10
```

### Run on Different Images

Currently processes all 100 images. To test on subset:

```bash
# Interactive testing on login node (NOT recommended for full run)
python main_solver.py --device cpu --pgd-steps 50 --restarts 3
```

### Save Custom Filename

```bash
python main_solver.py --save-name my_experiment.npz
```

## Performance Expectations

**On A100 GPU** (JURECA compute node):

- **Per image**: ~30-60 seconds
  - 8 binary search steps
  - 15 restarts × 150 PGD steps each
  - 5 models in ensemble

- **Full 100 images**: ~60-90 minutes

**Current config is optimized for quality, not speed.**

## Troubleshooting

### Job Stuck in Queue

```bash
# Check partition status
sinfo -p dc-gpu-devel

# If devel partition is full, try main partition (longer queue, longer time limit)
# Edit run_solver.sh:
#SBATCH --partition=dc-gpu
#SBATCH --time=04:00:00
```

### Out of Memory

Reduce batch processing or ensemble size:

Edit `models.py` line 36-41 to use fewer models:
```python
configs = [
    ('ResNet50', ...),
    ('DenseNet121', ...),
    # Comment out others if OOM
]
```

### Low Success Rate on API

If local success rate is high (>95%) but API success rate is low (<70%):

**Root cause**: Attacks not transferring to black-box.

**Phase 2 solution**: Calibrate `kappa` values:
1. Identify failed images from `analyze.py --mode api`
2. Increase their `kappa` in `local_state.json`
3. Re-run solver (it will use updated kappas)

**Example**:
```json
{
  "images": {
    "42": {
      "kappa": 5.0,  ← Increased from 0.0
      ...
    }
  }
}
```

### SLURM Job Fails Immediately

Check error log:
```bash
tail -n 50 logs/slurm_<job_id>.err
```

Common issues:
- Module load failed → Check if PyTorch module is available
- File not found → Check paths in `run_solver.sh`
- CUDA error → Wrong partition (need dc-gpu, not dc-cpu)

## What's Next (Phase 2)

Phase 2 will add the feedback loop:

1. **Automatic Submission** every 5 minutes
2. **Logit Analysis** every 15 minutes
3. **Kappa Calibration**:
   - Failed images → Increase κ (push stronger)
   - Overkill images → Decrease κ (reduce L2)
4. **Iterative Refinement** until convergence

Phase 1 provides the foundation: robust local solver + persistent state.

## Quick Reference Commands

```bash
# Launch attack
sbatch run_solver.sh

# Monitor job
squeue -u $USER
tail -f logs/slurm_*.out

# Quick stats
cat logs/stats_summary.json

# Analyze locally (fast)
python analyze.py output/submission_run1.npz --mode local

# Analyze with API (true score, 15min cooldown)
python analyze.py output/submission_run1.npz --mode api

# Submit to leaderboard (5min cooldown)
python submit.py output/submission_run1.npz

# Combined: logits + submit
python submit.py output/submission_run1.npz --action both
```

## Performance Metrics

**Target Goals**:
- Success rate: >95% (local), >85% (API)
- Average L2 (successful): <5.0 raw, <0.15 normalized
- Leaderboard score: <0.20 (lower is better)

**Current baseline** (to beat):
- Naive PGD: ~0.25-0.30
- This implementation should achieve: ~0.15-0.20

---

**Implementation Level**: PhD-tier
- Binary search optimization
- Multi-restart parallelization  
- Hybrid ensemble strategy
- Persistent state management
- Production-ready logging

**Ready for Phase 2 integration.**

