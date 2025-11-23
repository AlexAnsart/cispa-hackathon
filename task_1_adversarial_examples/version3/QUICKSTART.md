# Quick Start Guide - Phase 1 Solver

## TL;DR - Get Running in 3 Steps

```bash
cd /p/home/jusers/ansart1/jureca/code/task_1_adversarial_examples/version3

# 1. Launch attack on A100 GPU
sbatch run_solver.sh

# 2. Monitor progress
python monitor.py

# 3. Analyze results (after completion)
python analyze.py output/submission_run1.npz --mode api
```

---

## What This Does

Runs a sophisticated adversarial attack using:

- **Binary Search PGD**: Finds minimal L2 perturbation per image
- **15 Random Restarts**: Escapes local minima
- **5-Model Ensemble**: ResNet50, DenseNet121, VGG16_BN, EfficientNet_B0, ResNet18
- **Adaptive Hyperparameters**: Step size scales with epsilon
- **Per-Image Optimization**: Each image gets personalized attack

**Expected Runtime**: 60-90 minutes on A100 GPU for 100 images

**Expected Quality**: 
- Local success rate: >95%
- API success rate: >85%
- Average L2: ~4-6 (raw), ~0.12-0.18 (normalized)

---

## Workflow

### Step 1: Launch Attack

```bash
sbatch run_solver.sh
```

You'll get output like:
```
Submitted batch job 123456
```

### Step 2: Monitor Progress

**Check job status**:
```bash
squeue -u $USER
```

**Watch live output**:
```bash
tail -f logs/slurm_123456.out
```

**Quick stats dashboard**:
```bash
python monitor.py
```

Example monitor output:
```
📊 RUN INFORMATION
--------------------------------------------------
Total runs completed: 1
Last update: 2025-11-23 14:30:00

📈 LATEST STATISTICS
--------------------------------------------------
Total images:       100
Successful:         98 (98.0%)
Failed:             2

L2 Distances (average):
  All images:       4.2314
  Successful only:  4.1205
  Range:            [0.5432, 11.2345]
```

### Step 3: Results Are Ready

When job completes, you'll see:
```
Phase 1 Complete!
Next Steps:
1. Review logs in: ./logs
2. Submit using: python submit.py output/submission_run1.npz
3. Analyze results: python analyze.py output/submission_run1.npz
```

### Step 4: Analyze Results

**Fast local analysis** (no API call):
```bash
python analyze.py output/submission_run1.npz --mode local
```

Output:
```
L2 Distance Statistics:
  Average (normalized): 0.1234  ← Lower bound on leaderboard score
```

**True score from black-box** (15min cooldown):
```bash
python analyze.py output/submission_run1.npz --mode api
```

Output:
```
Success Rate: 87/100 (87.0%)
Leaderboard Score: 0.1876  ← This is your real score
```

### Step 5: Submit to Leaderboard

```bash
python submit.py output/submission_run1.npz
```

Output:
```
✓ Successfully submitted!
  Score: 0.187634
  Submission ID: abc123...
```

---

## Understanding the Output Files

### Submission File
- `output/submission_run1.npz`: Your adversarial examples (ready to submit)

### JSON Logs
- `logs/local_state.json`: Best result per image (persistent across runs)
- `logs/run_history.json`: Complete history of all runs
- `logs/stats_summary.json`: Quick stats (used by monitor.py)

### SLURM Logs
- `logs/slurm_123456.out`: Job output (progress messages)
- `logs/slurm_123456.err`: Errors (should be empty if successful)

### API Logs
- `logs/api/logits_*.json`: Logit responses from black-box
- `logs/api/submit_*.json`: Submission confirmations

---

## Troubleshooting

### Job Won't Start (Stuck in Queue)

**Check partition status**:
```bash
sinfo -p dc-gpu-devel
```

If many nodes are down, edit `run_solver.sh`:
```bash
#SBATCH --partition=dc-gpu        # Change from dc-gpu-devel
#SBATCH --time=04:00:00            # Increase time limit
```

### Job Fails Immediately

**Check error log**:
```bash
tail -n 50 logs/slurm_*.err
```

Common fixes:
- **Module error**: Run `module avail` to check available modules
- **Path error**: Verify `natural_images.pt` exists in parent directory
- **GPU error**: Ensure you're on `dc-gpu` or `dc-gpu-devel` partition

### Low Success Rate (<70% on API)

**Diagnosis**: Attacks not transferring to black-box.

**Phase 2 will fix this automatically**, but for now:

1. Check which images failed:
```bash
python analyze.py output/submission_run1.npz --mode api | grep FAILED
```

2. Manually increase their `kappa` in `logs/local_state.json`:
```json
{
  "images": {
    "42": {
      "kappa": 5.0,  ← Increase this (was 0.0)
      ...
    }
  }
}
```

3. Re-run solver (will use updated kappas):
```bash
sbatch run_solver.sh
```

---

## Interpreting Scores

### What is "Success"?

**Local success**: Surrogate ensemble predicts wrong class
- Target: >95%
- Achieved by BS-PGD optimization

**API success**: Black-box predicts wrong class
- Target: >85%
- Depends on how well surrogates match black-box

### What is "L2 Distance"?

**Raw L2**: Euclidean distance in pixel space
- For 28×28×3 images: typically 2-8
- Lower is better (less visible perturbation)

**Normalized L2**: Scaled to [0, 1]
- `normalized = raw / sqrt(3 × 28 × 28) = raw / 28.98`
- This is what the leaderboard uses

### What is "Leaderboard Score"?

```
score = average of all per-image scores
```

Per-image score:
- If **misclassified**: score = normalized L2 distance
- If **correctly classified**: score = 1.0 (penalty)

**Goal**: Minimize this score (lower is better)

### What is "Kappa (κ)"?

Confidence margin: how far past the decision boundary to push.

```
Success condition: logit_max_wrong - logit_true > κ
```

- `κ = 0.0`: Just barely cross boundary (minimal L2, but risky)
- `κ = 5.0`: Cross boundary with confidence (higher L2, but safer)

Phase 1 starts with `κ = 0.0` for all images.
Phase 2 will calibrate `κ` per image based on API feedback.

---

## Performance Expectations

### On A100 GPU (JURECA)

**Per image**:
- 8 binary search steps × 15 restarts × 150 PGD iterations
- 5 models in ensemble
- ~30-60 seconds per image

**Full 100 images**:
- Sequential processing (one at a time for memory efficiency)
- Total: 60-90 minutes

**Memory usage**:
- ~6-8 GB GPU memory
- Well within A100's 40 GB

### Quality Benchmarks

**Local (surrogate) success rate**: >95%
- If lower, increase `--restarts` or `--epsilon-max`

**API (black-box) success rate**: >85%
- If lower, use Phase 2 to calibrate kappas

**Average normalized L2** (successful attacks): 0.12-0.18
- Lower is better, but must maintain success rate

**Leaderboard score**: 0.15-0.20
- Competitive baseline
- Phase 2 refinement should push to 0.10-0.15

---

## Customization

### Run Faster (Lower Quality)

Edit `run_solver.sh`:
```bash
--bs-steps 5 \          # Default: 8
--pgd-steps 100 \       # Default: 150
--restarts 10 \         # Default: 15
```

**Expected**: 30-40 minutes, slightly higher L2

### Run Slower (Higher Quality)

```bash
--bs-steps 10 \         # More thorough epsilon search
--pgd-steps 200 \       # More optimization steps
--restarts 20 \         # More random restarts
```

**Expected**: 2-3 hours, slightly lower L2

### Attack Specific Images

Currently attacks all 100. To test on subset, modify `main_solver.py` (advanced).

---

## Next Steps (Phase 2)

Phase 2 will add automatic feedback loop:

1. **Auto-submit** every 5 minutes
2. **Query logits** every 15 minutes  
3. **Calibrate kappas** based on API feedback:
   - Failed → Increase κ (push stronger)
   - Overkill → Decrease κ (reduce L2)
4. **Iterative refinement** until convergence

Phase 1 provides the foundation:
- Robust local solver ✓
- Persistent state management ✓
- Production-ready logging ✓

---

## Support

**Full documentation**: See `README.md`

**Monitor progress**: `python monitor.py`

**Check logs**: 
- SLURM: `tail -f logs/slurm_*.out`
- Stats: `cat logs/stats_summary.json`

**API rate limits**:
- Logits: 15 minutes
- Submit: 5 minutes

