"""
Analysis Utility for Phase 1

Analyzes submission files and computes scores:
- Local mode: Compute L2 distances without API (lower bound)
- API mode: Query logits and compute actual score
"""

import torch
import numpy as np
import json
import os
import argparse
from datetime import datetime
from pathlib import Path


def load_dataset(dataset_path: str = "../natural_images.pt") -> dict:
    """Load natural images dataset."""
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        exit(1)
    return torch.load(dataset_path, weights_only=False)


def analyze_local(submission_path: str, dataset: dict) -> dict:
    """
    Local analysis: Compute L2 distances without API.
    
    This gives a LOWER BOUND on the actual score, since correctly
    classified images will be scored as 1.0 by the evaluator.
    """
    print("\n" + "=" * 70)
    print("Local Analysis (No API)")
    print("=" * 70)
    
    # Load submission
    sub = np.load(submission_path)
    adv_images = sub["images"]
    
    # Load originals
    original_images = dataset["images"].numpy()
    
    # Compute L2 distances
    N, C, H, W = original_images.shape
    l2_norm_factor = float(np.sqrt(C * H * W))
    
    diffs = adv_images - original_images
    l2_per_image = np.linalg.norm(diffs.reshape(len(diffs), -1), axis=1)
    l2_per_image_norm = np.clip(l2_per_image / l2_norm_factor, 0.0, 1.0)
    
    # Statistics
    avg_l2_raw = float(np.mean(l2_per_image))
    avg_l2_norm = float(np.mean(l2_per_image_norm))
    min_l2 = float(np.min(l2_per_image))
    max_l2 = float(np.max(l2_per_image))
    std_l2 = float(np.std(l2_per_image))
    
    print(f"\nL2 Distance Statistics:")
    print(f"  Average (raw):        {avg_l2_raw:.4f}")
    print(f"  Average (normalized): {avg_l2_norm:.6f}")
    print(f"  Min:                  {min_l2:.4f}")
    print(f"  Max:                  {max_l2:.4f}")
    print(f"  Std:                  {std_l2:.4f}")
    print(f"\nNOTE: This is a LOWER BOUND on the actual leaderboard score.")
    print(f"      Correctly classified images will be scored as 1.0 by the evaluator.")
    print(f"      Use --mode api to get the true score.")
    
    return {
        'mode': 'local',
        'timestamp': datetime.now().isoformat(),
        'submission_file': submission_path,
        'avg_l2_raw': avg_l2_raw,
        'avg_l2_normalized': avg_l2_norm,
        'min_l2': min_l2,
        'max_l2': max_l2,
        'std_l2': std_l2,
        'score_lower_bound': avg_l2_norm,
        'per_image': [
            {
                'image_id': int(i),
                'l2_raw': float(l2_per_image[i]),
                'l2_normalized': float(l2_per_image_norm[i])
            } for i in range(len(l2_per_image))
        ]
    }


def analyze_api(submission_path: str, dataset: dict, save_logits: bool = True) -> dict:
    """
    API analysis: Query logits and compute actual score.
    
    This gives the TRUE score as computed by the evaluator.
    """
    # Import submission utility for API calls (only when needed)
    from submit import get_logits
    
    print("\n" + "=" * 70)
    print("API Analysis")
    print("=" * 70)
    
    # Get logits from API
    logits_path = None
    if save_logits:
        log_dir = Path(submission_path).parent.parent / "logs" / "api"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(submission_path).stem
        logits_path = str(log_dir / f"logits_{base_name}_{timestamp}.json")
    
    logits_data = get_logits(submission_path, logits_path)
    
    # Load data
    sub = np.load(submission_path)
    adv_images = sub["images"]
    original_images = dataset["images"].numpy()
    true_labels = dataset["labels"].numpy()
    
    # Parse results
    results = logits_data["results"]
    results.sort(key=lambda x: x["image_id"])
    
    # Compute score
    N, C, H, W = original_images.shape
    l2_norm_factor = float(np.sqrt(C * H * W))
    
    scores_raw = []
    scores_norm = []
    successes = []
    per_image = []
    
    print("\n" + "=" * 70)
    print("Per-Image Results")
    print("=" * 70)
    print(f"{'ID':>3} | {'True':>4} | {'Pred':>4} | {'Status':>7} | {'L2 Raw':>8} | {'Score':>6}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        img_id = res["image_id"]
        logits = np.array(res["logits"])
        predicted_class = int(np.argmax(logits))
        true_class = int(true_labels[img_id])
        
        # Check misclassification
        is_misclassified = (predicted_class != true_class)
        
        # Compute L2
        diff = adv_images[i] - original_images[i]
        l2_dist = float(np.linalg.norm(diff))
        l2_dist_norm = float(np.clip(l2_dist / l2_norm_factor, 0.0, 1.0))
        
        # Score
        if is_misclassified:
            score_raw = l2_dist
            score_norm = l2_dist_norm
            status = "SUCCESS"
            successes.append(True)
        else:
            score_raw = 1.0
            score_norm = 1.0
            status = "FAILED"
            successes.append(False)
        
        scores_raw.append(score_raw)
        scores_norm.append(score_norm)
        
        # Print row
        print(f"{img_id:3d} | {true_class:4d} | {predicted_class:4d} | "
              f"{status:>7} | {l2_dist:8.4f} | {score_norm:6.4f}")
        
        per_image.append({
            'image_id': img_id,
            'true_label': true_class,
            'predicted_label': predicted_class,
            'misclassified': is_misclassified,
            'l2_raw': l2_dist,
            'l2_normalized': l2_dist_norm,
            'score': score_norm,
            'logits': logits.tolist()
        })
    
    # Aggregate statistics
    success_rate = (sum(successes) / len(successes)) * 100
    avg_score_raw = float(np.mean(scores_raw))
    avg_score_norm = float(np.mean(scores_norm))
    
    successful_l2 = [scores_norm[i] for i in range(len(scores_norm)) if successes[i]]
    failed_l2 = [scores_norm[i] for i in range(len(scores_norm)) if not successes[i]]
    
    avg_l2_success = float(np.mean(successful_l2)) if successful_l2 else 0.0
    avg_l2_failed = float(np.mean(failed_l2)) if failed_l2 else 0.0
    
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"\nSuccess Rate: {sum(successes)}/{len(successes)} ({success_rate:.1f}%)")
    print(f"\nLeaderboard Score (Normalized):")
    print(f"  Overall:           {avg_score_norm:.6f}")
    print(f"  Successful only:   {avg_l2_success:.6f}")
    print(f"  Failed (all 1.0):  {avg_l2_failed:.6f}")
    print(f"\nL2 Distance (Raw):")
    print(f"  Overall:           {avg_score_raw:.4f}")
    
    return {
        'mode': 'api',
        'timestamp': datetime.now().isoformat(),
        'submission_file': submission_path,
        'success_rate': success_rate,
        'num_successes': sum(successes),
        'num_failures': len(successes) - sum(successes),
        'leaderboard_score': avg_score_norm,
        'avg_score_raw': avg_score_raw,
        'avg_l2_success': avg_l2_success,
        'avg_l2_failed': avg_l2_failed,
        'per_image': per_image
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze adversarial examples submission',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('file', type=str,
                       help='Path to submission .npz file')
    parser.add_argument('--mode', type=str, choices=['local', 'api'],
                       default='local',
                       help='Analysis mode: local (fast, lower bound) or api (true score)')
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save analysis JSON (default: logs/analysis_<timestamp>.json)')
    parser.add_argument('--no-save-logits', action='store_true',
                       help='Do not save API logits response')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Run analysis
    if args.mode == 'local':
        analysis = analyze_local(args.file, dataset)
    else:
        analysis = analyze_api(args.file, dataset, save_logits=not args.no_save_logits)
    
    # Save analysis
    if args.output:
        output_path = args.output
    else:
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = log_dir / f"analysis_{args.mode}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {output_path}")


if __name__ == "__main__":
    main()

