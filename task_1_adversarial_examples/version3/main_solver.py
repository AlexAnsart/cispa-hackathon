"""
Phase 1: Local Solver with Persistent JSON Logging

This script implements the BS-PGD attack with comprehensive state tracking.
All progress is logged to JSON files for monitoring and Phase 2 integration.

Logging Structure:
- local_state.json: Current best results per image
- run_history.json: Detailed history of each solver run
- stats_summary.json: Quick stats for monitoring
"""

import torch
import numpy as np
import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from models import HybridEnsemble
from attack import BSPGD, AttackConfig, AttackResult, compute_success_stats


class PhaseOneSolver:
    """
    Phase 1 Solver: Local BS-PGD with State Management.
    
    Responsibilities:
    1. Run BS-PGD attack on all images
    2. Track best adversarial examples per image
    3. Log detailed state to JSON for Phase 2 integration
    4. Save submission files
    """
    
    def __init__(
        self, 
        dataset_path: str = "../natural_images.pt",
        output_dir: str = "./output",
        log_dir: str = "./logs",
        device: str = 'cuda'
    ):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # JSON file paths
        self.local_state_path = self.log_dir / "local_state.json"
        self.run_history_path = self.log_dir / "run_history.json"
        self.stats_path = self.log_dir / "stats_summary.json"
        
        # Load dataset
        print("=" * 70)
        print("Phase 1: Local Solver Initialization")
        print("=" * 70)
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"\nLoading dataset: {dataset_path}")
        data = torch.load(dataset_path, weights_only=False)
        self.images = data["images"]
        self.labels = data["labels"]
        self.image_ids = data["image_ids"].numpy()
        
        print(f"✓ Loaded {len(self.images)} images")
        print(f"  Shape: {self.images.shape}")
        print(f"  Labels: {self.labels.shape}")
        
        # Load or initialize local state
        self.local_state = self._load_local_state()
        
        # Load ensemble
        # Fast mode will be set from args in run() method
        self.fast_mode = False
        print(f"\nInitializing Hybrid Ensemble...")
        self.ensemble = None  # Will be initialized in run() with correct fast_mode
        
        print("=" * 70)
    
    def _load_local_state(self) -> Dict:
        """Load or initialize local state JSON."""
        if self.local_state_path.exists():
            with open(self.local_state_path, 'r') as f:
                state = json.load(f)
            print(f"\n✓ Loaded existing state from {self.local_state_path}")
            print(f"  Previous runs: {state.get('num_runs', 0)}")
            return state
        else:
            print(f"\n✓ Initializing new state")
            return {
                'num_runs': 0,
                'last_update': None,
                'images': {}  # {image_id: {best_l2, kappa, epsilon, success, ...}}
            }
    
    def _save_local_state(self):
        """Save current local state to JSON."""
        self.local_state['last_update'] = datetime.now().isoformat()
        with open(self.local_state_path, 'w') as f:
            json.dump(self.local_state, f, indent=2)
    
    def _load_kappas(self) -> Dict[int, float]:
        """Extract per-image kappas from local state."""
        kappas = {}
        for img_id_str, img_data in self.local_state.get('images', {}).items():
            img_id = int(img_id_str)
            kappas[img_id] = img_data.get('kappa', 0.0)
        return kappas
    
    def _update_local_state_with_results(self, results: list[AttackResult]):
        """Update local state with new attack results (only if better)."""
        for result in results:
            img_id_str = str(result.image_id)
            
            # Get existing best or initialize
            if img_id_str not in self.local_state['images']:
                self.local_state['images'][img_id_str] = {
                    'best_l2': float('inf'),
                    'kappa': 0.0,
                    'epsilon': 0.0,
                    'success': False,
                    'margin': -float('inf'),
                    'num_updates': 0
                }
            
            img_state = self.local_state['images'][img_id_str]
            
            # Update if better (lower L2 for success, or higher margin for failure)
            should_update = False
            if result.success and result.l2_distance < img_state['best_l2']:
                should_update = True
            elif not img_state['success'] and result.confidence_margin > img_state['margin']:
                should_update = True
            
            if should_update:
                img_state['best_l2'] = result.l2_distance
                img_state['kappa'] = result.kappa_used
                img_state['epsilon'] = result.epsilon_used
                img_state['success'] = result.success
                img_state['margin'] = result.confidence_margin
                img_state['num_updates'] += 1
                img_state['last_update'] = datetime.now().isoformat()
    
    def _save_run_history(self, run_id: int, config: AttackConfig, 
                         results: list[AttackResult], stats: dict, 
                         duration: float):
        """Append this run's details to run history."""
        # Load existing history
        if self.run_history_path.exists():
            with open(self.run_history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {'runs': []}
        
        # Create run entry
        run_entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'config': {
                'epsilon_min': config.epsilon_min,
                'epsilon_max': config.epsilon_max,
                'binary_search_steps': config.binary_search_steps,
                'pgd_steps': config.pgd_steps,
                'num_restarts': config.num_restarts,
                'alpha_factor': config.alpha_factor,
                'momentum': config.momentum,
                'use_input_diversity': config.use_input_diversity,
            },
            'stats': stats,
            'per_image_results': [
                {
                    'image_id': r.image_id,
                    'l2': r.l2_distance,
                    'epsilon': r.epsilon_used,
                    'kappa': r.kappa_used,
                    'success': r.success,
                    'margin': r.confidence_margin,
                    'num_restarts_succeeded': r.num_restarts_succeeded,
                } for r in results
            ]
        }
        
        history['runs'].append(run_entry)
        
        with open(self.run_history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _save_stats_summary(self, stats: dict):
        """Save quick stats for monitoring."""
        summary = {
            'last_update': datetime.now().isoformat(),
            'num_runs': self.local_state['num_runs'],
            'stats': stats,
        }
        
        with open(self.stats_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_adversarial_images(self, results: list[AttackResult], 
                                 filename: str = "submission.npz"):
        """Save adversarial images to submission format."""
        # Sort by image_id to ensure correct order
        results_sorted = sorted(results, key=lambda r: r.image_id)
        
        adv_images = torch.cat([r.adversarial for r in results_sorted], dim=0)
        image_ids_ordered = np.array([r.image_id for r in results_sorted])
        
        # Convert to numpy
        adv_np = adv_images.cpu().numpy().astype(np.float32)
        
        output_path = self.output_dir / filename
        np.savez_compressed(output_path, images=adv_np, image_ids=image_ids_ordered)
        
        print(f"\n✓ Saved submission to: {output_path}")
        print(f"  Shape: {adv_np.shape}")
        print(f"  Dtype: {adv_np.dtype}")
        
        return output_path
    
    def run(self, config: AttackConfig, save_name: Optional[str] = None, fast_mode: bool = False):
        """
        Run Phase 1 attack on all images.
        
        Args:
            config: Attack configuration
            save_name: Custom name for submission file (default: submission.npz)
            fast_mode: If True, use fewer models (2 instead of 5) for speed
        """
        # Initialize ensemble with correct fast_mode
        if self.ensemble is None or self.fast_mode != fast_mode:
            self.fast_mode = fast_mode
            print(f"\nInitializing Hybrid Ensemble (fast_mode={fast_mode})...")
            self.ensemble = HybridEnsemble(device=self.device, fast_mode=fast_mode)
        
        print("\n" + "=" * 70)
        print("Starting Phase 1 Attack")
        print("=" * 70)
        
        # Print config
        print("\nAttack Configuration:")
        print(f"  Epsilon range: [{config.epsilon_min}, {config.epsilon_max}]")
        print(f"  Binary search steps: {config.binary_search_steps}")
        print(f"  PGD steps per trial: {config.pgd_steps}")
        print(f"  Random restarts: {config.num_restarts}")
        print(f"  Alpha factor: {config.alpha_factor}")
        print(f"  Momentum: {config.momentum}")
        print(f"  Input diversity: {config.use_input_diversity}")
        print(f"  Base kappa: {config.kappa}")
        
        # Load per-image kappas from previous runs
        kappas = self._load_kappas()
        if kappas:
            kappa_values = list(kappas.values())
            print(f"\nLoaded {len(kappas)} per-image kappas from previous runs")
            print(f"  Kappa range: [{min(kappa_values):.3f}, {max(kappa_values):.3f}]")
            print(f"  Kappa mean: {np.mean(kappa_values):.3f}")
        else:
            print(f"\nNo previous kappas found, using default kappa={config.kappa}")
        
        # Initialize attacker
        print("\nInitializing BS-PGD attacker...")
        attacker = BSPGD(self.ensemble, config, device=self.device)
        
        # Run attack
        print("\n" + "=" * 70)
        print("Attacking Images")
        print("=" * 70)
        
        start_time = time.time()
        results = []
        
        num_images = len(self.images)
        for i in range(num_images):
            img_start = time.time()
            
            img = self.images[i:i+1]
            label = self.labels[i:i+1]
            img_id = int(self.image_ids[i])
            kappa = kappas.get(img_id, config.kappa)
            
            print(f"\n[{i+1:3d}/{num_images}] Image ID {img_id:3d} (Label: {label[0].item():3d}, κ={kappa:.2f})")
            
            result = attacker.binary_search_epsilon(
                img.to(self.device),
                label.to(self.device),
                img_id,
                kappa
            )
            
            results.append(result)
            
            img_duration = time.time() - img_start
            
            # Calculate normalized L2 for display
            # Normalization factor: sqrt(C*H*W) = sqrt(3*28*28) = sqrt(2352) ≈ 48.5
            C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
            l2_norm_factor = np.sqrt(C * H * W)
            l2_normalized = result.l2_distance / l2_norm_factor
            
            # Print result
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            print(f"  {status}")
            print(f"  L2: {result.l2_distance:.4f} (norm: {l2_normalized:.4f}) | " +
                  f"ε: {result.epsilon_used:.3f} | Margin: {result.confidence_margin:+.2f} | " +
                  f"Time: {img_duration:.1f}s")
            print(f"  BS steps with success: {result.num_restarts_succeeded}/{config.binary_search_steps}")
            print(f"  [ε = max L2 perturbation allowed (raw), Margin = logit_wrong - logit_true]")
            
            # Progress estimate
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (num_images - i - 1)
                print(f"\n  Progress: {i+1}/{num_images} | " +
                      f"Elapsed: {elapsed/60:.1f}min | " +
                      f"Remaining: ~{remaining/60:.1f}min")
        
        total_duration = time.time() - start_time
        
        # Compute statistics
        print("\n" + "=" * 70)
        print("Attack Complete - Computing Statistics")
        print("=" * 70)
        
        stats = compute_success_stats(results)
        
        print(f"\nResults:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Successful:   {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"  Failed:       {stats['failed']}")
        # Calculate normalized L2 for stats
        C, H, W = self.images.shape[1], self.images.shape[2], self.images.shape[3]
        l2_norm_factor = np.sqrt(C * H * W)
        
        print(f"\nL2 Distances:")
        print(f"  Average (all, raw):     {stats['avg_l2_all']:.4f}")
        print(f"  Average (all, norm):    {stats['avg_l2_all']/l2_norm_factor:.6f}")
        print(f"  Average (success, raw): {stats['avg_l2_success']:.4f}")
        print(f"  Average (success, norm): {stats['avg_l2_success']/l2_norm_factor:.6f}")
        print(f"  Range (raw):            [{stats['min_l2']:.4f}, {stats['max_l2']:.4f}]")
        print(f"  Range (norm):           [{stats['min_l2']/l2_norm_factor:.6f}, {stats['max_l2']/l2_norm_factor:.6f}]")
        print(f"\nConfidence Margins:")
        print(f"  Average (all):     {stats['avg_margin_all']:+.3f}")
        print(f"  Average (success): {stats['avg_margin_success']:+.3f}")
        print(f"  Average (failed):  {stats['avg_margin_failed']:+.3f}")
        print(f"\nAverage epsilon used: {stats['avg_epsilon']:.3f}")
        print(f"\nTotal duration: {total_duration/60:.2f} minutes")
        
        # Update and save state
        print("\n" + "=" * 70)
        print("Updating State and Saving Results")
        print("=" * 70)
        
        self.local_state['num_runs'] += 1
        run_id = self.local_state['num_runs']
        
        self._update_local_state_with_results(results)
        self._save_local_state()
        print(f"✓ Updated local_state.json")
        
        self._save_run_history(run_id, config, results, stats, total_duration)
        print(f"✓ Saved run history (Run #{run_id})")
        
        self._save_stats_summary(stats)
        print(f"✓ Updated stats_summary.json")
        
        # Save adversarial images
        submission_name = save_name if save_name else f"submission_run{run_id}.npz"
        self._save_adversarial_images(results, submission_name)
        
        print("\n" + "=" * 70)
        print("Phase 1 Complete!")
        print("=" * 70)
        print(f"\nNext Steps:")
        print(f"1. Review logs in: {self.log_dir}")
        print(f"2. Submit using: python submit.py {self.output_dir / submission_name}")
        print(f"3. Analyze results: python analyze.py {self.output_dir / submission_name}")
        
        return results, stats


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Local BS-PGD Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # I/O paths
    parser.add_argument('--dataset', type=str, default='../natural_images.pt',
                       help='Path to natural_images.pt')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for submissions')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Log directory for JSON state files')
    parser.add_argument('--save-name', type=str, default=None,
                       help='Custom name for submission file')
    
    # Attack config
    parser.add_argument('--epsilon-min', type=float, default=0.5,
                       help='Minimum epsilon for binary search')
    parser.add_argument('--epsilon-max', type=float, default=12.0,
                       help='Maximum epsilon for binary search')
    parser.add_argument('--bs-steps', type=int, default=8,
                       help='Binary search steps')
    parser.add_argument('--pgd-steps', type=int, default=150,
                       help='PGD iterations per trial')
    parser.add_argument('--restarts', type=int, default=15,
                       help='Number of random restarts')
    parser.add_argument('--alpha-factor', type=float, default=2.5,
                       help='Alpha factor (alpha = factor * epsilon / steps)')
    parser.add_argument('--kappa', type=float, default=0.0,
                       help='Base confidence margin (will use per-image if available)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='MI-FGSM momentum')
    parser.add_argument('--no-diversity', action='store_true',
                       help='Disable input diversity')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Fast mode: use fewer models (2 instead of 5) for speed')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create config
    config = AttackConfig(
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        binary_search_steps=args.bs_steps,
        pgd_steps=args.pgd_steps,
        num_restarts=args.restarts,
        alpha_factor=args.alpha_factor,
        kappa=args.kappa,
        use_input_diversity=not args.no_diversity,
        momentum=args.momentum,
    )
    
    # Initialize solver
    solver = PhaseOneSolver(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Run attack
    solver.run(config, save_name=args.save_name, fast_mode=args.fast_mode)


if __name__ == '__main__':
    main()

