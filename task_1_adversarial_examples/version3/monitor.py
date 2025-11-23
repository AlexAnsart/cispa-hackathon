#!/usr/bin/env python3
"""
Quick Monitoring Utility for Phase 1 Solver

Displays real-time progress and statistics from JSON logs.
Run this while the solver is running to see current state.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime


def load_json_safe(path):
    """Load JSON file, return None if not found or invalid."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None


def format_timestamp(iso_str):
    """Format ISO timestamp to readable string."""
    if iso_str is None:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_str


def print_separator(char="=", width=70):
    print(char * width)


def main():
    log_dir = Path("./logs")
    
    local_state_path = log_dir / "local_state.json"
    stats_path = log_dir / "stats_summary.json"
    run_history_path = log_dir / "run_history.json"
    
    print_separator()
    print("Phase 1 Solver - Live Monitor")
    print_separator()
    print(f"Monitoring: {log_dir.absolute()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator()
    
    # Load state files
    local_state = load_json_safe(local_state_path)
    stats = load_json_safe(stats_path)
    run_history = load_json_safe(run_history_path)
    
    # Check if solver has run
    if local_state is None:
        print("\n❌ No state found. Solver hasn't run yet.")
        print("\nTo start the solver:")
        print("  sbatch run_solver.sh")
        return
    
    # Display run info
    print("\n📊 RUN INFORMATION")
    print_separator("-")
    print(f"Total runs completed: {local_state.get('num_runs', 0)}")
    print(f"Last update: {format_timestamp(local_state.get('last_update'))}")
    
    # Display latest stats
    if stats:
        print("\n📈 LATEST STATISTICS")
        print_separator("-")
        s = stats.get('stats', {})
        print(f"Total images:       {s.get('total_images', 0)}")
        print(f"Successful:         {s.get('successful', 0)} ({s.get('success_rate', 0):.1f}%)")
        print(f"Failed:             {s.get('failed', 0)}")
        print(f"\nL2 Distances (average):")
        print(f"  All images:       {s.get('avg_l2_all', 0):.4f}")
        print(f"  Successful only:  {s.get('avg_l2_success', 0):.4f}")
        print(f"  Range:            [{s.get('min_l2', 0):.4f}, {s.get('max_l2', 0):.4f}]")
        print(f"\nConfidence Margins:")
        print(f"  Successful:       {s.get('avg_margin_success', 0):+.3f}")
        print(f"  Failed:           {s.get('avg_margin_failed', 0):+.3f}")
        print(f"\nAverage epsilon:    {s.get('avg_epsilon', 0):.3f}")
    
    # Display per-image state summary
    if local_state and 'images' in local_state:
        images = local_state['images']
        num_images = len(images)
        
        if num_images > 0:
            print(f"\n🎯 PER-IMAGE STATE ({num_images} images)")
            print_separator("-")
            
            # Aggregate stats
            all_l2 = [img['best_l2'] for img in images.values()]
            all_success = [img['success'] for img in images.values()]
            all_kappa = [img['kappa'] for img in images.values()]
            all_margin = [img['margin'] for img in images.values()]
            
            num_success = sum(all_success)
            
            print(f"Success rate:       {num_success}/{num_images} ({num_success/num_images*100:.1f}%)")
            print(f"\nL2 distances:")
            print(f"  Average:          {sum(all_l2)/len(all_l2):.4f}")
            print(f"  Min:              {min(all_l2):.4f}")
            print(f"  Max:              {max(all_l2):.4f}")
            print(f"\nKappa values:")
            print(f"  Average:          {sum(all_kappa)/len(all_kappa):.3f}")
            print(f"  Range:            [{min(all_kappa):.3f}, {max(all_kappa):.3f}]")
            print(f"\nMargins:")
            print(f"  Average:          {sum(all_margin)/len(all_margin):+.3f}")
            
            # Show worst performers (highest L2)
            print(f"\n🔴 WORST 5 IMAGES (Highest L2):")
            print(f"{'ID':>5} | {'L2':>8} | {'κ':>6} | {'Status':>7} | {'Margin':>8}")
            print_separator("-")
            
            sorted_images = sorted(
                [(int(img_id), data) for img_id, data in images.items()],
                key=lambda x: x[1]['best_l2'],
                reverse=True
            )[:5]
            
            for img_id, data in sorted_images:
                status = "SUCCESS" if data['success'] else "FAILED"
                print(f"{img_id:5d} | {data['best_l2']:8.4f} | "
                      f"{data['kappa']:6.2f} | {status:>7} | {data['margin']:+8.2f}")
    
    # Display run history
    if run_history and 'runs' in run_history:
        runs = run_history['runs']
        if runs:
            print(f"\n📜 RUN HISTORY ({len(runs)} runs)")
            print_separator("-")
            
            print(f"{'Run':>3} | {'Date/Time':>19} | {'Success Rate':>12} | "
                  f"{'Avg L2':>8} | {'Duration':>10}")
            print_separator("-")
            
            for run in runs[-5:]:  # Show last 5 runs
                run_id = run['run_id']
                timestamp = format_timestamp(run['timestamp'])
                success_rate = run['stats'].get('success_rate', 0)
                avg_l2 = run['stats'].get('avg_l2_all', 0)
                duration = run['duration_seconds'] / 60
                
                print(f"{run_id:3d} | {timestamp} | "
                      f"{success_rate:11.1f}% | {avg_l2:8.4f} | {duration:8.1f}min")
    
    # Check for active SLURM jobs
    print("\n🖥️  SLURM JOB STATUS")
    print_separator("-")
    
    import subprocess
    try:
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', 'ansart1'), '--name=phase1_solver', '--format=%i %T %M %N'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) <= 1:
            print("No active jobs found.")
        else:
            print(lines[0])  # Header
            print_separator("-")
            for line in lines[1:]:
                print(line)
    except:
        print("Could not query SLURM (squeue not available or timed out)")
    
    # Display latest log file
    slurm_logs = sorted(log_dir.glob("slurm_*.out"))
    if slurm_logs:
        latest_log = slurm_logs[-1]
        print(f"\n📄 LATEST LOG FILE")
        print_separator("-")
        print(f"File: {latest_log.name}")
        print(f"\nLast 10 lines:")
        print_separator("-")
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())
        except:
            print("Could not read log file")
    
    print("\n" + "=" * 70)
    print("Monitor complete. Run again to refresh.")
    print("\nUseful commands:")
    print("  tail -f logs/slurm_*.out     # Watch live output")
    print("  squeue -u $USER              # Check job status")
    print("  python monitor.py            # Refresh this view")
    print("=" * 70)


if __name__ == "__main__":
    main()

