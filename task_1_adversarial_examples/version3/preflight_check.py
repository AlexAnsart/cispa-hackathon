#!/usr/bin/env python3
"""
Pre-flight Check Script

Verifies that all dependencies and files are in place before running the solver.
Run this before submitting the SLURM job to catch issues early.
"""

import sys
import os
from pathlib import Path
import subprocess


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_item(description, condition, fix_hint=None):
    """Check a condition and print status."""
    status = "✓" if condition else "✗"
    print(f"{status} {description}")
    if not condition and fix_hint:
        print(f"  → {fix_hint}")
    return condition


def check_python_import(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def main():
    all_checks_passed = True
    
    print_header("Phase 1 Solver - Pre-flight Check")
    
    # Check 1: File structure
    print("\n📁 FILE STRUCTURE")
    print("-" * 70)
    
    version3_dir = Path(__file__).parent
    os.chdir(version3_dir)
    
    required_files = [
        ("models.py", "Core model ensemble"),
        ("attack.py", "BS-PGD implementation"),
        ("main_solver.py", "Main orchestrator"),
        ("submit.py", "Submission utility"),
        ("analyze.py", "Analysis utility"),
        ("monitor.py", "Monitoring dashboard"),
        ("run_solver.sh", "SLURM batch script"),
    ]
    
    for filename, description in required_files:
        exists = (version3_dir / filename).exists()
        all_checks_passed &= check_item(
            f"{filename:20s} - {description}",
            exists,
            f"File missing! Implementation incomplete."
        )
    
    # Check 2: Dataset
    print("\n📦 DATASET")
    print("-" * 70)
    
    dataset_path = version3_dir.parent / "natural_images.pt"
    dataset_exists = dataset_path.exists()
    all_checks_passed &= check_item(
        f"natural_images.pt exists at {dataset_path.parent}",
        dataset_exists,
        "Download dataset or adjust path in main_solver.py"
    )
    
    if dataset_exists:
        size_mb = dataset_path.stat().st_size / 1e6
        check_item(f"  Size: {size_mb:.2f} MB", size_mb > 0.5 and size_mb < 10)
    
    # Check 3: Python environment
    print("\n🐍 PYTHON ENVIRONMENT")
    print("-" * 70)
    
    python_version = sys.version.split()[0]
    check_item(f"Python version: {python_version}", True)
    
    required_modules = [
        ("torch", "PyTorch (core deep learning)"),
        ("torchvision", "PyTorch vision models"),
        ("numpy", "Numerical computing"),
        ("requests", "HTTP requests for API"),
    ]
    
    for module, description in required_modules:
        available = check_python_import(module)
        all_checks_passed &= check_item(
            f"{module:15s} - {description}",
            available,
            f"Install with: pip install {module}"
        )
    
    # Check 4: PyTorch CUDA
    print("\n🎮 GPU SUPPORT")
    print("-" * 70)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        all_checks_passed &= check_item(
            "CUDA available",
            cuda_available,
            "PyTorch not compiled with CUDA or no GPU. Solver will be SLOW on CPU."
        )
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            check_item(f"  GPU: {device_name}", True)
            check_item(f"  Memory: {device_memory:.1f} GB", device_memory > 10)
            
            if "A100" in device_name:
                print("  🚀 A100 detected! Optimal performance expected.")
            elif "RTX" in device_name:
                print("  ⚠️  RTX GPU detected. Performance OK but not optimal.")
                print("     Consider running on compute nodes for A100 access.")
    except Exception as e:
        all_checks_passed = False
        print(f"✗ Could not check CUDA: {e}")
    
    # Check 5: Directories
    print("\n📂 DIRECTORIES")
    print("-" * 70)
    
    output_dir = version3_dir / "output"
    log_dir = version3_dir / "logs"
    
    check_item(f"output/ directory", output_dir.exists())
    check_item(f"logs/ directory", log_dir.exists())
    
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)
        print("  → Created output/ directory")
    
    if not log_dir.exists():
        log_dir.mkdir(exist_ok=True)
        print("  → Created logs/ directory")
    
    # Check 6: SLURM availability
    print("\n🖥️  SLURM SYSTEM")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            ['squeue', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        slurm_available = result.returncode == 0
        check_item(
            "SLURM available",
            slurm_available,
            "Not on SLURM system. Use 'python main_solver.py' directly (slower)."
        )
        
        if slurm_available:
            # Check partitions
            result = subprocess.run(
                ['sinfo', '-p', 'dc-gpu-devel', '-h'],
                capture_output=True,
                text=True,
                timeout=5
            )
            partition_available = result.returncode == 0
            check_item(
                "  dc-gpu-devel partition accessible",
                partition_available,
                "Partition not available. Edit run_solver.sh to use different partition."
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        check_item(
            "SLURM available",
            False,
            "Not on SLURM system or squeue not in PATH."
        )
    
    # Check 7: Shell script permissions
    print("\n🔒 PERMISSIONS")
    print("-" * 70)
    
    run_script = version3_dir / "run_solver.sh"
    monitor_script = version3_dir / "monitor.py"
    
    run_executable = os.access(run_script, os.X_OK)
    monitor_executable = os.access(monitor_script, os.X_OK)
    
    check_item("run_solver.sh executable", run_executable)
    check_item("monitor.py executable", monitor_executable)
    
    if not run_executable:
        os.chmod(run_script, 0o755)
        print("  → Made run_solver.sh executable")
    
    if not monitor_executable:
        os.chmod(monitor_script, 0o755)
        print("  → Made monitor.py executable")
    
    # Check 8: Previous runs
    print("\n📊 STATE")
    print("-" * 70)
    
    local_state_path = log_dir / "local_state.json"
    if local_state_path.exists():
        import json
        with open(local_state_path, 'r') as f:
            state = json.load(f)
        num_runs = state.get('num_runs', 0)
        num_images = len(state.get('images', {}))
        check_item(f"Previous runs found: {num_runs}", True)
        if num_images > 0:
            print(f"  → State for {num_images} images available")
            print(f"     Solver will use previous kappas if available")
    else:
        check_item("No previous state (fresh start)", True)
    
    # Summary
    print_header("Pre-flight Check Complete")
    
    if all_checks_passed:
        print("\n✅ ALL CHECKS PASSED")
        print("\nYou're ready to launch:")
        print("  sbatch run_solver.sh")
        print("\nMonitor progress with:")
        print("  python monitor.py")
        print("  tail -f logs/slurm_*.out")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above before running.")
        print("Common fixes:")
        print("  - Load PyTorch module: module load PyTorch")
        print("  - Check dataset path")
        print("  - Ensure you're on a GPU partition")
    
    print("\n" + "=" * 70)
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())

