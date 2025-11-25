"""
Submission Utility for Phase 1

Submits adversarial examples to the API and optionally queries for logits.
Adapted from task_template.py with enhanced error handling and logging.
"""

import requests
import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path


# API Configuration
BASE_URL = "http://34.122.51.94:80"
API_KEY = "f62b1499d4e2bf13ae56be5683c974c1"
TASK_ID = "10-adversarial-examples"


def die(msg):
    """Print error and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def get_logits(query_path: str, log_path: str = None) -> dict:
    """
    Query API for logits.
    
    Args:
        query_path: Path to .npz file
        log_path: Optional path to save response JSON
    
    Returns:
        API response as dict
    """
    if not os.path.isfile(query_path):
        die(f"File not found: {query_path}")
    
    print(f"Querying logits for: {query_path}")
    print("⏳ Requesting logits from API...")
    
    try:
        with open(query_path, "rb") as f:
            files = {"npz": (query_path, f, "application/octet-stream")}
            response = requests.post(
                f"{BASE_URL}/{TASK_ID}/logits",
                files=files,
                headers={"X-API-Key": API_KEY},
                timeout=60
            )
        
        response.raise_for_status()
        data = response.json()
        
        print("✓ Request successful")
        print(f"  Received logits for {len(data.get('results', []))} images")
        
        # Save response if log path provided
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved response to: {log_path}")
        
        return data
        
    except requests.exceptions.Timeout:
        die("Request timed out. API may be overloaded.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {response.status_code}")
        print(f"Response: {response.text}")
        die(str(e))
    except Exception as e:
        die(f"Request failed: {e}")


def submit_file(file_path: str, log_path: str = None) -> dict:
    """
    Submit adversarial examples to leaderboard.
    
    Args:
        file_path: Path to submission .npz file
        log_path: Optional path to save submission result
    
    Returns:
        Submission response as dict
    """
    if not os.path.isfile(file_path):
        die(f"File not found: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    max_size = 200 * 1024 * 1024  # 200 MB
    if file_size > max_size:
        die(f"File too large: {file_size / 1e6:.1f} MB (max: 200 MB)")
    
    print(f"Submitting: {file_path}")
    print(f"  File size: {file_size / 1e6:.2f} MB")
    print("⏳ Uploading to server...")
    
    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )
        
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}
        
        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413)")
        
        resp.raise_for_status()
        
        submission_id = body.get("submission_id")
        score = body.get("score")
        
        print("✓ Successfully submitted!")
        print(f"  Server response: {body}")
        if submission_id:
            print(f"  Submission ID: {submission_id}")
        if score is not None:
            print(f"  Score: {score:.6f}")
        
        # Save submission result
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            result = {
                'timestamp': datetime.now().isoformat(),
                'file': file_path,
                'file_size_mb': file_size / 1e6,
                'response': body
            }
            with open(log_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Saved submission record to: {log_path}")
        
        return body
        
    except requests.exceptions.Timeout:
        die("Upload timed out. Try again.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {resp.status_code}")
        try:
            print(f"Server response: {resp.json()}")
        except:
            print(f"Server response: {resp.text}")
        die(str(e))
    except Exception as e:
        die(f"Submission failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Submit adversarial examples or query logits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('file', type=str,
                       help='Path to .npz submission file')
    parser.add_argument('--action', type=str, choices=['submit', 'logits', 'both'],
                       default='submit',
                       help='Action to perform')
    parser.add_argument('--log-dir', type=str, default='./logs/api',
                       help='Directory to save API responses')
    parser.add_argument('--no-save-logs', action='store_true',
                       help='Do not save API responses to files')
    
    args = parser.parse_args()
    
    # Generate log paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(args.file).stem
    
    if args.no_save_logs:
        logits_log = None
        submit_log = None
    else:
        logits_log = f"{args.log_dir}/logits_{base_name}_{timestamp}.json"
        submit_log = f"{args.log_dir}/submit_{base_name}_{timestamp}.json"
    
    print("=" * 70)
    print("Adversarial Examples Submission Tool")
    print("=" * 70)
    
    # Execute action
    if args.action in ['logits', 'both']:
        print("\n--- Querying Logits ---")
        get_logits(args.file, logits_log)
        
        if args.action == 'both':
            print("\n⏳ Waiting 5 seconds before submission...")
            import time
            time.sleep(5)
    
    if args.action in ['submit', 'both']:
        print("\n--- Submitting to Leaderboard ---")
        submit_file(args.file, submit_log)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()

