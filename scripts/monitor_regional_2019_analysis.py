"""
Monitor progress of regional 2019 analysis and prepare comparison.
"""

import sys
from pathlib import Path
import time
import os

def check_analysis_complete():
    """Check if regional 2019 analysis is complete."""
    pkl_path = Path('outputs/results/region_transfers_2019.pkl')
    csv_path = Path('outputs/tables/region_2019.csv')

    return pkl_path.exists() and csv_path.exists()

def monitor_log():
    """Monitor the log file for progress."""
    log_path = Path('outputs/logs/region_2019_analysis.log')

    if not log_path.exists():
        return None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Look for progress indicators
    for line in reversed(lines[-50:]):
        if 'completed' in line.lower() or 'Model fitting' in line:
            return line.strip()

    return None

def main():
    print("="*80)
    print("MONITORING REGIONAL 2019 ANALYSIS")
    print("="*80)

    max_wait = 3600  # 60 minutes
    wait_interval = 30  # check every 30 seconds

    elapsed = 0
    while elapsed < max_wait:
        if check_analysis_complete():
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)

            # Display summary
            import pickle
            with open('outputs/results/region_transfers_2019.pkl', 'rb') as f:
                data = pickle.load(f)

            results = data['results']

            print(f"\nMetropolitana:")
            print(f"  Circuits: {results['Metropolitana']['n_circuits']:,}")
            print(f"  CA -> FA: {results['Metropolitana']['transition_matrix'][0, 0]*100:.1f}%")

            print(f"\nInterior:")
            print(f"  Circuits: {results['Interior']['n_circuits']:,}")
            print(f"  CA -> FA: {results['Interior']['transition_matrix'][0, 0]*100:.1f}%")

            print("\nProceeding with comparison analysis...")
            break

        status = monitor_log()
        if status:
            print(f"Status: {status}")

        print(f"Waiting... ({elapsed}s / {max_wait}s)")
        time.sleep(wait_interval)
        elapsed += wait_interval

    if elapsed >= max_wait:
        print(f"\nTimeout after {max_wait} seconds. Analysis may still be running.")
        return

    # Run comparison
    print("\nExecuting comparison analysis...")
    os.system('python scripts/compare_regional_2019_2024.py')


if __name__ == '__main__':
    main()
