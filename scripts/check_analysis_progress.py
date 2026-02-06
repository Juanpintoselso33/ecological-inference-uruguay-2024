"""
Check progress of running analyses.
"""

import os
from pathlib import Path
from datetime import datetime

def check_file(path, description):
    """Check if file exists and show info."""
    path = Path(path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60
        print(f"[OK] {description}")
        print(f"  Path: {path}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.1f} min ago)")
        return True
    else:
        print(f"[NOT FOUND] {description}")
        print(f"  Path: {path}")
        return False


def check_log(path, description, tail_lines=20):
    """Check log file and show last few lines."""
    path = Path(path)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_minutes = (datetime.now() - mtime).total_seconds() / 60

        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        print(f"Size: {size_mb:.2f} MB | Last modified: {age_minutes:.1f} min ago")

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if len(lines) > tail_lines:
                print(f"\nLast {tail_lines} lines:")
                for line in lines[-tail_lines:]:
                    print(line.rstrip())
            else:
                print(f"\nAll {len(lines)} lines:")
                for line in lines:
                    print(line.rstrip())
        return True
    else:
        print(f"\n{description}: LOG NOT FOUND ({path})")
        return False


def main():
    print("="*80)
    print("ANALYSIS PROGRESS CHECK")
    print("="*80)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check output files
    print("OUTPUT FILES:")
    print("-"*80)
    check_file("outputs/tables/transfers_by_department_with_pi.csv",
               "Departmental analysis CSV")
    print()
    check_file("outputs/results/national_transfers_with_pi.pkl",
               "National analysis PKL")
    print()
    check_file("outputs/results/region_transfers_with_pi.pkl",
               "Stratified analysis PKL")
    print()
    check_file("outputs/tables/region_with_pi.csv",
               "Stratified analysis CSV")

    # Check logs
    print("\n\n")
    check_log("outputs/logs/dept_analysis_with_pi.log",
              "DEPARTMENTAL ANALYSIS LOG", tail_lines=30)

    check_log("outputs/logs/stratified_analysis_with_pi.log",
              "STRATIFIED ANALYSIS LOG", tail_lines=30)

    check_log("outputs/logs/national_analysis_with_pi.log",
              "NATIONAL ANALYSIS LOG", tail_lines=30)


if __name__ == '__main__':
    main()
