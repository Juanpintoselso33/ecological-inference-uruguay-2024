"""
Monitor progress of 2019 analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from pathlib import Path
import pickle


def check_files():
    """Check which output files exist."""

    results_dir = Path('outputs/results')
    tables_dir = Path('outputs/tables')

    expected_files = {
        'results': [
            'national_transfers_2019.pkl',
            'department_transfers_2019.pkl',
            'urban_rural_transfers_2019.pkl',
            'region_transfers_2019.pkl',
        ],
        'tables': [
            'transfers_by_department_2019.csv',
            'urban_rural_2019.csv',
            'region_2019.csv',
        ]
    }

    print("="*70)
    print("2019 ANALYSIS PROGRESS CHECK")
    print("="*70)
    print()

    # Check results files
    print("RESULTS FILES (pickles):")
    print("-"*70)
    for fname in expected_files['results']:
        fpath = results_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            mtime = time.ctime(fpath.stat().st_mtime)
            print(f"  [OK] {fname:40s} ({size_kb:>8.1f} KB) - {mtime}")
        else:
            print(f"  [--] {fname:40s} (not found)")

    print()
    print("TABLE FILES (CSV):")
    print("-"*70)
    for fname in expected_files['tables']:
        fpath = tables_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            mtime = time.ctime(fpath.stat().st_mtime)

            # Count rows
            try:
                with open(fpath) as f:
                    nrows = sum(1 for _ in f) - 1  # subtract header
                print(f"  [OK] {fname:40s} ({size_kb:>6.1f} KB, {nrows:>3d} rows) - {mtime}")
            except:
                print(f"  [OK] {fname:40s} ({size_kb:>6.1f} KB) - {mtime}")
        else:
            print(f"  [--] {fname:40s} (not found)")

    print()

    # Try to load and summarize national results
    national_pkl = results_dir / 'national_transfers_2019.pkl'
    if national_pkl.exists():
        try:
            with open(national_pkl, 'rb') as f:
                result = pickle.load(f)

            print("="*70)
            print("NATIONAL RESULTS PREVIEW")
            print("="*70)

            T = result['transition_matrix']
            print(f"\nCircuits: {result['n_circuits']:,}")
            print(f"Max R-hat: {max(max(r) if isinstance(r, list) else r for r in result['diagnostics']['rhat']):.4f}")
            print(f"Min ESS: {min(min(e) if isinstance(e, list) else e for e in result['diagnostics']['ess']):.0f}")

            print("\nTransition rates to FA:")
            for i, party in enumerate(result['origin_cols']):
                party_name = party.replace('_primera', '').upper()
                rate = T[i][0]
                votes = result[f'{party}_votes']
                defected = rate * votes
                print(f"  {party_name:6s}: {rate*100:5.1f}% ({defected:>10,.0f} votes defected to FA)")

        except Exception as e:
            print(f"Error loading national results: {e}")


def check_log():
    """Check last lines of log."""

    log_path = Path('logs/analyze_2019_complete.log')

    if log_path.exists():
        print("\n" + "="*70)
        print("LAST 30 LINES OF LOG")
        print("="*70)

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"Error reading log: {e}")
    else:
        print("\nLog file not found: logs/analyze_2019_complete.log")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor 2019 analysis progress')
    parser.add_argument('--watch', action='store_true', help='Watch mode (refresh every 30s)')

    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                check_files()
                check_log()
                print("\n" + "="*70)
                print("Refreshing in 30 seconds... (Ctrl+C to stop)")
                print("="*70)
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        check_files()
        check_log()
