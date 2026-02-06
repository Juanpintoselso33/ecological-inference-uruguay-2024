"""
Analyze coalition transfers by department.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def analyze_department(dept_name, df_dept, quick=False):
    """Analyze a single department."""

    if len(df_dept) < 20:
        logger.warning(f"Skipping {dept_name}: too few circuits ({len(df_dept)})")
        return None

    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Fit model
    samples = 500 if quick else 1000
    model = KingEI(num_samples=samples, num_chains=2, num_warmup=samples, random_seed=42)

    try:
        model.fit(
            data=df_dept,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=False
        )
    except Exception as e:
        logger.error(f"Failed to fit {dept_name}: {e}")
        return None

    # Extract results
    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(0.95)

    # Coalition indices
    ca_idx, pc_idx, pn_idx = 0, 3, 4

    results = {
        'departamento': dept_name,
        'n_circuits': len(df_dept),

        # CA transfers
        'ca_to_fa': T[ca_idx, 0],
        'ca_to_pn': T[ca_idx, 1],
        'ca_to_fa_lower': ci['lower'][ca_idx, 0],
        'ca_to_fa_upper': ci['upper'][ca_idx, 0],

        # PC transfers
        'pc_to_fa': T[pc_idx, 0],
        'pc_to_pn': T[pc_idx, 1],
        'pc_to_fa_lower': ci['lower'][pc_idx, 0],
        'pc_to_fa_upper': ci['upper'][pc_idx, 0],

        # PN transfers
        'pn_to_fa': T[pn_idx, 0],
        'pn_to_pn': T[pn_idx, 1],
        'pn_to_fa_lower': ci['lower'][pn_idx, 0],
        'pn_to_fa_upper': ci['upper'][pn_idx, 0],

        # Votes
        'ca_votes': df_dept['ca_primera'].sum(),
        'pc_votes': df_dept['pc_primera'].sum(),
        'pn_votes': df_dept['pn_primera'].sum(),
    }

    # Diagnostics
    diag = model.get_diagnostics()
    results['max_rhat'] = np.max(diag['rhat'])
    results['min_ess'] = np.min(diag['ess'])

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze coalition transfers by department'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged.parquet',
        help='Path to merged data'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick analysis (500 samples)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/tables/transfers_by_department.csv',
        help='Output CSV path'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    print(f"Total circuits: {len(df)}")
    print(f"Departments: {df['departamento'].nunique()}\n")

    # Analyze each department
    results_list = []

    departments = sorted(df['departamento'].unique())

    for i, dept in enumerate(departments, 1):
        print(f"[{i}/{len(departments)}] Analyzing {dept}...")

        df_dept = df[df['departamento'] == dept]
        result = analyze_department(dept, df_dept, quick=args.quick)

        if result:
            results_list.append(result)
            print(f"  CA to FA: {result['ca_to_fa']*100:.1f}%, "
                  f"PC to FA: {result['pc_to_fa']*100:.1f}%, "
                  f"PN to FA: {result['pn_to_fa']*100:.1f}%")

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY BY DEPARTMENT")
    print("="*70)

    # Sort by CA defection rate
    results_df_sorted = results_df.sort_values('ca_to_fa', ascending=False)

    print("\nCA Defection to FA (highest to lowest):")
    print("-"*70)
    for _, row in results_df_sorted.iterrows():
        print(f"{row['departamento']:20s}: {row['ca_to_fa']*100:5.1f}% "
              f"[{row['ca_to_fa_lower']*100:5.1f}% - {row['ca_to_fa_upper']*100:5.1f}%] "
              f"({row['ca_votes']:,.0f} votes)")

    print("\n" + "="*70)
    print("PC Defection to FA (highest to lowest):")
    print("-"*70)
    results_df_sorted = results_df.sort_values('pc_to_fa', ascending=False)
    for _, row in results_df_sorted.head(10).iterrows():
        print(f"{row['departamento']:20s}: {row['pc_to_fa']*100:5.1f}% "
              f"({row['pc_votes']:,.0f} votes)")

    print("\n" + "="*70)
    print("PN Defection to FA (highest to lowest):")
    print("-"*70)
    results_df_sorted = results_df.sort_values('pn_to_fa', ascending=False)
    for _, row in results_df_sorted.head(10).iterrows():
        print(f"{row['departamento']:20s}: {row['pn_to_fa']*100:5.1f}% "
              f"({row['pn_votes']:,.0f} votes)")


if __name__ == '__main__':
    main()
