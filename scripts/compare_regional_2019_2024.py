"""
Comparative analysis: Regional King's EI results 2019 vs 2024.
Identifies key differences in vote transfers between metropolitan and interior regions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def load_regional_results(pkl_path, year):
    """Load regional analysis results from pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def compare_regions(results_2019, results_2024):
    """Compare regional results between 2019 and 2024."""

    print("\n" + "="*100)
    print(f"COMPARATIVE ANALYSIS: REGIONAL VOTE TRANSFERS 2019 vs 2024")
    print("="*100)

    comparison_data = []

    # Compare each region
    for region in ['Metropolitana', 'Interior']:
        r19 = results_2019['results'][region]
        r24 = results_2024['results'][region]

        T19 = r19['transition_matrix']
        T24 = r24['transition_matrix']

        # Party indices: ca=0, fa=1, otros=2, pc=3, pi=4, pn=5
        ca_idx, pc_idx, pi_idx, pn_idx = 0, 3, 4, 5

        print(f"\n{'='*100}")
        print(f"REGION: {region}")
        print(f"{'='*100}")

        print(f"\nCircuits: 2019: {r19['n_circuits']:,} | 2024: {r24['n_circuits']:,}")
        print(f"Votes (primera): 2019: {r19['votes']['total_primera']:,.0f} | 2024: {r24['votes']['total_primera']:,.0f}")

        # CA analysis
        ca_to_fa_19 = T19[ca_idx, 0] * 100
        ca_to_fa_24 = T24[ca_idx, 0] * 100
        ca_to_fa_diff = ca_to_fa_24 - ca_to_fa_19

        print(f"\n{'-'*100}")
        print(f"CABILDO ABIERTO (CA) defection to FA:")
        print(f"  2019: {ca_to_fa_19:6.1f}%")
        print(f"  2024: {ca_to_fa_24:6.1f}%")
        print(f"  Change: {ca_to_fa_diff:+6.1f} pp")
        print(f"  CA votes: 2019: {r19['votes']['ca']:,.0f} | 2024: {r24['votes']['ca']:,.0f}")

        # PC analysis
        pc_to_fa_19 = T19[pc_idx, 0] * 100
        pc_to_fa_24 = T24[pc_idx, 0] * 100
        pc_to_fa_diff = pc_to_fa_24 - pc_to_fa_19

        print(f"\n{'-'*100}")
        print(f"PARTIDO COLORADO (PC) defection to FA:")
        print(f"  2019: {pc_to_fa_19:6.1f}%")
        print(f"  2024: {pc_to_fa_24:6.1f}%")
        print(f"  Change: {pc_to_fa_diff:+6.1f} pp")
        print(f"  PC votes: 2019: {r19['votes']['pc']:,.0f} | 2024: {r24['votes']['pc']:,.0f}")

        # PI analysis
        pi_to_fa_19 = T19[pi_idx, 0] * 100
        pi_to_fa_24 = T24[pi_idx, 0] * 100
        pi_to_fa_diff = pi_to_fa_24 - pi_to_fa_19

        print(f"\n{'-'*100}")
        print(f"PATRIOTA INDEPENDIENTE (PI) defection to FA:")
        print(f"  2019: {pi_to_fa_19:6.1f}%")
        print(f"  2024: {pi_to_fa_24:6.1f}%")
        print(f"  Change: {pi_to_fa_diff:+6.1f} pp")
        print(f"  PI votes: 2019: {r19['votes']['pi']:,.0f} | 2024: {r24['votes']['pi']:,.0f}")

        # PN analysis
        pn_to_fa_19 = T19[pn_idx, 0] * 100
        pn_to_fa_24 = T24[pn_idx, 0] * 100
        pn_to_fa_diff = pn_to_fa_24 - pn_to_fa_19

        print(f"\n{'-'*100}")
        print(f"PARTIDO NACIONAL (PN) defection to FA:")
        print(f"  2019: {pn_to_fa_19:6.1f}%")
        print(f"  2024: {pn_to_fa_24:6.1f}%")
        print(f"  Change: {pn_to_fa_diff:+6.1f} pp")
        print(f"  PN votes: 2019: {r19['votes']['pn']:,.0f} | 2024: {r24['votes']['pn']:,.0f}")

        # Store comparison data
        comparison_data.append({
            'year': 2019,
            'region': region,
            'party': 'CA',
            'to_fa_pct': ca_to_fa_19,
            'votes': r19['votes']['ca'],
            'circuits': r19['n_circuits']
        })
        comparison_data.append({
            'year': 2024,
            'region': region,
            'party': 'CA',
            'to_fa_pct': ca_to_fa_24,
            'votes': r24['votes']['ca'],
            'circuits': r24['n_circuits']
        })

        comparison_data.append({
            'year': 2019,
            'region': region,
            'party': 'PC',
            'to_fa_pct': pc_to_fa_19,
            'votes': r19['votes']['pc'],
            'circuits': r19['n_circuits']
        })
        comparison_data.append({
            'year': 2024,
            'region': region,
            'party': 'PC',
            'to_fa_pct': pc_to_fa_24,
            'votes': r24['votes']['pc'],
            'circuits': r24['n_circuits']
        })

        comparison_data.append({
            'year': 2019,
            'region': region,
            'party': 'PI',
            'to_fa_pct': pi_to_fa_19,
            'votes': r19['votes']['pi'],
            'circuits': r19['n_circuits']
        })
        comparison_data.append({
            'year': 2024,
            'region': region,
            'party': 'PI',
            'to_fa_pct': pi_to_fa_24,
            'votes': r24['votes']['pi'],
            'circuits': r24['n_circuits']
        })

        comparison_data.append({
            'year': 2019,
            'region': region,
            'party': 'PN',
            'to_fa_pct': pn_to_fa_19,
            'votes': r19['votes']['pn'],
            'circuits': r19['n_circuits']
        })
        comparison_data.append({
            'year': 2024,
            'region': region,
            'party': 'PN',
            'to_fa_pct': pn_to_fa_24,
            'votes': r24['votes']['pn'],
            'circuits': r24['n_circuits']
        })

    # Key findings
    print(f"\n\n{'='*100}")
    print("KEY REGIONAL DIFFERENCES: 2024 vs 2019")
    print("="*100)

    metro_19 = results_2019['results']['Metropolitana']
    metro_24 = results_2024['results']['Metropolitana']
    interior_19 = results_2019['results']['Interior']
    interior_24 = results_2024['results']['Interior']

    T19_m = metro_19['transition_matrix']
    T24_m = metro_24['transition_matrix']
    T19_i = interior_19['transition_matrix']
    T24_i = interior_24['transition_matrix']

    ca_idx, pc_idx, pi_idx, pn_idx = 0, 3, 4, 5

    print("\n[METROPOLITANA]")
    print(f"  CA -> FA: 2019={T19_m[ca_idx, 0]*100:.1f}% → 2024={T24_m[ca_idx, 0]*100:.1f}% (Δ={((T24_m[ca_idx, 0]-T19_m[ca_idx, 0])*100):+.1f} pp)")
    print(f"  PC -> FA: 2019={T19_m[pc_idx, 0]*100:.1f}% → 2024={T24_m[pc_idx, 0]*100:.1f}% (Δ={((T24_m[pc_idx, 0]-T19_m[pc_idx, 0])*100):+.1f} pp)")
    print(f"  PI -> FA: 2019={T19_m[pi_idx, 0]*100:.1f}% → 2024={T24_m[pi_idx, 0]*100:.1f}% (Δ={((T24_m[pi_idx, 0]-T19_m[pi_idx, 0])*100):+.1f} pp)")
    print(f"  PN -> FA: 2019={T19_m[pn_idx, 0]*100:.1f}% → 2024={T24_m[pn_idx, 0]*100:.1f}% (Δ={((T24_m[pn_idx, 0]-T19_m[pn_idx, 0])*100):+.1f} pp)")

    print("\n[INTERIOR]")
    print(f"  CA -> FA: 2019={T19_i[ca_idx, 0]*100:.1f}% → 2024={T24_i[ca_idx, 0]*100:.1f}% (Δ={((T24_i[ca_idx, 0]-T19_i[ca_idx, 0])*100):+.1f} pp)")
    print(f"  PC -> FA: 2019={T19_i[pc_idx, 0]*100:.1f}% → 2024={T24_i[pc_idx, 0]*100:.1f}% (Δ={((T24_i[pc_idx, 0]-T19_i[pc_idx, 0])*100):+.1f} pp)")
    print(f"  PI -> FA: 2019={T19_i[pi_idx, 0]*100:.1f}% → 2024={T24_i[pi_idx, 0]*100:.1f}% (Δ={((T24_i[pi_idx, 0]-T19_i[pi_idx, 0])*100):+.1f} pp)")
    print(f"  PN -> FA: 2019={T19_i[pn_idx, 0]*100:.1f}% → 2024={T24_i[pn_idx, 0]*100:.1f}% (Δ={((T24_i[pn_idx, 0]-T19_i[pn_idx, 0])*100):+.1f} pp)")

    # Metropolitan-Interior gaps
    print(f"\n[METROPOLITANA-INTERIOR GAP]")
    print(f"  CA -> FA (2019): {(T19_m[ca_idx, 0] - T19_i[ca_idx, 0])*100:+.1f} pp gap")
    print(f"  CA -> FA (2024): {(T24_m[ca_idx, 0] - T24_i[ca_idx, 0])*100:+.1f} pp gap")
    print(f"  Gap change: {((T24_m[ca_idx, 0] - T24_i[ca_idx, 0]) - (T19_m[ca_idx, 0] - T19_i[ca_idx, 0]))*100:+.1f} pp")

    print(f"\n  PC -> FA (2019): {(T19_m[pc_idx, 0] - T19_i[pc_idx, 0])*100:+.1f} pp gap")
    print(f"  PC -> FA (2024): {(T24_m[pc_idx, 0] - T24_i[pc_idx, 0])*100:+.1f} pp gap")
    print(f"  Gap change: {((T24_m[pc_idx, 0] - T24_i[pc_idx, 0]) - (T19_m[pc_idx, 0] - T19_i[pc_idx, 0]))*100:+.1f} pp")

    # Save comparison CSV
    df_comparison = pd.DataFrame(comparison_data)
    comparison_csv = Path('outputs/tables/regional_comparison_2019_2024.csv')
    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    df_comparison.to_csv(comparison_csv, index=False)
    print(f"\n\nComparison table saved to: {comparison_csv}")

    return df_comparison


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare regional King\'s EI results between 2019 and 2024 elections'
    )
    parser.add_argument(
        '--data-2019',
        type=str,
        default='outputs/results/region_transfers_2019.pkl',
        help='Path to 2019 regional results pickle'
    )
    parser.add_argument(
        '--data-2024',
        type=str,
        default='outputs/results/region_transfers_with_pi.pkl',
        help='Path to 2024 regional results pickle'
    )

    args = parser.parse_args()

    print("="*100)
    print("LOADING REGIONAL ANALYSIS RESULTS")
    print("="*100)

    # Load results
    print(f"\nLoading 2019 regional results from {args.data_2019}...")
    results_2019 = load_regional_results(args.data_2019, 2019)

    print(f"Loading 2024 regional results from {args.data_2024}...")
    results_2024 = load_regional_results(args.data_2024, 2024)

    # Compare
    comparison_df = compare_regions(results_2019, results_2024)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
