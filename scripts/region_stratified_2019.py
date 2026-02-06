"""
Regional stratified King's EI analysis for 2019 elections.
Compares Metropolitana (Montevideo + Canelones) vs Interior.
Includes CA, PC, PI, PN, FA with PI separated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def analyze_stratum(stratum_name, df_stratum, samples=2000, chains=3):
    """Analyze a single regional stratum for 2019 elections."""

    print(f"\n{'='*80}")
    print(f"Analyzing: {stratum_name}")
    print(f"{'='*80}")
    print(f"Circuits: {len(df_stratum):,}")
    print(f"Total votes (primera): {df_stratum['total_primera'].sum():,.0f}")
    print(f"Total votes (ballotage): {df_stratum['total_ballotage'].sum():,.0f}")

    # Party votes
    print(f"\nParty votes (primera vuelta):")
    print(f"  CA: {df_stratum['ca_primera'].sum():,.0f} ({df_stratum['ca_primera'].sum()/df_stratum['total_primera'].sum():.2%})")
    print(f"  PC: {df_stratum['pc_primera'].sum():,.0f} ({df_stratum['pc_primera'].sum()/df_stratum['total_primera'].sum():.2%})")
    print(f"  PI: {df_stratum['pi_primera'].sum():,.0f} ({df_stratum['pi_primera'].sum()/df_stratum['total_primera'].sum():.2%})")
    print(f"  PN: {df_stratum['pn_primera'].sum():,.0f} ({df_stratum['pn_primera'].sum()/df_stratum['total_primera'].sum():.2%})")
    print(f"  FA: {df_stratum['fa_primera'].sum():,.0f} ({df_stratum['fa_primera'].sum()/df_stratum['total_primera'].sum():.2%})")
    print(f"  OTROS: {df_stratum['otros_primera'].sum():,.0f} ({df_stratum['otros_primera'].sum()/df_stratum['total_primera'].sum():.2%})")

    # Define columns with PI separated
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pi_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Fit model
    print(f"\nFitting King's EI model ({samples} samples, {chains} chains)...")
    start_time = datetime.now()

    model = KingEI(
        num_samples=samples,
        num_chains=chains,
        num_warmup=samples,
        random_seed=42
    )

    model.fit(
        data=df_stratum,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Model fitting completed in {elapsed/60:.1f} minutes")

    # Get results
    T = model.get_transition_matrix()
    ci_95 = model.get_credible_intervals(0.95)
    diag = model.get_diagnostics()

    print(f"\nDiagnostics:")
    print(f"  Max R-hat: {np.max(diag['rhat']):.4f}")
    print(f"  Min ESS: {np.min(diag['ess']):.0f}")

    # Party indices: ca=0, fa=1, otros=2, pc=3, pi=4, pn=5
    ca_idx, pc_idx, pi_idx, pn_idx = 0, 3, 4, 5

    # Print results
    print(f"\nTransfer rates (mean [95% CI]):")
    print(f"  CA -> FA: {T[ca_idx, 0]*100:.1f}% [{ci_95['lower'][ca_idx, 0]*100:.1f}% - {ci_95['upper'][ca_idx, 0]*100:.1f}%]")
    print(f"  CA -> PN: {T[ca_idx, 1]*100:.1f}% [{ci_95['lower'][ca_idx, 1]*100:.1f}% - {ci_95['upper'][ca_idx, 1]*100:.1f}%]")
    print(f"  PC -> FA: {T[pc_idx, 0]*100:.1f}% [{ci_95['lower'][pc_idx, 0]*100:.1f}% - {ci_95['upper'][pc_idx, 0]*100:.1f}%]")
    print(f"  PC -> PN: {T[pc_idx, 1]*100:.1f}% [{ci_95['lower'][pc_idx, 1]*100:.1f}% - {ci_95['upper'][pc_idx, 1]*100:.1f}%]")
    print(f"  PI -> FA: {T[pi_idx, 0]*100:.1f}% [{ci_95['lower'][pi_idx, 0]*100:.1f}% - {ci_95['upper'][pi_idx, 0]*100:.1f}%]")
    print(f"  PI -> PN: {T[pi_idx, 1]*100:.1f}% [{ci_95['lower'][pi_idx, 1]*100:.1f}% - {ci_95['upper'][pi_idx, 1]*100:.1f}%]")
    print(f"  PN -> FA: {T[pn_idx, 0]*100:.1f}% [{ci_95['lower'][pn_idx, 0]*100:.1f}% - {ci_95['upper'][pn_idx, 0]*100:.1f}%]")
    print(f"  PN -> PN: {T[pn_idx, 1]*100:.1f}% [{ci_95['lower'][pn_idx, 1]*100:.1f}% - {ci_95['upper'][pn_idx, 1]*100:.1f}%]")

    # Return results
    return {
        'name': stratum_name,
        'transition_matrix': T,
        'ci_95': ci_95,
        'diagnostics': diag,
        'n_circuits': len(df_stratum),
        'votes': {
            'ca': df_stratum['ca_primera'].sum(),
            'pc': df_stratum['pc_primera'].sum(),
            'pi': df_stratum['pi_primera'].sum(),
            'pn': df_stratum['pn_primera'].sum(),
            'fa': df_stratum['fa_primera'].sum(),
            'otros': df_stratum['otros_primera'].sum(),
            'total_primera': df_stratum['total_primera'].sum(),
            'total_ballotage': df_stratum['total_ballotage'].sum(),
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Regional stratified King\'s EI analysis for 2019 elections'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged_2019.parquet',
        help='Path to 2019 merged data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of MCMC samples (default: 2000)'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=3,
        help='Number of MCMC chains (default: 3)'
    )
    parser.add_argument(
        '--output-pkl',
        type=str,
        default='outputs/results/region_transfers_2019.pkl',
        help='Output pickle path'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='outputs/tables/region_2019.csv',
        help='Output CSV path'
    )

    args = parser.parse_args()

    print("="*80)
    print("REGIONAL STRATIFIED ANALYSIS - 2019 ELECTIONS")
    print("="*80)
    print(f"Data: {args.data}")
    print(f"Samples: {args.samples}")
    print(f"Chains: {args.chains}")
    print("="*80)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)

    print(f"Total circuits: {len(df):,}")
    print(f"Total votes (primera): {df['total_primera'].sum():,.0f}")

    # Define regions
    # Metropolitana = Montevideo (MO) + Canelones (CA)
    # Interior = Rest

    # Map abbreviations to full names
    dept_map = {
        'MO': 'Montevideo',
        'CA': 'Canelones'
    }

    df['departamento_full'] = df['departamento'].map(dept_map).fillna(df['departamento'])

    df['region'] = df['departamento'].apply(
        lambda x: 'Metropolitana' if x in ['MO', 'CA'] else 'Interior'
    )

    print(f"\nRegion distribution:")
    print(df['region'].value_counts())
    print(f"\nMetropolitana departments: {df[df['region'] == 'Metropolitana']['departamento'].unique()}")

    # Analyze each region
    results = {}

    for region in ['Metropolitana', 'Interior']:
        df_region = df[df['region'] == region]
        result = analyze_stratum(
            stratum_name=region,
            df_stratum=df_region,
            samples=args.samples,
            chains=args.chains
        )
        results[region] = result

    # Save pickle results
    output_pkl = Path(args.output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'results': results,
        'election_year': 2019,
        'samples': args.samples,
        'chains': args.chains,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_pkl, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"\n\nPickle results saved to: {output_pkl}")

    # Create CSV summary
    csv_rows = []

    for region_name, result in results.items():
        T = result['transition_matrix']
        ci = result['ci_95']

        # Party indices: ca=0, fa=1, otros=2, pc=3, pi=4, pn=5
        ca_idx, pc_idx, pi_idx, pn_idx = 0, 3, 4, 5

        row = {
            'year': 2019,
            'region': region_name,
            'n_circuits': result['n_circuits'],

            # CA
            'ca_to_fa': T[ca_idx, 0],
            'ca_to_pn': T[ca_idx, 1],
            'ca_to_blancos': T[ca_idx, 2],
            'ca_to_fa_lower': ci['lower'][ca_idx, 0],
            'ca_to_fa_upper': ci['upper'][ca_idx, 0],
            'ca_votes': result['votes']['ca'],

            # PC
            'pc_to_fa': T[pc_idx, 0],
            'pc_to_pn': T[pc_idx, 1],
            'pc_to_blancos': T[pc_idx, 2],
            'pc_to_fa_lower': ci['lower'][pc_idx, 0],
            'pc_to_fa_upper': ci['upper'][pc_idx, 0],
            'pc_votes': result['votes']['pc'],

            # PI
            'pi_to_fa': T[pi_idx, 0],
            'pi_to_pn': T[pi_idx, 1],
            'pi_to_blancos': T[pi_idx, 2],
            'pi_to_fa_lower': ci['lower'][pi_idx, 0],
            'pi_to_fa_upper': ci['upper'][pi_idx, 0],
            'pi_votes': result['votes']['pi'],

            # PN
            'pn_to_fa': T[pn_idx, 0],
            'pn_to_pn': T[pn_idx, 1],
            'pn_to_blancos': T[pn_idx, 2],
            'pn_to_fa_lower': ci['lower'][pn_idx, 0],
            'pn_to_fa_upper': ci['upper'][pn_idx, 0],
            'pn_votes': result['votes']['pn'],

            # Diagnostics
            'max_rhat': np.max(result['diagnostics']['rhat']),
            'min_ess': np.min(result['diagnostics']['ess']),
        }

        csv_rows.append(row)

    df_results = pd.DataFrame(csv_rows)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)

    print(f"CSV results saved to: {output_csv}")

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON: METROPOLITANA VS INTERIOR (2019)")
    print("="*80)

    metro = results['Metropolitana']
    interior = results['Interior']

    print(f"\nCA defection to FA:")
    print(f"  Metropolitana: {metro['transition_matrix'][0, 0]*100:.1f}%")
    print(f"  Interior:      {interior['transition_matrix'][0, 0]*100:.1f}%")
    print(f"  Difference:    {(metro['transition_matrix'][0, 0] - interior['transition_matrix'][0, 0])*100:+.1f} pp")

    print(f"\nPC defection to FA:")
    print(f"  Metropolitana: {metro['transition_matrix'][3, 0]*100:.1f}%")
    print(f"  Interior:      {interior['transition_matrix'][3, 0]*100:.1f}%")
    print(f"  Difference:    {(metro['transition_matrix'][3, 0] - interior['transition_matrix'][3, 0])*100:+.1f} pp")

    print(f"\nPI defection to FA:")
    print(f"  Metropolitana: {metro['transition_matrix'][4, 0]*100:.1f}%")
    print(f"  Interior:      {interior['transition_matrix'][4, 0]*100:.1f}%")
    print(f"  Difference:    {(metro['transition_matrix'][4, 0] - interior['transition_matrix'][4, 0])*100:+.1f} pp")

    print(f"\nPN defection to FA:")
    print(f"  Metropolitana: {metro['transition_matrix'][5, 0]*100:.1f}%")
    print(f"  Interior:      {interior['transition_matrix'][5, 0]*100:.1f}%")
    print(f"  Difference:    {(metro['transition_matrix'][5, 0] - interior['transition_matrix'][5, 0])*100:+.1f} pp")

    print("\n" + "="*80)
    print("VOTE IMPACT ANALYSIS")
    print("="*80)

    # Calculate vote transfers
    for region_name, result in results.items():
        T = result['transition_matrix']
        ca_idx, pc_idx, pi_idx = 0, 3, 4

        ca_votes = result['votes']['ca']
        pc_votes = result['votes']['pc']
        pi_votes = result['votes']['pi']

        ca_to_fa = ca_votes * T[ca_idx, 0]
        pc_to_fa = pc_votes * T[pc_idx, 0]
        pi_to_fa = pi_votes * T[pi_idx, 0]
        total_to_fa = ca_to_fa + pc_to_fa + pi_to_fa

        print(f"\n{region_name}:")
        print(f"  CA -> FA: {ca_to_fa:,.0f} votes ({ca_to_fa/result['votes']['total_primera']*100:.2%} of total)")
        print(f"  PC -> FA: {pc_to_fa:,.0f} votes ({pc_to_fa/result['votes']['total_primera']*100:.2%} of total)")
        print(f"  PI -> FA: {pi_to_fa:,.0f} votes ({pi_to_fa/result['votes']['total_primera']*100:.2%} of total)")
        print(f"  TOTAL coalition -> FA: {total_to_fa:,.0f} votes ({total_to_fa/result['votes']['total_primera']*100:.2%} of total)")


if __name__ == '__main__':
    main()
