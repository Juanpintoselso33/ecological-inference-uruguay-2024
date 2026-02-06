"""
National-level King's EI analysis WITH PI separate - HIGH QUALITY (4000 samples, 4 chains).
Publication-quality analysis.
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='National-level King\'s EI analysis with PI (publication quality)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged.parquet',
        help='Path to merged data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=4000,
        help='Number of MCMC samples (default: 4000 for publication quality)'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Number of MCMC chains (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results/national_transfers_with_pi.pkl',
        help='Output pickle path'
    )

    args = parser.parse_args()

    print("="*80)
    print("NATIONAL-LEVEL KING'S EI ANALYSIS - WITH PI SEPARATE")
    print("="*80)
    print(f"Samples: {args.samples}")
    print(f"Chains: {args.chains}")
    print(f"Warmup: {args.samples} (same as samples)")
    print("="*80 + "\n")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    print(f"Total circuits: {len(df):,}")
    print(f"Total votes (primera vuelta): {df['total_primera'].sum():,.0f}")
    print(f"Total votes (ballotage): {df['total_ballotage'].sum():,.0f}")
    print(f"\nParty votes (primera vuelta):")
    print(f"  FA: {df['fa_primera'].sum():,.0f} ({df['fa_primera'].sum()/df['total_primera'].sum():.2%})")
    print(f"  PN: {df['pn_primera'].sum():,.0f} ({df['pn_primera'].sum()/df['total_primera'].sum():.2%})")
    print(f"  PC: {df['pc_primera'].sum():,.0f} ({df['pc_primera'].sum()/df['total_primera'].sum():.2%})")
    print(f"  CA: {df['ca_primera'].sum():,.0f} ({df['ca_primera'].sum()/df['total_primera'].sum():.2%})")
    print(f"  PI: {df['pi_primera'].sum():,.0f} ({df['pi_primera'].sum()/df['total_primera'].sum():.2%})")
    print(f"  OTROS: {df['otros_primera'].sum():,.0f} ({df['otros_primera'].sum()/df['total_primera'].sum():.2%})")

    # Define columns - NOW WITH PI
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pi_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    print(f"\nOrigin parties: {origin_cols}")
    print(f"Destination categories: {destination_cols}\n")

    # Fit model
    print(f"Fitting King's EI model (this will take ~6-8 hours)...")
    start_time = datetime.now()

    model = KingEI(
        num_samples=args.samples,
        num_chains=args.chains,
        num_warmup=args.samples,
        random_seed=42
    )

    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\nModel fitting completed in {elapsed/3600:.2f} hours")

    # Get results
    T = model.get_transition_matrix()
    ci_95 = model.get_credible_intervals(0.95)
    ci_90 = model.get_credible_intervals(0.90)

    # Diagnostics
    diag = model.get_diagnostics()

    print("\n" + "="*80)
    print("DIAGNOSTICS")
    print("="*80)
    print(f"Max R-hat: {np.max(diag['rhat']):.4f} (should be < 1.01)")
    print(f"Min ESS: {np.min(diag['ess']):.0f} (should be > 1000)")
    print(f"Mean ESS: {np.mean(diag['ess']):.0f}")

    # Results
    print("\n" + "="*80)
    print("TRANSITION MATRIX (mean estimates)")
    print("="*80)
    print("Rows: Origin parties (CA, FA, OTROS, PC, PI, PN)")
    print("Cols: Destinations (FA, PN, Blancos)\n")
    print(T)

    # Detailed results by party
    party_names = ['CA', 'FA', 'OTROS', 'PC', 'PI', 'PN']
    dest_names = ['FA', 'PN', 'Blancos']

    print("\n" + "="*80)
    print("DETAILED TRANSFER RATES (with 95% CI)")
    print("="*80)

    for i, party in enumerate(party_names):
        print(f"\n{party} voters:")
        for j, dest in enumerate(dest_names):
            mean = T[i, j] * 100
            lower = ci_95['lower'][i, j] * 100
            upper = ci_95['upper'][i, j] * 100
            print(f"  -> {dest:10s}: {mean:5.1f}% [{lower:5.1f}% - {upper:5.1f}%]")

    # Calculate absolute vote transfers
    print("\n" + "="*80)
    print("ABSOLUTE VOTE TRANSFERS (estimated)")
    print("="*80)

    party_votes = {
        'CA': df['ca_primera'].sum(),
        'PC': df['pc_primera'].sum(),
        'PI': df['pi_primera'].sum(),
        'PN': df['pn_primera'].sum(),
    }

    # CA transfers
    ca_idx = 0
    print("\nCA transfers:")
    print(f"  Total CA votes: {party_votes['CA']:,.0f}")
    print(f"  -> FA: {T[ca_idx, 0] * party_votes['CA']:,.0f} votes ({T[ca_idx, 0]*100:.1f}%)")
    print(f"  -> PN: {T[ca_idx, 1] * party_votes['CA']:,.0f} votes ({T[ca_idx, 1]*100:.1f}%)")
    print(f"  -> Blancos: {T[ca_idx, 2] * party_votes['CA']:,.0f} votes ({T[ca_idx, 2]*100:.1f}%)")

    # PC transfers
    pc_idx = 3
    print("\nPC transfers:")
    print(f"  Total PC votes: {party_votes['PC']:,.0f}")
    print(f"  -> FA: {T[pc_idx, 0] * party_votes['PC']:,.0f} votes ({T[pc_idx, 0]*100:.1f}%)")
    print(f"  -> PN: {T[pc_idx, 1] * party_votes['PC']:,.0f} votes ({T[pc_idx, 1]*100:.1f}%)")
    print(f"  -> Blancos: {T[pc_idx, 2] * party_votes['PC']:,.0f} votes ({T[pc_idx, 2]*100:.1f}%)")

    # PI transfers (NEW)
    pi_idx = 4
    print("\nPI transfers:")
    print(f"  Total PI votes: {party_votes['PI']:,.0f}")
    print(f"  -> FA: {T[pi_idx, 0] * party_votes['PI']:,.0f} votes ({T[pi_idx, 0]*100:.1f}%)")
    print(f"  -> PN: {T[pi_idx, 1] * party_votes['PI']:,.0f} votes ({T[pi_idx, 1]*100:.1f}%)")
    print(f"  -> Blancos: {T[pi_idx, 2] * party_votes['PI']:,.0f} votes ({T[pi_idx, 2]*100:.1f}%)")

    # PN transfers
    pn_idx = 5
    print("\nPN transfers:")
    print(f"  Total PN votes: {party_votes['PN']:,.0f}")
    print(f"  -> FA: {T[pn_idx, 0] * party_votes['PN']:,.0f} votes ({T[pn_idx, 0]*100:.1f}%)")
    print(f"  -> PN: {T[pn_idx, 1] * party_votes['PN']:,.0f} votes ({T[pn_idx, 1]*100:.1f}%)")
    print(f"  -> Blancos: {T[pn_idx, 2] * party_votes['PN']:,.0f} votes ({T[pn_idx, 2]*100:.1f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'posterior_mean': T,
        'credible_interval_lower': ci_95['lower'],
        'credible_interval_upper': ci_95['upper'],
        'ci_90_lower': ci_90['lower'],
        'ci_90_upper': ci_90['upper'],
        'diagnostics': diag,
        'party_names': party_names,
        'dest_names': dest_names,
        'party_votes': party_votes,
        'n_circuits': len(df),
        'samples': args.samples,
        'chains': args.chains,
        'elapsed_hours': elapsed / 3600,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n\nResults saved to: {output_path}")
    print(f"Total execution time: {elapsed/3600:.2f} hours")


if __name__ == '__main__':
    main()
