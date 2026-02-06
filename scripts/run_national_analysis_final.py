"""
Execute scientifically rigorous national King's EI analysis.
Final version with optimal MCMC parameters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    print("="*70)
    print("NATIONAL ECOLOGICAL INFERENCE ANALYSIS - FINAL")
    print("="*70)
    print()
    print("Parameters:")
    print("  - Samples: 4000 (post-warmup)")
    print("  - Chains: 4")
    print("  - Warmup: 2000")
    print("  - Target accept: 0.95")
    print()

    # Load data
    print("Loading data...")
    df = pd.read_parquet('data/processed/circuitos_merged.parquet')
    print(f"Data loaded: {len(df)} circuits")
    print()

    # Define columns
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Create model
    print("Creating King's EI model...")
    model = KingEI(
        num_samples=4000,
        num_chains=4,
        num_warmup=2000,
        target_accept=0.95,
        random_seed=42
    )

    # Fit
    print("Fitting model (this will take ~10-15 minutes)...")
    print()
    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Get results
    T = model.get_transition_matrix()
    ci_95 = model.get_credible_intervals(0.95)
    ci_99 = model.get_credible_intervals(0.99)

    # Party names
    origin_parties = ['CA', 'FA', 'OTROS', 'PC', 'PN']
    destination_parties = ['FA', 'PN', 'Blancos/Nulos']

    # Print transition matrix
    print("Transition Matrix (Posterior Mean):")
    print("-"*70)
    print(f'{"":8s}', end='')
    for dest in destination_parties:
        print(f'{dest:>15s}', end='')
    print()
    print("-"*70)

    for i, orig in enumerate(origin_parties):
        print(f'{orig:8s}', end='')
        for j in range(len(destination_parties)):
            print(f'{T[i,j]*100:14.1f}%', end='')
        print()

    # Key findings
    print()
    print("="*70)
    print("KEY COALITION FINDINGS")
    print("="*70)
    print()

    ca_idx, pc_idx, pn_idx = 0, 3, 4
    fa_dest, pn_dest = 0, 1

    print("CABILDO ABIERTO (CA):")
    print(f"  CA -> FA: {T[ca_idx, fa_dest]*100:5.1f}% [{ci_95['lower'][ca_idx, fa_dest]*100:5.1f}% - {ci_95['upper'][ca_idx, fa_dest]*100:5.1f}%]")
    print(f"  CA -> PN: {T[ca_idx, pn_dest]*100:5.1f}% [{ci_95['lower'][ca_idx, pn_dest]*100:5.1f}% - {ci_95['upper'][ca_idx, pn_dest]*100:5.1f}%]")

    print()
    print("PARTIDO COLORADO (PC):")
    print(f"  PC -> FA: {T[pc_idx, fa_dest]*100:5.1f}% [{ci_95['lower'][pc_idx, fa_dest]*100:5.1f}% - {ci_95['upper'][pc_idx, fa_dest]*100:5.1f}%]")
    print(f"  PC -> PN: {T[pc_idx, pn_dest]*100:5.1f}% [{ci_95['lower'][pc_idx, pn_dest]*100:5.1f}% - {ci_95['upper'][pc_idx, pn_dest]*100:5.1f}%]")

    print()
    print("PARTIDO NACIONAL (PN):")
    print(f"  PN -> FA: {T[pn_idx, fa_dest]*100:5.1f}% [{ci_95['lower'][pn_idx, fa_dest]*100:5.1f}% - {ci_95['upper'][pn_idx, fa_dest]*100:5.1f}%]")
    print(f"  PN -> PN: {T[pn_idx, pn_dest]*100:5.1f}% [{ci_95['lower'][pn_idx, pn_dest]*100:5.1f}% - {ci_95['upper'][pn_idx, pn_dest]*100:5.1f}%]")

    # Vote impact
    print()
    print("="*70)
    print("VOTE IMPACT")
    print("="*70)
    print()

    ca_votes = df['ca_primera'].sum()
    pc_votes = df['pc_primera'].sum()
    pn_votes = df['pn_primera'].sum()

    ca_to_fa = ca_votes * T[ca_idx, fa_dest]
    pc_to_fa = pc_votes * T[pc_idx, fa_dest]
    pn_to_fa = pn_votes * T[pn_idx, fa_dest]

    print(f"CA votes to FA: {ca_to_fa:,.0f}")
    print(f"PC votes to FA: {pc_to_fa:,.0f}")
    print(f"PN votes to FA: {pn_to_fa:,.0f}")
    print(f"TOTAL coalition to FA: {ca_to_fa + pc_to_fa + pn_to_fa:,.0f}")

    # Diagnostics
    print()
    print("="*70)
    print("MODEL DIAGNOSTICS")
    print("="*70)
    print()

    diag = model.get_diagnostics()
    max_rhat = np.max(diag['rhat'])
    min_ess = np.min(diag['ess'])
    mean_ess = np.mean(diag['ess'])

    print(f"Max R-hat: {max_rhat:.6f} (threshold: < 1.01) {'PASS' if max_rhat < 1.01 else 'FAIL'}")
    print(f"Min ESS: {min_ess:.0f} (threshold: > 1000) {'PASS' if min_ess > 1000 else 'WARNING'}")
    print(f"Mean ESS: {mean_ess:.0f}")

    # Save results
    print()
    print("="*70)
    print("SAVING RESULTS")
    print("="*70)
    print()

    # Transition matrix
    results_df = pd.DataFrame(T, index=origin_parties, columns=destination_parties)
    results_df.to_csv('outputs/tables/national_transition_matrix_final.csv')
    print("Saved: outputs/tables/national_transition_matrix_final.csv")

    # Detailed results with CIs
    detailed_results = []
    for i, orig in enumerate(origin_parties):
        for j, dest in enumerate(destination_parties):
            detailed_results.append({
                'origin': orig,
                'destination': dest,
                'probability': T[i, j],
                'ci_95_lower': ci_95['lower'][i, j],
                'ci_95_upper': ci_95['upper'][i, j],
                'ci_99_lower': ci_99['lower'][i, j],
                'ci_99_upper': ci_99['upper'][i, j]
            })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('outputs/tables/national_transitions_with_ci_final.csv', index=False)
    print("Saved: outputs/tables/national_transitions_with_ci_final.csv")

    # Diagnostics
    diag_df = pd.DataFrame({
        'parameter': [f'{origin_parties[i]}_to_{destination_parties[j]}'
                      for i in range(len(origin_parties))
                      for j in range(len(destination_parties))],
        'rhat': diag['rhat'].flatten(),
        'ess': diag['ess'].flatten()
    })
    diag_df.to_csv('outputs/tables/national_mcmc_diagnostics_final.csv', index=False)
    print("Saved: outputs/tables/national_mcmc_diagnostics_final.csv")

    # Summary
    summary = {
        'total_circuits': len(df),
        'ca_to_fa': T[ca_idx, fa_dest],
        'ca_to_pn': T[ca_idx, pn_dest],
        'pc_to_fa': T[pc_idx, fa_dest],
        'pn_to_fa': T[pn_idx, fa_dest],
        'max_rhat': max_rhat,
        'min_ess': min_ess,
        'ca_votes_to_fa': ca_to_fa,
        'total_coalition_to_fa': ca_to_fa + pc_to_fa + pn_to_fa
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('outputs/tables/national_analysis_summary.csv', index=False)
    print("Saved: outputs/tables/national_analysis_summary.csv")

    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Status: {'SUCCESS' if max_rhat < 1.01 and min_ess > 1000 else 'NEEDS REVIEW'}")
    print("Ready for report generation.")


if __name__ == '__main__':
    main()
