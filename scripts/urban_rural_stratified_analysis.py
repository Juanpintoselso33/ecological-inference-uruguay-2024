"""
Stratified King's Ecological Inference: Urban vs Rural Circuits

Compares Cabildo Abierto defection patterns between urban and rural areas.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def run_stratified_analysis(
    data_path: str,
    output_dir: str = "outputs/tables",
    reports_dir: str = "outputs/reports",
    num_samples: int = 2000,
    num_chains: int = 4,
    num_warmup: int = 2000,
    random_seed: int = 42
) -> Dict:
    """
    Run stratified EI analysis comparing urban vs rural circuits.

    Args:
        data_path: Path to full dataset with urban_rural column
        output_dir: Directory to save result tables
        reports_dir: Directory to save summary report
        num_samples: MCMC samples per chain
        num_chains: Number of MCMC chains
        num_warmup: Warmup samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with results for both strata
    """
    logger.info("="*70)
    logger.info("STRATIFIED KING'S EI ANALYSIS: URBAN vs RURAL")
    logger.info("="*70)

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Total circuits: {len(df):,}")

    # Check urban/rural distribution
    logger.info("\nUrban/Rural distribution:")
    urban_rural_counts = df['urban_rural'].value_counts()
    for category, count in urban_rural_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  {category}: {count:,} ({pct:.1f}%)")

    # Split into urban and rural
    df_urban = df[df['urban_rural'] == 'urbano'].copy()
    df_rural = df[df['urban_rural'] == 'rural'].copy()

    # Verify sample sizes
    logger.info(f"\nUrban circuits: {len(df_urban):,}")
    logger.info(f"Rural circuits: {len(df_rural):,}")

    if len(df_urban) < 100:
        logger.warning(f"WARNING: Urban sample size ({len(df_urban)}) is below 100!")
    if len(df_rural) < 100:
        logger.warning(f"WARNING: Rural sample size ({len(df_rural)}) is below 100!")

    # Define origin and destination columns
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pi_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Results dictionary
    results = {
        'urban': {},
        'rural': {},
        'comparison': {}
    }

    # Run EI for both strata
    for stratum_name, df_stratum in [('urban', df_urban), ('rural', df_rural)]:
        logger.info("\n" + "="*70)
        logger.info(f"RUNNING KING'S EI FOR {stratum_name.upper()} CIRCUITS")
        logger.info("="*70)
        logger.info(f"Sample size: {len(df_stratum):,} circuits")

        # Initialize model
        model = KingEI(
            num_samples=num_samples,
            num_chains=num_chains,
            num_warmup=num_warmup,
            target_accept=0.9,
            random_seed=random_seed
        )

        # Fit model
        logger.info(f"\nFitting King's EI model ({num_chains} chains, {num_samples} samples, {num_warmup} warmup)...")
        model.fit(
            data=df_stratum,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=True
        )

        # Extract results
        T = model.get_transition_matrix()
        uncertainty = model.get_uncertainty()
        ci_95 = model.get_credible_intervals(0.95)
        diagnostics = model.get_diagnostics()

        # Store results
        results[stratum_name] = {
            'model': model,
            'n_circuits': len(df_stratum),
            'transition_matrix': T,
            'uncertainty': uncertainty,
            'credible_intervals': ci_95,
            'diagnostics': diagnostics,
            'origin_parties': model.origin_party_names_,
            'destination_parties': model.destination_party_names_
        }

        # Print summary
        logger.info(f"\n{stratum_name.upper()} RESULTS:")
        logger.info(f"Max R-hat: {np.max(diagnostics['rhat']):.4f} (should be < 1.01)")
        logger.info(f"Min ESS: {np.min(diagnostics['ess']):.0f} (should be > 400)")

        # Print key transitions
        logger.info(f"\nKey transitions for {stratum_name.upper()}:")
        party_idx_map = {party: i for i, party in enumerate(model.origin_party_names_)}

        # CA transitions
        if 'CA' in party_idx_map:
            ca_idx = party_idx_map['CA']
            ca_to_fa = T[ca_idx, 0]  # → FA
            ca_to_pn = T[ca_idx, 1]  # → PN
            ca_to_fa_ci = (ci_95['lower'][ca_idx, 0], ci_95['upper'][ca_idx, 0])
            ca_to_pn_ci = (ci_95['lower'][ca_idx, 1], ci_95['upper'][ca_idx, 1])

            logger.info(f"  CA -> FA: {ca_to_fa*100:.2f}% [{ca_to_fa_ci[0]*100:.2f}%, {ca_to_fa_ci[1]*100:.2f}%]")
            logger.info(f"  CA -> PN: {ca_to_pn*100:.2f}% [{ca_to_pn_ci[0]*100:.2f}%, {ca_to_pn_ci[1]*100:.2f}%]")

        # PC transitions
        if 'PC' in party_idx_map:
            pc_idx = party_idx_map['PC']
            pc_to_fa = T[pc_idx, 0]
            pc_to_pn = T[pc_idx, 1]
            pc_to_fa_ci = (ci_95['lower'][pc_idx, 0], ci_95['upper'][pc_idx, 0])
            pc_to_pn_ci = (ci_95['lower'][pc_idx, 1], ci_95['upper'][pc_idx, 1])

            logger.info(f"  PC -> FA: {pc_to_fa*100:.2f}% [{pc_to_fa_ci[0]*100:.2f}%, {pc_to_fa_ci[1]*100:.2f}%]")
            logger.info(f"  PC -> PN: {pc_to_pn*100:.2f}% [{pc_to_pn_ci[0]*100:.2f}%, {pc_to_pn_ci[1]*100:.2f}%]")

        # PN loyalty
        if 'PN' in party_idx_map:
            pn_idx = party_idx_map['PN']
            pn_to_fa = T[pn_idx, 0]
            pn_to_pn = T[pn_idx, 1]
            pn_to_fa_ci = (ci_95['lower'][pn_idx, 0], ci_95['upper'][pn_idx, 0])

            logger.info(f"  PN -> FA: {pn_to_fa*100:.2f}% [{pn_to_fa_ci[0]*100:.2f}%, {pn_to_fa_ci[1]*100:.2f}%]")
            logger.info(f"  PN -> PN: {pn_to_pn*100:.2f}%")

    # Compare results
    logger.info("\n" + "="*70)
    logger.info("URBAN vs RURAL COMPARISON")
    logger.info("="*70)

    urban_T = results['urban']['transition_matrix']
    rural_T = results['rural']['transition_matrix']
    urban_ci = results['urban']['credible_intervals']
    rural_ci = results['rural']['credible_intervals']

    # Get party indices
    party_idx_map = {party: i for i, party in enumerate(results['urban']['origin_parties'])}

    comparison_data = []

    # CA comparison
    if 'CA' in party_idx_map:
        ca_idx = party_idx_map['CA']

        # CA → FA
        urban_ca_fa = urban_T[ca_idx, 0]
        rural_ca_fa = rural_T[ca_idx, 0]
        diff_ca_fa = urban_ca_fa - rural_ca_fa
        urban_ci_ca_fa = (urban_ci['lower'][ca_idx, 0], urban_ci['upper'][ca_idx, 0])
        rural_ci_ca_fa = (rural_ci['lower'][ca_idx, 0], rural_ci['upper'][ca_idx, 0])

        logger.info(f"\nCA -> FA:")
        logger.info(f"  Urban: {urban_ca_fa*100:.2f}% [{urban_ci_ca_fa[0]*100:.2f}%, {urban_ci_ca_fa[1]*100:.2f}%]")
        logger.info(f"  Rural: {rural_ca_fa*100:.2f}% [{rural_ci_ca_fa[0]*100:.2f}%, {rural_ci_ca_fa[1]*100:.2f}%]")
        logger.info(f"  Difference: {diff_ca_fa*100:+.2f} pp")

        # Check if CIs overlap
        ci_overlap = not (urban_ci_ca_fa[1] < rural_ci_ca_fa[0] or rural_ci_ca_fa[1] < urban_ci_ca_fa[0])
        logger.info(f"  Credible intervals overlap: {ci_overlap}")

        comparison_data.append({
            'transition': 'CA → FA',
            'urban_mean': urban_ca_fa,
            'urban_ci_lower': urban_ci_ca_fa[0],
            'urban_ci_upper': urban_ci_ca_fa[1],
            'rural_mean': rural_ca_fa,
            'rural_ci_lower': rural_ci_ca_fa[0],
            'rural_ci_upper': rural_ci_ca_fa[1],
            'difference': diff_ca_fa,
            'ci_overlap': ci_overlap
        })

        # CA → PN
        urban_ca_pn = urban_T[ca_idx, 1]
        rural_ca_pn = rural_T[ca_idx, 1]
        diff_ca_pn = urban_ca_pn - rural_ca_pn
        urban_ci_ca_pn = (urban_ci['lower'][ca_idx, 1], urban_ci['upper'][ca_idx, 1])
        rural_ci_ca_pn = (rural_ci['lower'][ca_idx, 1], rural_ci['upper'][ca_idx, 1])

        logger.info(f"\nCA -> PN:")
        logger.info(f"  Urban: {urban_ca_pn*100:.2f}% [{urban_ci_ca_pn[0]*100:.2f}%, {urban_ci_ca_pn[1]*100:.2f}%]")
        logger.info(f"  Rural: {rural_ca_pn*100:.2f}% [{rural_ci_ca_pn[0]*100:.2f}%, {rural_ci_ca_pn[1]*100:.2f}%]")
        logger.info(f"  Difference: {diff_ca_pn*100:+.2f} pp")

        ci_overlap_pn = not (urban_ci_ca_pn[1] < rural_ci_ca_pn[0] or rural_ci_ca_pn[1] < urban_ci_ca_pn[0])
        logger.info(f"  Credible intervals overlap: {ci_overlap_pn}")

        comparison_data.append({
            'transition': 'CA → PN',
            'urban_mean': urban_ca_pn,
            'urban_ci_lower': urban_ci_ca_pn[0],
            'urban_ci_upper': urban_ci_ca_pn[1],
            'rural_mean': rural_ca_pn,
            'rural_ci_lower': rural_ci_ca_pn[0],
            'rural_ci_upper': rural_ci_ca_pn[1],
            'difference': diff_ca_pn,
            'ci_overlap': ci_overlap_pn
        })

    # PC comparison
    if 'PC' in party_idx_map:
        pc_idx = party_idx_map['PC']

        urban_pc_fa = urban_T[pc_idx, 0]
        rural_pc_fa = rural_T[pc_idx, 0]
        diff_pc_fa = urban_pc_fa - rural_pc_fa
        urban_ci_pc_fa = (urban_ci['lower'][pc_idx, 0], urban_ci['upper'][pc_idx, 0])
        rural_ci_pc_fa = (rural_ci['lower'][pc_idx, 0], rural_ci['upper'][pc_idx, 0])

        logger.info(f"\nPC -> FA:")
        logger.info(f"  Urban: {urban_pc_fa*100:.2f}% [{urban_ci_pc_fa[0]*100:.2f}%, {urban_ci_pc_fa[1]*100:.2f}%]")
        logger.info(f"  Rural: {rural_pc_fa*100:.2f}% [{rural_ci_pc_fa[0]*100:.2f}%, {rural_ci_pc_fa[1]*100:.2f}%]")
        logger.info(f"  Difference: {diff_pc_fa*100:+.2f} pp")

        ci_overlap_pc = not (urban_ci_pc_fa[1] < rural_ci_pc_fa[0] or rural_ci_pc_fa[1] < urban_ci_pc_fa[0])
        logger.info(f"  Credible intervals overlap: {ci_overlap_pc}")

        comparison_data.append({
            'transition': 'PC → FA',
            'urban_mean': urban_pc_fa,
            'urban_ci_lower': urban_ci_pc_fa[0],
            'urban_ci_upper': urban_ci_pc_fa[1],
            'rural_mean': rural_pc_fa,
            'rural_ci_lower': rural_ci_pc_fa[0],
            'rural_ci_upper': rural_ci_pc_fa[1],
            'difference': diff_pc_fa,
            'ci_overlap': ci_overlap_pc
        })

    # PN comparison
    if 'PN' in party_idx_map:
        pn_idx = party_idx_map['PN']

        urban_pn_fa = urban_T[pn_idx, 0]
        rural_pn_fa = rural_T[pn_idx, 0]
        diff_pn_fa = urban_pn_fa - rural_pn_fa
        urban_ci_pn_fa = (urban_ci['lower'][pn_idx, 0], urban_ci['upper'][pn_idx, 0])
        rural_ci_pn_fa = (rural_ci['lower'][pn_idx, 0], rural_ci['upper'][pn_idx, 0])

        logger.info(f"\nPN -> FA:")
        logger.info(f"  Urban: {urban_pn_fa*100:.2f}% [{urban_ci_pn_fa[0]*100:.2f}%, {urban_ci_pn_fa[1]*100:.2f}%]")
        logger.info(f"  Rural: {rural_pn_fa*100:.2f}% [{rural_ci_pn_fa[0]*100:.2f}%, {rural_ci_pn_fa[1]*100:.2f}%]")
        logger.info(f"  Difference: {diff_pn_fa*100:+.2f} pp")

        ci_overlap_pn_fa = not (urban_ci_pn_fa[1] < rural_ci_pn_fa[0] or rural_ci_pn_fa[1] < urban_ci_pn_fa[0])
        logger.info(f"  Credible intervals overlap: {ci_overlap_pn_fa}")

        comparison_data.append({
            'transition': 'PN → FA',
            'urban_mean': urban_pn_fa,
            'urban_ci_lower': urban_ci_pn_fa[0],
            'urban_ci_upper': urban_ci_pn_fa[1],
            'rural_mean': rural_pn_fa,
            'rural_ci_lower': rural_ci_pn_fa[0],
            'rural_ci_upper': rural_ci_pn_fa[1],
            'difference': diff_pn_fa,
            'ci_overlap': ci_overlap_pn_fa
        })

    results['comparison'] = comparison_data

    # Calculate vote impact
    logger.info("\n" + "="*70)
    logger.info("VOTE IMPACT ESTIMATION")
    logger.info("="*70)

    # Get total CA votes in each stratum
    ca_votes_urban = df_urban['ca_primera'].sum()
    ca_votes_rural = df_rural['ca_primera'].sum()

    logger.info(f"\nTotal CA votes (primera vuelta):")
    logger.info(f"  Urban: {ca_votes_urban:,}")
    logger.info(f"  Rural: {ca_votes_rural:,}")

    if 'CA' in party_idx_map:
        # Estimated CA→FA votes
        ca_to_fa_votes_urban = ca_votes_urban * urban_ca_fa
        ca_to_fa_votes_rural = ca_votes_rural * rural_ca_fa

        logger.info(f"\nEstimated CA -> FA votes (ballotage):")
        logger.info(f"  Urban: {ca_to_fa_votes_urban:,.0f} ({urban_ca_fa*100:.1f}% of {ca_votes_urban:,})")
        logger.info(f"  Rural: {ca_to_fa_votes_rural:,.0f} ({rural_ca_fa*100:.1f}% of {ca_votes_rural:,})")
        logger.info(f"  Total: {ca_to_fa_votes_urban + ca_to_fa_votes_rural:,.0f}")

    # Save results to tables
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)

    # Summary table
    summary_data = []
    for stratum_name in ['urban', 'rural']:
        stratum_results = results[stratum_name]
        T = stratum_results['transition_matrix']
        ci = stratum_results['credible_intervals']
        diag = stratum_results['diagnostics']

        if 'CA' in party_idx_map:
            ca_idx = party_idx_map['CA']
            summary_data.append({
                'stratum': stratum_name,
                'n_circuits': stratum_results['n_circuits'],
                'ca_to_fa': T[ca_idx, 0],
                'ca_to_fa_ci_lower': ci['lower'][ca_idx, 0],
                'ca_to_fa_ci_upper': ci['upper'][ca_idx, 0],
                'ca_to_pn': T[ca_idx, 1],
                'ca_to_pn_ci_lower': ci['lower'][ca_idx, 1],
                'ca_to_pn_ci_upper': ci['upper'][ca_idx, 1],
                'max_rhat': np.max(diag['rhat']),
                'min_ess': np.min(diag['ess'])
            })

    summary_df = pd.DataFrame(summary_data)
    summary_path = Path(output_dir) / 'urban_rural_comparison.csv'
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary table saved to: {summary_path}")

    # Detailed comparison table
    comparison_df = pd.DataFrame(results['comparison'])
    comparison_path = Path(output_dir) / 'urban_rural_detailed_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Detailed comparison saved to: {comparison_path}")

    # Full transition matrices
    for stratum_name in ['urban', 'rural']:
        T = results[stratum_name]['transition_matrix']
        origin_parties = results[stratum_name]['origin_parties']
        dest_parties = results[stratum_name]['destination_parties']

        transition_df = pd.DataFrame(
            T,
            index=origin_parties,
            columns=dest_parties
        )
        transition_path = Path(output_dir) / f'transition_matrix_{stratum_name}.csv'
        transition_df.to_csv(transition_path)
        logger.info(f"{stratum_name.capitalize()} transition matrix saved to: {transition_path}")

    # Generate summary report
    generate_summary_report(results, Path(reports_dir) / 'urban_rural_analysis_summary.md')

    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)

    return results


def generate_summary_report(results: Dict, output_path: Path) -> None:
    """Generate markdown summary report."""

    report_lines = []
    report_lines.append("# Urban vs Rural Stratified Analysis: King's Ecological Inference")
    report_lines.append("")
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append("This analysis compares Cabildo Abierto (CA) defection patterns between urban and rural circuits in Uruguay's 2024 electoral runoff using King's Ecological Inference.")
    report_lines.append("")

    # Sample sizes
    report_lines.append("## Sample Sizes")
    report_lines.append("")
    report_lines.append(f"- **Urban circuits**: {results['urban']['n_circuits']:,} (86.8%)")
    report_lines.append(f"- **Rural circuits**: {results['rural']['n_circuits']:,} (13.2%)")
    report_lines.append("")

    # Key findings
    report_lines.append("## Key Findings")
    report_lines.append("")

    # Get CA results
    party_idx_map = {party: i for i, party in enumerate(results['urban']['origin_parties'])}

    if 'CA' in party_idx_map:
        ca_idx = party_idx_map['CA']

        urban_T = results['urban']['transition_matrix']
        rural_T = results['rural']['transition_matrix']
        urban_ci = results['urban']['credible_intervals']
        rural_ci = results['rural']['credible_intervals']

        urban_ca_fa = urban_T[ca_idx, 0]
        rural_ca_fa = rural_T[ca_idx, 0]
        diff_ca_fa = urban_ca_fa - rural_ca_fa

        urban_ca_pn = urban_T[ca_idx, 1]
        rural_ca_pn = rural_T[ca_idx, 1]

        report_lines.append("### Cabildo Abierto → Frente Amplio")
        report_lines.append("")
        report_lines.append(f"- **Urban**: {urban_ca_fa*100:.2f}% [95% CI: {urban_ci['lower'][ca_idx, 0]*100:.2f}%, {urban_ci['upper'][ca_idx, 0]*100:.2f}%]")
        report_lines.append(f"- **Rural**: {rural_ca_fa*100:.2f}% [95% CI: {rural_ci['lower'][ca_idx, 0]*100:.2f}%, {rural_ci['upper'][ca_idx, 0]*100:.2f}%]")
        report_lines.append(f"- **Difference**: {diff_ca_fa*100:+.2f} percentage points")
        report_lines.append("")

        # Interpretation
        if abs(diff_ca_fa) > 0.05:
            direction = "higher" if diff_ca_fa > 0 else "lower"
            report_lines.append(f"**Interpretation**: Urban areas show **substantively {direction}** CA→FA defection rates (difference > 5 pp).")
        else:
            report_lines.append("**Interpretation**: Urban and rural CA→FA defection rates are **similar** (difference < 5 pp).")
        report_lines.append("")

        report_lines.append("### Cabildo Abierto → Partido Nacional (Coalition Loyalty)")
        report_lines.append("")
        report_lines.append(f"- **Urban**: {urban_ca_pn*100:.2f}% [95% CI: {urban_ci['lower'][ca_idx, 1]*100:.2f}%, {urban_ci['upper'][ca_idx, 1]*100:.2f}%]")
        report_lines.append(f"- **Rural**: {rural_ca_pn*100:.2f}% [95% CI: {rural_ci['lower'][ca_idx, 1]*100:.2f}%, {rural_ci['upper'][ca_idx, 1]*100:.2f}%]")
        report_lines.append("")

    # Other coalition parties
    if 'PC' in party_idx_map:
        pc_idx = party_idx_map['PC']
        urban_pc_fa = results['urban']['transition_matrix'][pc_idx, 0]
        rural_pc_fa = results['rural']['transition_matrix'][pc_idx, 0]

        report_lines.append("### Partido Colorado → Frente Amplio")
        report_lines.append("")
        report_lines.append(f"- **Urban**: {urban_pc_fa*100:.2f}%")
        report_lines.append(f"- **Rural**: {rural_pc_fa*100:.2f}%")
        report_lines.append("")

    if 'PN' in party_idx_map:
        pn_idx = party_idx_map['PN']
        urban_pn_fa = results['urban']['transition_matrix'][pn_idx, 0]
        rural_pn_fa = results['rural']['transition_matrix'][pn_idx, 0]

        report_lines.append("### Partido Nacional → Frente Amplio (Direct Defection)")
        report_lines.append("")
        report_lines.append(f"- **Urban**: {urban_pn_fa*100:.2f}%")
        report_lines.append(f"- **Rural**: {rural_pn_fa*100:.2f}%")
        report_lines.append("")

    # Model diagnostics
    report_lines.append("## Model Diagnostics")
    report_lines.append("")
    report_lines.append("### Urban Model")
    report_lines.append("")
    urban_diag = results['urban']['diagnostics']
    report_lines.append(f"- **Max R-hat**: {np.max(urban_diag['rhat']):.4f} (target: < 1.01)")
    report_lines.append(f"- **Min ESS**: {np.min(urban_diag['ess']):.0f} (target: > 400)")
    report_lines.append("")

    report_lines.append("### Rural Model")
    report_lines.append("")
    rural_diag = results['rural']['diagnostics']
    report_lines.append(f"- **Max R-hat**: {np.max(rural_diag['rhat']):.4f} (target: < 1.01)")
    report_lines.append(f"- **Min ESS**: {np.min(rural_diag['ess']):.0f} (target: > 400)")
    report_lines.append("")

    # Context and interpretation
    report_lines.append("## Political Context")
    report_lines.append("")
    report_lines.append("Cabildo Abierto is a right-wing party that formed part of the coalition government (2020-2024) alongside Partido Nacional and Partido Colorado. In the 2024 runoff, CA voters faced a choice between:")
    report_lines.append("")
    report_lines.append("1. **Coalition loyalty** (PN): Supporting the coalition partner")
    report_lines.append("2. **Ideological defection** (FA): Crossing to the left-wing alternative")
    report_lines.append("")
    report_lines.append("### Why Might Urban/Rural Patterns Differ?")
    report_lines.append("")
    report_lines.append("1. **Socioeconomic composition**: Urban areas typically more educated, professional class")
    report_lines.append("2. **Media exposure**: Urban voters have greater access to diverse media sources")
    report_lines.append("3. **Traditional values**: Rural areas often show stronger traditional/conservative orientation")
    report_lines.append("4. **Coalition ties**: Rural areas may have stronger historical ties to PN")
    report_lines.append("")

    # Methodology
    report_lines.append("## Methodology")
    report_lines.append("")
    report_lines.append("**Model**: King's Ecological Inference (Bayesian)")
    report_lines.append("- MCMC sampling: 2,000 samples per chain")
    report_lines.append("- Chains: 4")
    report_lines.append("- Warmup: 2,000")
    report_lines.append("- Prior: Dirichlet (ensures row probabilities sum to 1)")
    report_lines.append("")
    report_lines.append("**Data**: 7,271 electoral circuits from Uruguay's 2024 elections")
    report_lines.append("- Primera vuelta (October): Origin votes by party")
    report_lines.append("- Ballotage (November): Destination votes (FA vs PN)")
    report_lines.append("")

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Summary report saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run stratified King\'s EI analysis comparing urban vs rural circuits'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged_with_urban.parquet',
        help='Path to data with urban_rural column'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/tables',
        help='Directory to save result tables'
    )
    parser.add_argument(
        '--reports-dir',
        type=str,
        default='outputs/reports',
        help='Directory to save summary report'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of MCMC samples per chain'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Number of MCMC chains'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2000,
        help='Number of warmup samples'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Run analysis
    results = run_stratified_analysis(
        data_path=args.data,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        num_samples=args.samples,
        num_chains=args.chains,
        num_warmup=args.warmup,
        random_seed=args.seed
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - {args.output_dir}/urban_rural_comparison.csv")
    print(f"  - {args.output_dir}/urban_rural_detailed_comparison.csv")
    print(f"  - {args.output_dir}/transition_matrix_urban.csv")
    print(f"  - {args.output_dir}/transition_matrix_rural.csv")
    print(f"  - {args.reports_dir}/urban_rural_analysis_summary.md")


if __name__ == '__main__':
    main()
