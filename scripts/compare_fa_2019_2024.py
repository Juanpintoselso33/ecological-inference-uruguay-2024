"""
Temporal Comparison: FA behavior 2019 vs 2024
"""

import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.king_ei import KingEI
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(config):
    """Load both 2019 and 2024 data."""
    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])

    df_2024 = pd.read_parquet(processed_dir / 'circuitos_merged.parquet')
    df_2019 = pd.read_parquet(processed_dir / 'circuitos_merged_2019.parquet')

    logger.info(f"2024: {len(df_2024)} circuitos")
    logger.info(f"2019: {len(df_2019)} circuitos")

    return df_2024, df_2019


def analyze_year(df, year, num_samples=4000, num_chains=4):
    """Analyze single year."""
    logger.info(f"Analizando {year}...")

    # FA analysis
    origin_cols = ['fa_primera', 'otros_primera_recalc']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Prepare data
    df = df.copy()
    df['blancos_ballotage'] = df.get('blancos_ballotage', 0) + df.get('anulados_ballotage', 0)
    # Recalcular otros_primera como complemento de FA (incluye PN+PC+CA+PI+otros)
    df['otros_primera_recalc'] = df['total_primera'] - df['fa_primera']

    # Filter
    min_votes = 50
    valid = (df['total_primera'] >= min_votes) & (df['total_ballotage'] >= min_votes)
    df_clean = df[valid]

    logger.info(f"  Circuitos válidos: {len(df_clean)}")

    # Fit model
    model = KingEI(
        num_samples=num_samples,
        num_chains=num_chains,
        num_warmup=num_samples // 2,
        random_seed=42
    )

    model.fit(
        data=df_clean,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    # Extract results
    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(prob=0.95)
    diag = model.get_diagnostics()

    result = {
        'year': year,
        'n_circuits': len(df_clean),
        'fa_retention': T[0, 0],
        'fa_retention_ci_lower': ci['lower'][0, 0],
        'fa_retention_ci_upper': ci['upper'][0, 0],
        'fa_to_pn': T[0, 1],
        'fa_to_pn_ci_lower': ci['lower'][0, 1],
        'fa_to_pn_ci_upper': ci['upper'][0, 1],
        'fa_to_blancos': T[0, 2],
        'rhat_max': np.max(diag['rhat']),
        'ess_min': np.min(diag['ess']),
        'trace': model.trace_
    }

    logger.info(f"  {year}: FA retention = {T[0,0]:.1%}")

    return result


def save_results(result_2024, result_2019, config):
    """Save comparison results."""
    logger.info("Guardando resultados...")

    output_dirs = config.get_output_dirs()
    results_dir = Path(config.project_root_path) / 'outputs' / 'results'
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Save traces
    with open(results_dir / 'fa_temporal_2024.pkl', 'wb') as f:
        pickle.dump(result_2024['trace'], f)

    with open(results_dir / 'fa_temporal_2019.pkl', 'wb') as f:
        pickle.dump(result_2019['trace'], f)

    # Create comparison table
    comparison = []
    for metric in ['fa_retention', 'fa_to_pn', 'fa_to_blancos']:
        comparison.append({
            'Metric': metric.replace('_', ' ').title(),
            'Value_2019': result_2019[metric],
            'CI_2019': f"[{result_2019.get(metric + '_ci_lower', 0):.1%}, {result_2019.get(metric + '_ci_upper', 0):.1%}]",
            'Value_2024': result_2024[metric],
            'CI_2024': f"[{result_2024.get(metric + '_ci_lower', 0):.1%}, {result_2024.get(metric + '_ci_upper', 0):.1%}]",
            'Change': result_2024[metric] - result_2019[metric]
        })

    df_comparison = pd.DataFrame(comparison)

    # Save CSV
    df_comparison.to_csv(tables_dir / 'fa_temporal_comparison_2019_2024.csv', index=False, float_format='%.4f')

    # Save LaTeX
    df_latex = df_comparison.copy()
    df_latex['Value_2019'] = df_latex['Value_2019'].apply(lambda x: f"{x:.1%}")
    df_latex['Value_2024'] = df_latex['Value_2024'].apply(lambda x: f"{x:.1%}")
    df_latex['Change'] = df_latex['Change'].apply(lambda x: f"{x:+.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lcccc',
        caption='Comparación temporal FA 2019 vs 2024',
        label='tab:fa_temporal_comparison'
    )

    with open(latex_dir / 'fa_temporal_comparison.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    logger.info("Resultados guardados")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--chains', type=int, default=4)
    args = parser.parse_args()

    config = get_config()

    logger.info("="*70)
    logger.info("COMPARACIÓN TEMPORAL FA 2019 vs 2024")
    logger.info("="*70)

    # Load data
    df_2024, df_2019 = load_data(config)

    # Analyze both years
    result_2024 = analyze_year(df_2024, 2024, args.samples, args.chains)
    result_2019 = analyze_year(df_2019, 2019, args.samples, args.chains)

    # Save
    save_results(result_2024, result_2019, config)

    # Print comparison
    print("\n" + "="*70)
    print("COMPARACIÓN TEMPORAL FA")
    print("="*70 + "\n")
    print(f"{'Métrica':<20s} {'2019':<12s} {'2024':<12s} {'Cambio':<12s}")
    print("-"*70)
    print(f"{'FA Retention':<20s} {result_2019['fa_retention']:>10.1%}  {result_2024['fa_retention']:>10.1%}  "
          f"{result_2024['fa_retention'] - result_2019['fa_retention']:>+10.1%}")
    print(f"{'FA → PN':<20s} {result_2019['fa_to_pn']:>10.1%}  {result_2024['fa_to_pn']:>10.1%}  "
          f"{result_2024['fa_to_pn'] - result_2019['fa_to_pn']:>+10.1%}")
    print("\n" + "="*70 + "\n")

    logger.info("Análisis completado")

    return 0


if __name__ == '__main__':
    sys.exit(main())
