"""
FA Stratified Analysis: Urban/Rural, Metro/Interior
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
    """Load data with covariates."""
    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])

    # Try full covariates first
    data_path_cov = processed_dir / 'circuitos_full_covariates.parquet'
    data_path = processed_dir / 'circuitos_merged.parquet'

    if data_path_cov.exists():
        return pd.read_parquet(data_path_cov)
    else:
        return pd.read_parquet(data_path)


def classify_circuits(df):
    """Classify circuits by urban/rural and metro/interior."""
    df = df.copy()

    # Urban/Rural classification (if not already present)
    if 'urban_rural' not in df.columns:
        # Heuristic: circuits with > 1000 voters are urban
        df['urban_rural'] = df['total_primera'].apply(
            lambda x: 'URBANO' if x > 1000 else 'RURAL'
        )

    # Metro/Interior classification
    dept_col = 'departamento' if 'departamento' in df.columns else 'DEPARTAMENTO'
    if dept_col in df.columns:
        metro_depts = ['Montevideo', 'Canelones', 'MONTEVIDEO', 'CANELONES']
        df['metro_interior'] = df[dept_col].apply(
            lambda x: 'METROPOLITANO' if x in metro_depts else 'INTERIOR'
        )
    else:
        df['metro_interior'] = 'UNKNOWN'

    return df


def analyze_stratum(df_stratum, stratum_name, num_samples=4000, num_chains=4):
    """Analyze single stratum."""
    logger.info(f"Analizando: {stratum_name} ({len(df_stratum)} circuitos)")

    # FA analysis
    origin_cols = ['fa_primera', 'otros_primera_recalc']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Prepare
    df_stratum = df_stratum.copy()
    df_stratum['blancos_ballotage'] = df_stratum.get('blancos_ballotage', 0) + df_stratum.get('anulados_ballotage', 0)
    # Recalcular otros_primera como complemento de FA (incluye PN+PC+CA+PI+otros)
    df_stratum['otros_primera_recalc'] = df_stratum['total_primera'] - df_stratum['fa_primera']

    # Filter
    min_votes = 30
    valid = (df_stratum['total_primera'] >= min_votes) & (df_stratum['total_ballotage'] >= min_votes)
    df_clean = df_stratum[valid]

    if len(df_clean) < 20:
        logger.warning(f"  {stratum_name}: Pocos circuitos, saltando")
        return None

    logger.info(f"  Circuitos válidos: {len(df_clean)}")

    try:
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
            progressbar=False
        )

        T = model.get_transition_matrix()
        ci = model.get_credible_intervals(prob=0.95)
        diag = model.get_diagnostics()

        result = {
            'stratum': stratum_name,
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

        logger.info(f"  {stratum_name}: FA retention = {T[0,0]:.1%}")

        return result

    except Exception as e:
        logger.error(f"  {stratum_name}: Error - {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--chains', type=int, default=4)
    args = parser.parse_args()

    config = get_config()

    logger.info("="*70)
    logger.info("ANÁLISIS ESTRATIFICADO FA")
    logger.info("="*70)

    # Load data
    df = load_data(config)

    # Classify
    df = classify_circuits(df)

    # Analyze strata
    results = []

    # Urban/Rural
    for stratum_value in ['URBANO', 'RURAL']:
        df_stratum = df[df['urban_rural'] == stratum_value]
        result = analyze_stratum(df_stratum, f"UrbanRural_{stratum_value}", args.samples, args.chains)
        if result:
            results.append(result)

    # Metro/Interior
    for stratum_value in ['METROPOLITANO', 'INTERIOR']:
        df_stratum = df[df['metro_interior'] == stratum_value]
        result = analyze_stratum(df_stratum, f"MetroInterior_{stratum_value}", args.samples, args.chains)
        if result:
            results.append(result)

    # Save results
    output_dirs = config.get_output_dirs()
    results_dir = Path(config.project_root_path) / 'outputs' / 'results'
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Save traces
    for r in results:
        with open(results_dir / f"fa_stratified_{r['stratum']}_2024.pkl", 'wb') as f:
            pickle.dump(r['trace'], f)

    # Summary table
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'trace'} for r in results])

    # Save CSV
    df_results.to_csv(tables_dir / 'fa_stratified_summary_2024.csv', index=False, float_format='%.4f')

    # Save LaTeX
    df_latex = df_results[['stratum', 'fa_retention', 'fa_retention_ci_lower', 'fa_retention_ci_upper']].copy()
    df_latex['CI'] = df_latex.apply(
        lambda row: f"[{row['fa_retention_ci_lower']:.1%}, {row['fa_retention_ci_upper']:.1%}]",
        axis=1
    )
    df_latex = df_latex[['stratum', 'fa_retention', 'CI']]
    df_latex.columns = ['Estrato', 'Retención FA', 'IC 95%']
    df_latex['Retención FA'] = df_latex['Retención FA'].apply(lambda x: f"{x:.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lcc',
        caption='Análisis estratificado FA 2024',
        label='tab:fa_stratified'
    )

    with open(latex_dir / 'fa_stratified_summary.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    logger.info("Análisis estratificado completado")

    return 0


if __name__ == '__main__':
    sys.exit(main())
