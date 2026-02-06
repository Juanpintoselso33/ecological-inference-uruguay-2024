"""
Análisis Departamental - Comportamiento Electoral del Frente Amplio 2024

Estima las tasas de transferencia de votos FA por departamento (19 departamentos).
"""

import sys
from pathlib import Path
import pickle
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.king_ei import KingEI
from src.utils.config import get_config
from src.utils.logger import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


def load_data(config):
    """Load processed electoral data."""
    logger.info("Cargando datos procesados...")

    data_dirs = config.get_data_dirs()
    data_path = Path(data_dirs['processed']) / 'circuitos_merged.parquet'
    df = pd.read_parquet(data_path)

    logger.info(f"Datos cargados: {len(df)} circuitos")
    return df


def analyze_department(df_dept, dept_name, num_samples=4000, num_chains=4):
    """Analyze single department."""
    logger.info(f"Analizando departamento: {dept_name} ({len(df_dept)} circuitos)")

    # Define columns
    origin_cols = ['fa_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Create combined blancos column
    df_dept = df_dept.copy()
    df_dept['blancos_ballotage'] = df_dept['blancos_ballotage'] + df_dept['anulados_ballotage']

    # Filter valid circuits
    min_votes = 30
    valid = (df_dept['total_primera'] >= min_votes) & (df_dept['total_ballotage'] >= min_votes)
    df_clean = df_dept[valid].copy()

    if len(df_clean) < 10:
        logger.warning(f"  {dept_name}: Pocos circuitos válidos ({len(df_clean)}), saltando")
        return None

    try:
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
            progressbar=False
        )

        # Extract results
        T = model.get_transition_matrix()
        ci = model.get_credible_intervals(prob=0.95)
        diag = model.get_diagnostics()

        # FA retention
        fa_retention = T[0, 0]
        fa_retention_ci_lower = ci['lower'][0, 0]
        fa_retention_ci_upper = ci['upper'][0, 0]

        # FA defection to PN
        fa_to_pn = T[0, 1]
        fa_to_pn_ci_lower = ci['lower'][0, 1]
        fa_to_pn_ci_upper = ci['upper'][0, 1]

        # FA to blancos
        fa_to_blancos = T[0, 2]

        # Diagnostics
        rhat_max = np.max(diag['rhat'])
        ess_min = np.min(diag['ess'])

        result = {
            'departamento': dept_name,
            'circuitos': len(df_clean),
            'fa_retention': fa_retention,
            'fa_retention_ci_lower': fa_retention_ci_lower,
            'fa_retention_ci_upper': fa_retention_ci_upper,
            'fa_to_pn': fa_to_pn,
            'fa_to_pn_ci_lower': fa_to_pn_ci_lower,
            'fa_to_pn_ci_upper': fa_to_pn_ci_upper,
            'fa_to_blancos': fa_to_blancos,
            'rhat_max': rhat_max,
            'ess_min': ess_min,
            'trace': model.trace_
        }

        logger.info(f"  {dept_name}: FA retention = {fa_retention:.1%} [{fa_retention_ci_lower:.1%}, {fa_retention_ci_upper:.1%}]")

        return result

    except Exception as e:
        logger.error(f"  {dept_name}: Error - {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=4000)
    parser.add_argument('--chains', type=int, default=4)
    args = parser.parse_args()

    config = get_config()

    logger.info("="*70)
    logger.info("ANÁLISIS DEPARTAMENTAL - FRENTE AMPLIO 2024")
    logger.info("="*70)

    # Load data
    df = load_data(config)

    # Get department column
    dept_col = 'departamento' if 'departamento' in df.columns else 'DEPARTAMENTO'
    departments = sorted(df[dept_col].unique())

    logger.info(f"Departamentos a analizar: {len(departments)}")

    # Analyze each department
    results = []
    for dept in departments:
        df_dept = df[df[dept_col] == dept]
        result = analyze_department(df_dept, dept, args.samples, args.chains)
        if result is not None:
            results.append(result)

    # Create summary DataFrame
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'trace'}
        for r in results
    ])

    # Sort by FA retention
    df_results = df_results.sort_values('fa_retention', ascending=False)

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
        trace_path = results_dir / f"fa_departmental_{r['departamento']}_2024.pkl"
        with open(trace_path, 'wb') as f:
            pickle.dump(r['trace'], f)

    # Save CSV
    csv_path = tables_dir / 'fa_departmental_summary_2024.csv'
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Resultados guardados: {csv_path}")

    # Save LaTeX - Top 10 retention
    latex_path = latex_dir / 'fa_departmental_top10_retention.tex'
    df_top10 = df_results.head(10).copy()
    df_latex = df_top10[['departamento', 'fa_retention', 'fa_retention_ci_lower', 'fa_retention_ci_upper']].copy()
    df_latex['CI'] = df_latex.apply(
        lambda row: f"[{row['fa_retention_ci_lower']:.1%}, {row['fa_retention_ci_upper']:.1%}]",
        axis=1
    )
    df_latex = df_latex[['departamento', 'fa_retention', 'CI']]
    df_latex.columns = ['Departamento', 'Retención FA', 'IC 95%']
    df_latex['Retención FA'] = df_latex['Retención FA'].apply(lambda x: f"{x:.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lcc',
        caption='Top 10 departamentos por retención FA 2024',
        label='tab:fa_retention_top10'
    )

    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)

    logger.info(f"LaTeX guardado: {latex_path}")
    logger.info("Análisis departamental completado")

    return 0


if __name__ == '__main__':
    sys.exit(main())
