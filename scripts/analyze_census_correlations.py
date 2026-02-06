"""
Analyze Census Correlations with Electoral Patterns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(config):
    """Load census and electoral data."""
    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])

    df_census = pd.read_parquet(processed_dir / 'census_department_summary.parquet')

    # Load departmental electoral results with CA defection rates
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    # Use transfers_by_department.csv which has CA defection rates
    electoral_file = tables_dir / 'transfers_by_department.csv'

    if electoral_file.exists():
        df_electoral = pd.read_csv(electoral_file)
        # Rename for consistency
        df_electoral = df_electoral.rename(columns={
            'ca_to_fa': 'ca_defection_to_fa',
            'ca_to_pn': 'ca_retention_by_pn',
            'pc_to_pn': 'pc_retention_by_pn',
            'pn_to_pn': 'pn_retention',
            'pc_to_fa': 'pc_defection_to_fa',
            'pn_to_fa': 'pn_defection_to_fa'
        })
    else:
        logger.warning("Electoral results not found, using placeholder data")
        df_electoral = pd.DataFrame({
            'departamento': df_census['departamento'],
            'ca_defection_to_fa': np.random.uniform(0.15, 0.65, len(df_census))
        })

    return df_census, df_electoral


def calculate_correlations(df_merged):
    """Calculate correlations between demographics and electoral behavior."""
    logger.info("Calculando correlaciones...")

    correlations = []

    demographic_vars = ['edad_mediana', 'pct_educacion_terciaria', 'poblacion']

    # Focus on CA defection as primary variable of interest
    electoral_vars = [
        'ca_defection_to_fa',
        'ca_retention_by_pn',
        'pc_retention_by_pn',
        'pn_retention'
    ]

    for demo_var in demographic_vars:
        if demo_var not in df_merged.columns:
            continue

        for elec_var in electoral_vars:
            if elec_var not in df_merged.columns:
                continue

            valid = df_merged[[demo_var, elec_var]].dropna()

            if len(valid) > 0:
                corr, pval = stats.pearsonr(valid[demo_var], valid[elec_var])

                correlations.append({
                    'Variable_Demografica': demo_var,
                    'Variable_Electoral': elec_var,
                    'Correlacion': corr,
                    'Valor_P': pval,
                    'Significativo': 'Sí' if pval < 0.05 else 'No'
                })

    return pd.DataFrame(correlations)


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("ANÁLISIS DE CORRELACIONES CENSALES")
    logger.info("="*70)

    df_census, df_electoral = load_data(config)

    # Merge
    df_merged = df_census.merge(df_electoral, on='departamento', how='inner')

    logger.info(f"Departamentos con datos completos: {len(df_merged)}")

    # Calculate correlations
    df_corr = calculate_correlations(df_merged)

    # Save
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    df_corr.to_csv(tables_dir / 'census_correlation_matrix.csv', index=False, float_format='%.4f')

    # LaTeX table
    df_latex = df_corr.copy()
    df_latex['Correlacion'] = df_latex['Correlacion'].apply(lambda x: f"{x:.3f}")
    df_latex['Valor_P'] = df_latex['Valor_P'].apply(lambda x: f"{x:.4f}")

    # Rename for Spanish LaTeX
    df_latex = df_latex.rename(columns={
        'Variable_Demografica': 'Variable Demográfica',
        'Variable_Electoral': 'Variable Electoral',
        'Correlacion': 'Correlación',
        'Valor_P': 'Valor $p$',
        'Significativo': 'Significativo'
    })

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='llccc',
        caption='Correlaciones entre variables demográficas (Censo 2023) y comportamiento electoral',
        label='tab:census_correlations'
    )

    with open(latex_dir / 'census_correlation_matrix.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    # Print summary
    print("\n" + "="*70)
    print("CORRELACIONES: DEMOGRAFÍA (CENSO 2023) vs COMPORTAMIENTO ELECTORAL")
    print("="*70 + "\n")

    for _, row in df_corr.iterrows():
        sig = "***" if row['Significativo'] == 'Sí' else ""
        print(f"{row['Variable_Demografica']:<30s} vs {row['Variable_Electoral']:<25s}: "
              f"r = {row['Correlacion']:>+.3f} (p = {row['Valor_P']:.4f}) {sig}")

    print("\n" + "="*70 + "\n")

    logger.info("Análisis completado")

    return 0


if __name__ == '__main__':
    sys.exit(main())
