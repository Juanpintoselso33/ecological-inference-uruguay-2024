"""
Generate Blank Votes Analysis Figures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_results(config):
    """Load blank votes results."""
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    df_corr = pd.read_csv(tables_dir / 'blank_votes_correlation.csv')
    df_dept = pd.read_csv(tables_dir / 'blank_votes_by_department.csv')

    return df_corr, df_dept


def load_circuit_data(config):
    """Load circuit-level data."""
    data_dirs = config.get_data_dirs()
    return pd.read_parquet(Path(data_dirs['processed']) / 'circuitos_merged.parquet')


def figure1_scatter_blank_vs_ca(df, output_dir):
    """Scatter: Blank rate vs CA share."""
    logger.info("Generando figura 1: Scatter blank rate vs CA share...")

    df = df.copy()
    df['blank_rate'] = (df['blancos_ballotage'] + df['anulados_ballotage']) / df['total_ballotage']
    df['ca_share'] = df['ca_primera'] / df['total_primera']

    # Remove extreme outliers
    df_clean = df[(df['blank_rate'] < 0.15) & (df['ca_share'] < 0.3)]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df_clean['ca_share'] * 100, df_clean['blank_rate'] * 100,
               alpha=0.3, s=20, color='#E74C3C', edgecolors='none')

    # Regression line
    from scipy import stats
    valid = df_clean[['ca_share', 'blank_rate']].dropna()
    if len(valid) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid['ca_share'], valid['blank_rate']
        )
        x_line = np.array([valid['ca_share'].min(), valid['ca_share'].max()])
        y_line = (slope * x_line + intercept) * 100

        label = f'r = {r_value:.3f}, p = {p_value:.4f}'
        if p_value >= 0.05:
            label += ' (NO significativo)'

        ax.plot(x_line * 100, y_line, 'b-', linewidth=2, label=label)

    ax.set_xlabel('Share CA Primera Vuelta (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasa de Voto en Blanco/Anulado (%)', fontsize=12, fontweight='bold')
    ax.set_title('Relación entre Voto en Blanco y Share de Cabildo Abierto\nBallotage 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'scatter_blank_rate_vs_ca_share.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_blank_rate_vs_ca_share.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 1 guardada")


def figure2_blank_rate_ranking(df_dept, output_dir):
    """Bar chart: Blank rate by department."""
    logger.info("Generando figura 2: Ranking blank rate por departamento...")

    if 'blank_rate_calc' in df_dept.columns:
        df_sorted = df_dept.sort_values('blank_rate_calc', ascending=False)
        blank_col = 'blank_rate_calc'
    else:
        df_sorted = df_dept.sort_values('blank_rate', ascending=False)
        blank_col = 'blank_rate'

    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['#E74C3C' if x > df_sorted[blank_col].median() else '#3498DB'
              for x in df_sorted[blank_col]]

    ax.barh(range(len(df_sorted)), df_sorted[blank_col] * 100,
            color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['departamento'], fontsize=9)

    ax.axvline(df_sorted[blank_col].median() * 100, color='black',
               linestyle='--', linewidth=2, alpha=0.5, label='Mediana Nacional')

    ax.set_xlabel('Tasa de Voto en Blanco/Anulado (%)', fontsize=12, fontweight='bold')
    ax.set_title('Tasa de Voto en Blanco por Departamento\nBallotage 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'blank_rate_ranking.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'blank_rate_ranking.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 2 guardada")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS - VOTOS EN BLANCO")
    logger.info("="*70)

    df_corr, df_dept = load_results(config)
    df_circuits = load_circuit_data(config)

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure1_scatter_blank_vs_ca(df_circuits, figures_dir)
    figure2_blank_rate_ranking(df_dept, figures_dir)

    logger.info(f"\n2 figuras guardadas en: {figures_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
