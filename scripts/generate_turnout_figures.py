"""
Generate turnout dynamics visualizations.
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_results(config):
    """Load turnout analysis results."""
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    df_dept = pd.read_csv(tables_dir / 'turnout_by_department.csv')
    df_party = pd.read_csv(tables_dir / 'turnout_by_dominant_party.csv')
    df_corr = pd.read_csv(tables_dir / 'turnout_correlation_matrix.csv')

    return df_dept, df_party, df_corr


def load_circuit_data(config):
    """Load circuit-level data for scatter plots."""
    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])

    data_path_cov = processed_dir / 'circuitos_full_covariates.parquet'
    data_path = processed_dir / 'circuitos_merged.parquet'

    if data_path_cov.exists():
        return pd.read_parquet(data_path_cov)
    else:
        return pd.read_parquet(data_path)


def figure1_turnout_by_dominant_party(df_party, output_dir):
    """Bar chart turnout change by dominant party."""
    logger.info("Generando figura 1: Turnout por partido dominante...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    parties = df_party['dominant_party'].values
    turnout_change = df_party['turnout_change_pp'].values

    colors = {'FA': '#9B59B6', 'PN': '#2ECC71', 'PC': '#E74C3C', 'CA': '#F39C12'}
    bar_colors = [colors.get(p, '#95A5A6') for p in parties]

    ax.bar(parties, turnout_change, color=bar_colors, alpha=0.8, edgecolor='black')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Partido Dominante (Primera Vuelta)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cambio en Participación (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Cambio de Participación por Partido Dominante\nPrimera Vuelta → Ballotage 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'turnout_by_dominant_party.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'turnout_by_dominant_party.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 1 guardada")


def figure2_scatter_turnout_vs_fa_swing(df, output_dir):
    """Scatter plot: turnout change vs FA swing."""
    logger.info("Generando figura 2: Scatter turnout vs FA swing...")

    # Calculate metrics
    df = df.copy()
    df['turnout_primera'] = df['total_primera'] / df.get('habilitados', df['total_primera'] * 1.2)
    df['turnout_ballotage'] = df['total_ballotage'] / df.get('habilitados', df['total_ballotage'] * 1.2)
    df['participacion_change_pp'] = (df['turnout_ballotage'] - df['turnout_primera']) * 100

    df['fa_share_primera'] = df['fa_primera'] / df['total_primera']
    df['fa_share_ballotage'] = df['fa_ballotage'] / df['total_ballotage']
    df['fa_swing'] = df['fa_share_ballotage'] - df['fa_share_primera']

    # Remove outliers
    df_clean = df[(df['participacion_change_pp'] > -10) & (df['participacion_change_pp'] < 20)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter
    ax.scatter(df_clean['participacion_change_pp'], df_clean['fa_swing'] * 100,
               alpha=0.3, s=20, color='#9B59B6', edgecolors='none')

    # Regression line
    from scipy import stats
    valid = df_clean[['participacion_change_pp', 'fa_swing']].dropna()
    if len(valid) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid['participacion_change_pp'], valid['fa_swing'] * 100
        )
        x_line = np.array([valid['participacion_change_pp'].min(), valid['participacion_change_pp'].max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'r = {r_value:.3f} (p < 0.001)')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Cambio en Participación (pp)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Swing FA (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Relación entre Cambio de Participación y Swing de FA\nBallotage 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'scatter_turnout_vs_fa_swing.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_turnout_vs_fa_swing.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 2 guardada")


def figure3_map_turnout_change(df_dept, output_dir):
    """Map: turnout change by department (simplified bar chart if no shapefiles)."""
    logger.info("Generando figura 3: Mapa cambio participación...")

    # For now, create horizontal bar chart (map requires shapefiles)
    fig, ax = plt.subplots(figsize=(10, 10))

    df_sorted = df_dept.sort_values('turnout_change_pp')

    colors = ['#d73027' if x < 0 else '#4575b4' for x in df_sorted['turnout_change_pp']]

    ax.barh(range(len(df_sorted)), df_sorted['turnout_change_pp'], color=colors, alpha=0.8, edgecolor='black')

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['departamento'], fontsize=9)

    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Cambio en Participación (pp)', fontsize=12, fontweight='bold')
    ax.set_title('Cambio de Participación por Departamento\nPrimera Vuelta → Ballotage 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / 'mapa_turnout_change.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mapa_turnout_change.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 3 guardada")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS - DINÁMICA DE PARTICIPACIÓN")
    logger.info("="*70)

    # Load results
    df_dept, df_party, df_corr = load_results(config)
    df_circuits = load_circuit_data(config)

    # Output directory
    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    figure1_turnout_by_dominant_party(df_party, figures_dir)
    figure2_scatter_turnout_vs_fa_swing(df_circuits, figures_dir)
    figure3_map_turnout_change(df_dept, figures_dir)

    logger.info(f"\n3 figuras guardadas en: {figures_dir}")
    logger.info("Generación de figuras completada")

    return 0


if __name__ == '__main__':
    sys.exit(main())
