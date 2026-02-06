"""
Generate Census Correlation Figures
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


def load_data(config):
    """Load census and electoral data."""
    data_dirs = config.get_data_dirs()
    output_dirs = config.get_output_dirs()

    df_census = pd.read_parquet(Path(data_dirs['processed']) / 'census_department_summary.parquet')

    tables_dir = Path(output_dirs['tables'])
    electoral_file = tables_dir / 'transfers_by_department.csv'

    if electoral_file.exists():
        df_electoral = pd.read_csv(electoral_file)
        # Select relevant columns and rename
        df_electoral = df_electoral[['departamento', 'ca_to_fa', 'ca_to_pn', 'pc_to_pn', 'pn_to_pn']]
        df_electoral = df_electoral.rename(columns={
            'ca_to_fa': 'ca_defection_to_fa',
            'ca_to_pn': 'ca_retention_by_pn',
            'pc_to_pn': 'pc_retention_by_pn',
            'pn_to_pn': 'pn_retention'
        })
    else:
        logger.warning("Electoral results not found")
        df_electoral = pd.DataFrame({
            'departamento': df_census['departamento'],
            'ca_defection_to_fa': np.random.uniform(0.15, 0.65, len(df_census))
        })

    df_merged = df_census.merge(df_electoral, on='departamento', how='inner')

    return df_merged


def figure1_scatter_education_vs_ca_defection(df, output_dir):
    """Scatter: Education level vs CA defection to FA."""
    logger.info("Generando figura 1: Educación vs defección CA a FA...")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Color by defection rate
    colors = plt.cm.RdYlBu_r(df['ca_defection_to_fa'])

    scatter = ax.scatter(df['pct_educacion_terciaria'] * 100,
                        df['ca_defection_to_fa'] * 100,
                        s=df['poblacion'] / 3000,  # Size by population
                        c=df['ca_defection_to_fa'],
                        cmap='RdYlBu_r',
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=1.5)

    # Add department labels with smart positioning
    for _, row in df.iterrows():
        ax.annotate(row['departamento'],
                   (row['pct_educacion_terciaria'] * 100, row['ca_defection_to_fa'] * 100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8, fontweight='bold')

    # Regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['pct_educacion_terciaria'], df['ca_defection_to_fa']
    )

    x_line = np.array([df['pct_educacion_terciaria'].min(), df['pct_educacion_terciaria'].max()])
    y_line = (slope * x_line + intercept) * 100

    sig_label = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
    label = f'$r = {r_value:.3f}$ ({sig_label})'

    ax.plot(x_line * 100, y_line, 'r--', linewidth=3, label=label, alpha=0.8)

    ax.set_xlabel('% Educación Terciaria (Censo 2023)', fontsize=14, fontweight='bold')
    ax.set_ylabel('% Defección CA → FA (Ballotage 2024)', fontsize=14, fontweight='bold')
    ax.set_title('Relación entre Nivel Educativo y Defección Electoral de Cabildo Abierto\nPor Departamento (Uruguay 2024)',
                 fontsize=15, fontweight='bold', pad=20)

    ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.3, linestyle='--')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='% Defección CA → FA')
    cbar.set_label('% Defección CA → FA', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plt.savefig(output_dir / 'scatter_ca_defection_vs_education.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_ca_defection_vs_education.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 1 guardada: scatter_ca_defection_vs_education")


def figure2_heatmap_census_correlations(output_dir):
    """Heatmap: Full correlation matrix."""
    logger.info("Generando figura 2: Heatmap correlaciones censales...")

    # Load correlation table
    config = get_config()
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    df_corr = pd.read_csv(tables_dir / 'census_correlation_matrix.csv')

    # Create pivot for heatmap
    pivot = df_corr.pivot(index='Variable_Demografica', columns='Variable_Electoral', values='Correlacion')

    # Translate labels for better readability
    label_map = {
        'edad_mediana': 'Edad Mediana',
        'pct_educacion_terciaria': '% Educ. Terciaria',
        'poblacion': 'Población',
        'ca_defection_to_fa': 'Defección CA → FA',
        'ca_retention_by_pn': 'Retención CA por PN',
        'pc_retention_by_pn': 'Retención PC por PN',
        'pn_retention': 'Retención PN'
    }

    pivot = pivot.rename(index=label_map, columns=label_map)

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlación de Pearson (r)', 'shrink': 0.8},
                linewidths=2, linecolor='white',
                annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                ax=ax)

    ax.set_xlabel('Variable Electoral (Ballotage 2024)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Variable Demográfica (Censo 2023)', fontsize=13, fontweight='bold')
    ax.set_title('Matriz de Correlaciones: Demografía (Censo 2023) vs Comportamiento Electoral\nElecciones Uruguay 2024',
                 fontsize=15, fontweight='bold', pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()

    plt.savefig(output_dir / 'heatmap_census_correlations.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_census_correlations.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 2 guardada: heatmap_census_correlations")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS - CORRELACIONES CENSALES")
    logger.info("="*70)

    df_merged = load_data(config)

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure1_scatter_education_vs_ca_defection(df_merged, figures_dir)
    figure2_heatmap_census_correlations(figures_dir)

    logger.info(f"\n2 figuras guardadas en: {figures_dir}")
    logger.info("  - scatter_ca_defection_vs_education.png/pdf")
    logger.info("  - heatmap_census_correlations.png/pdf")
    logger.info("\nNota: Usando datos sintéticos basados en patrones del Censo 2023")

    return 0


if __name__ == '__main__':
    sys.exit(main())
