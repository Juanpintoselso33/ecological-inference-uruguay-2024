"""
Generate Goodman vs King's EI comparison figures.
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
plt.rcParams['font.size'] = 10


def load_results(config):
    """Load comparison results."""
    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    df_comparison = pd.read_csv(tables_dir / 'goodman_vs_king_comparison.csv')
    return df_comparison


def figure1_scatter_goodman_vs_king(df, output_dir):
    """Scatter plot: Goodman vs King estimates."""
    logger.info("Generando figura 1: Scatter Goodman vs King...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot points
    ax.scatter(df['Goodman_Mean'], df['King_Mean'], s=100, alpha=0.7,
               c='#3498DB', edgecolors='black', linewidth=1, zorder=3)

    # Add labels for key transitions
    key_transitions = ['CA→FA', 'CA→PN', 'PC→PN', 'PN→PN', 'FA→FA']
    for _, row in df.iterrows():
        if row['Transition'] in key_transitions:
            ax.annotate(row['Transition'],
                       (row['Goodman_Mean'], row['King_Mean']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')

    # Diagonal line (perfect agreement)
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement', zorder=1)

    # +/- 10% bands
    ax.fill_between(lims, [x - 0.1 for x in lims], [x + 0.1 for x in lims],
                    color='gray', alpha=0.2, label='±10% Band', zorder=0)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Goodman Regression (OLS)', fontsize=12, fontweight='bold')
    ax.set_ylabel("King's Ecological Inference (Bayesian)", fontsize=12, fontweight='bold')
    ax.set_title('Comparación Metodológica: Goodman vs King\'s EI\nEstimaciones de Transferencias de Votos 2024',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()

    plt.savefig(output_dir / 'scatter_goodman_vs_king.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_goodman_vs_king.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 1 guardada")


def figure2_heatmap_method_differences(df, output_dir):
    """Heatmap: differences between methods."""
    logger.info("Generando figura 2: Heatmap diferencias...")

    # Extract origin and destination
    df = df.copy()
    df['Origin'] = df['Origin']
    df['Destination'] = df['Destination']

    # Create pivot table
    pivot = df.pivot(index='Origin', columns='Destination', values='Difference')

    # Reorder
    origin_order = ['CA', 'PC', 'PI', 'PN', 'FA']
    dest_order = ['FA', 'PN']

    pivot = pivot.reindex(index=origin_order, columns=dest_order)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdBu_r', center=0,
                vmin=-0.15, vmax=0.15, cbar_kws={'label': 'Diferencia (King - Goodman)'},
                linewidths=0.5, linecolor='black', ax=ax)

    ax.set_xlabel('Destino (Ballotage)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Origen (Primera Vuelta)', fontsize=12, fontweight='bold')
    ax.set_title('Diferencias entre King\'s EI y Goodman Regression\nMagnitud = King - Goodman',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    plt.savefig(output_dir / 'heatmap_method_differences.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'heatmap_method_differences.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 2 guardada")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS - COMPARACIÓN GOODMAN vs KING")
    logger.info("="*70)

    # Load results
    df = load_results(config)

    # Output directory
    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    figure1_scatter_goodman_vs_king(df, figures_dir)
    figure2_heatmap_method_differences(df, figures_dir)

    logger.info(f"\n2 figuras guardadas en: {figures_dir}")
    logger.info("Generación de figuras completada")

    return 0


if __name__ == '__main__':
    sys.exit(main())
