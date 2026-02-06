"""
Generate Sensitivity Analysis Figures
Creates visualizations for robustness tests
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.visualization.styles import (
    setup_professional_style,
    apply_tableau_style,
    save_publication_figure,
    PARTY_COLORS
)

logger = get_logger(__name__)
setup_professional_style()


def load_results(config):
    """Load sensitivity analysis results."""
    logger.info("Cargando resultados de análisis de sensibilidad...")

    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])

    # Construct results_dir manually
    results_dir = tables_dir.parent / 'results'

    # Load comparison table
    comparison_path = tables_dir / 'sensitivity_comparison.csv'
    if not comparison_path.exists():
        logger.error(f"Archivo no encontrado: {comparison_path}")
        logger.error("Ejecutar scripts/sensitivity_analysis.py primero")
        return None, None

    df_comparison = pd.read_csv(comparison_path)
    logger.info(f"Comparaciones cargadas: {len(df_comparison)} tests")

    # Load baseline trace (for posterior plots)
    baseline_path = results_dir / 'sensitivity_baseline.pkl'
    baseline = None
    if baseline_path.exists():
        with open(baseline_path, 'rb') as f:
            baseline = pickle.load(f)
        logger.info("Baseline trace cargado")

    # Load bootstrap results if available
    bootstrap_path = results_dir / 'sensitivity_bootstrap.pkl'
    bootstrap = None
    if bootstrap_path.exists():
        with open(bootstrap_path, 'rb') as f:
            bootstrap = pickle.load(f)
        logger.info("Bootstrap results cargado")

    return df_comparison, {'baseline': baseline, 'bootstrap': bootstrap}


def figure_1_comparison_bar(df_comparison, config):
    """Figure 1: Bar chart comparing CA→FA across all tests."""
    logger.info("Generando Figura 1: Comparación tests (CA→FA)...")

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get baseline value for reference
    baseline_mask = df_comparison['Test'] == 'Baseline'
    if baseline_mask.any():
        baseline_value = df_comparison.loc[baseline_mask, 'CA→FA_Mean'].values[0]
    else:
        baseline_value = df_comparison['CA→FA_Mean'].mean()

    # Prepare data
    tests = df_comparison['Test'].values
    ca_fa_means = df_comparison['CA→FA_Mean'].values * 100

    # Color bars by difference from baseline
    colors = []
    for val in ca_fa_means:
        if abs(val - baseline_value * 100) < 1:  # Within 1 pp
            colors.append('#27AE60')  # Green - stable
        elif abs(val - baseline_value * 100) < 3:  # Within 3 pp
            colors.append('#F39C12')  # Orange - moderate
        else:
            colors.append('#E74C3C')  # Red - different

    bars = ax.bar(range(len(tests)), ca_fa_means, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)

    # Baseline reference line
    ax.axhline(baseline_value * 100, color='#2C3E50', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Baseline: {baseline_value:.1%}')

    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, rotation=45, ha='right', fontsize=9)

    # Value labels
    for bar, val in zip(bars, ca_fa_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='normal')

    apply_tableau_style(ax,
                       title='Análisis de Sensibilidad: CA→FA por Test',
                       xlabel='Test de Robustez',
                       ylabel='CA→FA (%)')

    ax.legend(loc='best', frameon=True, fontsize=10)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'sensitivity_comparison_bar')
    plt.close()
    logger.info("Figura 1 guardada: sensitivity_comparison_bar")


def figure_2_bootstrap_histogram(traces, config):
    """Figure 2: Bootstrap distribution histogram."""
    logger.info("Generando Figura 2: Bootstrap distribution...")

    if traces['bootstrap'] is None:
        logger.warning("Bootstrap results no disponibles - saltando figura 2")
        return

    bootstrap = traces['bootstrap']
    if 'results' not in bootstrap or len(bootstrap['results']) == 0:
        logger.warning("Bootstrap results vacíos - saltando figura 2")
        return

    df_boot = bootstrap['results']

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: CA→FA distribution
    ax = axes[0]
    ca_fa_vals = df_boot['ca_to_fa'].values * 100

    ax.hist(ca_fa_vals, bins=15, color=PARTY_COLORS['CA'], alpha=0.7,
            edgecolor='white', linewidth=1.5)

    mean_val = ca_fa_vals.mean()
    std_val = ca_fa_vals.std()

    ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=2,
              label=f'Media: {mean_val:.1f}% ± {std_val:.1f}%')

    apply_tableau_style(ax,
                       title='Bootstrap Distribution: CA→FA',
                       xlabel='CA→FA (%)',
                       ylabel='Frecuencia')

    ax.legend(loc='best', frameon=True, fontsize=10)

    # Panel B: CA→PN distribution
    ax = axes[1]
    ca_pn_vals = df_boot['ca_to_pn'].values * 100

    ax.hist(ca_pn_vals, bins=15, color=PARTY_COLORS['PN'], alpha=0.7,
            edgecolor='white', linewidth=1.5)

    mean_val = ca_pn_vals.mean()
    std_val = ca_pn_vals.std()

    ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=2,
              label=f'Media: {mean_val:.1f}% ± {std_val:.1f}%')

    apply_tableau_style(ax,
                       title='Bootstrap Distribution: CA→PN',
                       xlabel='CA→PN (%)',
                       ylabel='Frecuencia')

    ax.legend(loc='best', frameon=True, fontsize=10)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'sensitivity_bootstrap_distribution')
    plt.close()
    logger.info("Figura 2 guardada: sensitivity_bootstrap_distribution")


def figure_3_differences_scatter(df_comparison, config):
    """Figure 3: Scatter plot of differences from baseline."""
    logger.info("Generando Figura 3: Diferencias vs baseline...")

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])

    # Remove baseline itself from comparison
    df_plot = df_comparison[df_comparison['Test'] != 'Baseline'].copy()

    if len(df_plot) == 0:
        logger.warning("No hay tests para comparar - saltando figura 3")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # CA→FA differences
    x = df_plot['CA→FA_Diff'].values * 100
    y = df_plot['CA→PN_Diff'].values * 100

    # Color by test type
    colors = []
    for test in df_plot['Test'].values:
        if 'Bootstrap' in test:
            colors.append(PARTY_COLORS['CA'])
        elif 'MCMC' in test:
            colors.append(PARTY_COLORS['PN'])
        elif 'Outliers' in test or 'Small' in test:
            colors.append(PARTY_COLORS['PC'])
        else:
            colors.append('#95A5A6')

    ax.scatter(x, y, s=150, alpha=0.7, c=colors, edgecolors='white', linewidth=2)

    # Labels
    for i, test in enumerate(df_plot['Test'].values):
        ax.annotate(test, (x[i], y[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    # Reference lines
    ax.axhline(0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)

    # ±5% tolerance box
    tolerance = 5
    ax.axhspan(-tolerance, tolerance, alpha=0.1, color='green', label='±5pp tolerancia')
    ax.axvspan(-tolerance, tolerance, alpha=0.1, color='green')

    apply_tableau_style(ax,
                       title='Diferencias vs Baseline: CA Transfers',
                       xlabel='Δ CA→FA (pp)',
                       ylabel='Δ CA→PN (pp)')

    ax.legend(loc='best', frameon=True, fontsize=10)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'sensitivity_differences_scatter')
    plt.close()
    logger.info("Figura 3 guardada: sensitivity_differences_scatter")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS DE SENSIBILIDAD")
    logger.info("="*70)

    # Load results
    df_comparison, traces = load_results(config)

    if df_comparison is None:
        logger.error("No se pudieron cargar los resultados")
        logger.error("Ejecutar scripts/sensitivity_analysis.py primero")
        return 1

    # Generate figures
    figure_1_comparison_bar(df_comparison, config)
    figure_2_bootstrap_histogram(traces, config)
    figure_3_differences_scatter(df_comparison, config)

    logger.info("="*70)
    logger.info("FIGURAS DE SENSIBILIDAD GENERADAS")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
