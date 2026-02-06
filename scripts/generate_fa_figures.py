"""
Generación de Figuras: Análisis FA (Frente Amplio)
Crea 9 figuras @ 300 DPI mostrando comportamiento electoral FA 2019-2024
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from src.visualization.styles import (
    setup_professional_style,
    apply_tableau_style,
    get_party_color,
    save_publication_figure,
    PARTY_COLORS,
    format_percentage_axis,
    add_value_labels,
    apply_minimal_grid
)
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Setup styles
setup_professional_style()

def load_results(config):
    """Load all FA analysis results."""
    results_dir = Path(config.project_root_path) / 'outputs' / 'results'
    tables_dir = Path(config.project_root_path) / 'outputs' / 'tables'

    results = {}

    # National 2024
    try:
        results['national_2024'] = pd.read_csv(tables_dir / 'fa_national_matrix_2024.csv')
        logger.info("✓ Loaded FA national 2024")
    except FileNotFoundError:
        logger.warning("⚠ FA national 2024 not found - skipping related figures")
        results['national_2024'] = None

    # Departmental 2024
    try:
        results['departmental_2024'] = pd.read_csv(tables_dir / 'fa_departmental_summary_2024.csv')
        logger.info("✓ Loaded FA departmental 2024")
    except FileNotFoundError:
        logger.warning("⚠ FA departmental 2024 not found - skipping related figures")
        results['departmental_2024'] = None

    # Temporal comparison
    try:
        results['temporal'] = pd.read_csv(tables_dir / 'fa_temporal_comparison_2019_2024.csv')
        logger.info("✓ Loaded FA temporal comparison")
    except FileNotFoundError:
        logger.warning("⚠ FA temporal comparison not found - skipping related figures")
        results['temporal'] = None

    # Stratified
    try:
        results['stratified'] = pd.read_csv(tables_dir / 'fa_stratified_summary_2024.csv')
        if len(results['stratified']) == 0:
            logger.warning("FA stratified file empty - skipping related figures")
            results['stratified'] = None
        else:
            logger.info("Loaded FA stratified")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.warning("FA stratified not found or empty - skipping related figures")
        results['stratified'] = None

    return results


def figure_1_forest_plot_retention(results, output_dir):
    """
    Figure 1: Forest plot - FA retention by department
    Shows retention rate with 95% CI for all 19 departments
    """
    df = results['departmental_2024']
    if df is None:
        logger.warning("Skipping Figure 1: No departmental data")
        return

    logger.info("Creating Figure 1: Forest plot FA retention by department")

    # Sort by retention rate
    df_sorted = df.sort_values('fa_retention', ascending=True).copy()

    fig, ax = plt.subplots(figsize=(12, 10))

    departments = df_sorted['departamento'].values
    retention = df_sorted['fa_retention'].values * 100
    ci_lower = df_sorted['fa_retention_ci_lower'].values * 100
    ci_upper = df_sorted['fa_retention_ci_upper'].values * 100

    y_pos = np.arange(len(departments))

    # Plot points and error bars
    ax.errorbar(retention, y_pos,
                xerr=[retention - ci_lower, ci_upper - retention],
                fmt='o', markersize=8, capsize=4, capthick=2,
                color=PARTY_COLORS['FA'], ecolor=PARTY_COLORS['FA'],
                alpha=0.8, linewidth=2)

    # Add vertical line at national mean
    national_mean = results['national_2024']['Mean'].iloc[0] * 100 if results['national_2024'] is not None else None
    if national_mean:
        ax.axvline(national_mean, color='#7F8C8D', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'Media Nacional: {national_mean:.1f}%')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(departments, fontsize=10)

    apply_tableau_style(ax,
                       title='Retención Electoral del Frente Amplio por Departamento',
                       xlabel='Retención FA (%)',
                       ylabel='Departamento')

    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(20, 70)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'forest_plot_fa_retention_departments')
    plt.close()
    logger.info("✓ Figure 1 saved")


def figure_2_sankey_national(results, output_dir):
    """
    Figure 2: Sankey diagram - FA vote flow nacional 2024
    Shows FA primera → {FA, PN, Blancos} ballotage
    """
    if results['national_2024'] is None:
        logger.warning("Skipping Figure 2: No national data")
        return

    logger.info("Creating Figure 2: Sankey diagram FA national flow")

    # This requires plotly - simplified bar chart version
    df = results['national_2024']

    fig, ax = plt.subplots(figsize=(14, 8))

    destinations = ['FA Ballotage', 'PN Ballotage', 'Blancos']
    values = df['Mean'].values[:3] * 100  # Assuming first 3 rows are FA→FA, FA→PN, FA→Blancos
    colors = [PARTY_COLORS['FA'], PARTY_COLORS['PN'], '#BDC3C7']

    bars = ax.bar(destinations, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    apply_tableau_style(ax,
                       title='Flujo Electoral del Frente Amplio: Primera Vuelta → Ballotage 2024',
                       xlabel='Destino',
                       ylabel='Proporción (%)')

    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'sankey_fa_national_2024')
    plt.close()
    logger.info("✓ Figure 2 saved")


def figure_3_temporal_comparison(results, output_dir):
    """
    Figure 3: Temporal comparison 2019 vs 2024
    Grouped bar chart showing FA→FA, FA→PN changes
    """
    df = results['temporal']
    if df is None:
        logger.warning("Skipping Figure 3: No temporal data")
        return

    logger.info("Creating Figure 3: Temporal comparison 2019 vs 2024")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract values
    metrics = ['Retención FA', 'Defección FA→PN']
    values_2019 = [
        df[df['Metric'] == 'Fa Retention']['Value_2019'].values[0] * 100,
        df[df['Metric'] == 'Fa To Pn']['Value_2019'].values[0] * 100
    ]
    values_2024 = [
        df[df['Metric'] == 'Fa Retention']['Value_2024'].values[0] * 100,
        df[df['Metric'] == 'Fa To Pn']['Value_2024'].values[0] * 100
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, values_2019, width, label='2019',
                   color='#3498DB', alpha=0.8, edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, values_2024, width, label='2024',
                   color=PARTY_COLORS['FA'], alpha=0.8, edgecolor='white', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='normal')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)

    apply_tableau_style(ax,
                       title='Evolución del Comportamiento Electoral FA: 2019 vs 2024',
                       xlabel='Métrica',
                       ylabel='Porcentaje (%)')

    ax.legend(loc='upper right', frameon=True, fontsize=11)
    ax.set_ylim(0, max(max(values_2019), max(values_2024)) * 1.15)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_retention_comparison_2019_2024')
    plt.close()
    logger.info("✓ Figure 3 saved")


def figure_4_urban_rural(results, output_dir):
    """
    Figure 4: Urban vs Rural stratification
    """
    df = results['stratified']
    if df is None:
        logger.warning("Skipping Figure 4: No stratified data")
        return

    logger.info("Creating Figure 4: Urban vs Rural comparison")

    # Filter urban/rural
    df_ur = df[df['stratum'].str.contains('UrbanRural')].copy()

    if len(df_ur) == 0:
        logger.warning("No Urban/Rural data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    strata = df_ur['stratum'].str.replace('UrbanRural_', '').values
    retention = df_ur['fa_retention'].values * 100
    ci_lower = df_ur['fa_retention_ci_lower'].values * 100
    ci_upper = df_ur['fa_retention_ci_upper'].values * 100

    colors = ['#27AE60' if 'RURAL' in s else '#3498DB' for s in df_ur['stratum']]

    bars = ax.bar(strata, retention, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)

    # Error bars
    ax.errorbar(strata, retention,
                yerr=[retention - ci_lower, ci_upper - retention],
                fmt='none', capsize=8, capthick=2, color='#2C3E50', linewidth=2)

    # Value labels
    for bar, val in zip(bars, retention):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    apply_tableau_style(ax,
                       title='Retención FA: Circuitos Urbanos vs Rurales',
                       xlabel='Tipo de Circuito',
                       ylabel='Retención FA (%)')

    ax.set_ylim(0, max(retention) * 1.15)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_retention_urban_rural')
    plt.close()
    logger.info("✓ Figure 4 saved")


def figure_5_metro_interior(results, output_dir):
    """
    Figure 5: Metropolitano vs Interior stratification
    """
    df = results['stratified']
    if df is None:
        logger.warning("Skipping Figure 5: No stratified data")
        return

    logger.info("Creating Figure 5: Metro vs Interior comparison")

    # Filter metro/interior
    df_mi = df[df['stratum'].str.contains('MetroInterior')].copy()

    if len(df_mi) == 0:
        logger.warning("No Metro/Interior data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    strata = df_mi['stratum'].str.replace('MetroInterior_', '').values
    retention = df_mi['fa_retention'].values * 100
    ci_lower = df_mi['fa_retention_ci_lower'].values * 100
    ci_upper = df_mi['fa_retention_ci_upper'].values * 100

    colors = [PARTY_COLORS['FA'] if 'METRO' in s else '#E67E22' for s in df_mi['stratum']]

    bars = ax.bar(strata, retention, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)

    # Error bars
    ax.errorbar(strata, retention,
                yerr=[retention - ci_lower, ci_upper - retention],
                fmt='none', capsize=8, capthick=2, color='#2C3E50', linewidth=2)

    # Value labels
    for bar, val in zip(bars, retention):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    apply_tableau_style(ax,
                       title='Retención FA: Área Metropolitana vs Interior',
                       xlabel='Región',
                       ylabel='Retención FA (%)')

    ax.set_ylim(0, max(retention) * 1.15)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_retention_metro_interior')
    plt.close()
    logger.info("✓ Figure 5 saved")


def figure_6_ranking_top_bottom(results, output_dir):
    """
    Figure 6: Top 10 and Bottom 10 departments by FA retention
    """
    df = results['departmental_2024']
    if df is None:
        logger.warning("Skipping Figure 6: No departmental data")
        return

    logger.info("Creating Figure 6: Top/Bottom 10 departments")

    df_sorted = df.sort_values('fa_retention', ascending=False).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top 10
    top10 = df_sorted.head(10)
    ax1.barh(range(len(top10)), top10['fa_retention'].values * 100,
             color=PARTY_COLORS['FA'], alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_yticks(range(len(top10)))
    ax1.set_yticklabels(top10['departamento'].values, fontsize=10)
    ax1.invert_yaxis()

    apply_tableau_style(ax1,
                       title='Top 10: Mayor Retención FA',
                       xlabel='Retención FA (%)',
                       ylabel='')

    # Add value labels
    for i, (val, dept) in enumerate(zip(top10['fa_retention'].values * 100, top10['departamento'].values)):
        ax1.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)

    # Bottom 10
    bottom10 = df_sorted.tail(10).sort_values('fa_retention', ascending=True)
    ax2.barh(range(len(bottom10)), bottom10['fa_retention'].values * 100,
             color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_yticks(range(len(bottom10)))
    ax2.set_yticklabels(bottom10['departamento'].values, fontsize=10)
    ax2.invert_yaxis()

    apply_tableau_style(ax2,
                       title='Bottom 10: Menor Retención FA',
                       xlabel='Retención FA (%)',
                       ylabel='')

    # Add value labels
    for i, (val, dept) in enumerate(zip(bottom10['fa_retention'].values * 100, bottom10['departamento'].values)):
        ax2.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_retention_ranking')
    plt.close()
    logger.info("✓ Figure 6 saved")


def figure_7_heatmap_departments(results, output_dir):
    """
    Figure 7: Heatmap - FA transitions by department
    Rows = departments, Cols = destinations (FA, PN, Blancos)
    """
    df = results['departmental_2024']
    if df is None or 'fa_to_pn' not in df.columns:
        logger.warning("Skipping Figure 7: Insufficient departmental data")
        return

    logger.info("Creating Figure 7: Heatmap FA transitions by department")

    # Prepare data matrix
    departments = df['departamento'].values
    matrix = np.column_stack([
        df['fa_retention'].values * 100,
        df['fa_to_pn'].values * 100,
        df.get('fa_to_blancos', np.zeros(len(df))).values * 100
    ])

    fig, ax = plt.subplots(figsize=(10, 12))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['FA→FA', 'FA→PN', 'FA→Blancos'], fontsize=11)
    ax.set_yticks(range(len(departments)))
    ax.set_yticklabels(departments, fontsize=9)

    # Add text annotations
    for i in range(len(departments)):
        for j in range(3):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax, label='Porcentaje (%)')

    ax.set_title('Matriz de Transiciones FA por Departamento',
                 fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'heatmap_fa_transitions_departmental')
    plt.close()
    logger.info("✓ Figure 7 saved")


def figure_8_change_magnitude(results, output_dir):
    """
    Figure 8: Magnitude of temporal change 2019→2024 by department
    (Requires departmental temporal comparison - may not exist yet)
    """
    # This would require departmental-level temporal comparison
    # For now, show national-level change
    df_temp = results['temporal']
    if df_temp is None:
        logger.warning("Skipping Figure 8: No temporal data")
        return

    logger.info("Creating Figure 8: Temporal change magnitude")

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = df_temp['Metric'].values
    changes = df_temp['Change'].values * 100

    colors = ['#27AE60' if c > 0 else '#E74C3C' for c in changes]

    bars = ax.barh(metrics, changes, color=colors, alpha=0.8,
                   edgecolor='white', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, changes):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:+.1f}pp',
                ha='left' if val > 0 else 'right', va='center',
                fontsize=10, fontweight='bold')

    ax.axvline(0, color='#2C3E50', linestyle='-', linewidth=1.5)

    apply_tableau_style(ax,
                       title='Cambio en Comportamiento Electoral FA: 2019 → 2024',
                       xlabel='Cambio (puntos porcentuales)',
                       ylabel='')

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_temporal_change_magnitude')
    plt.close()
    logger.info("✓ Figure 8 saved")


def figure_9_posterior_distributions(results, output_dir):
    """
    Figure 9: Posterior distributions for key FA transitions (national)
    Shows full Bayesian posterior for FA→FA, FA→PN
    """
    config = get_config()
    results_dir = Path(config.project_root_path) / 'outputs' / 'results'

    # Try to load trace
    try:
        with open(results_dir / 'fa_national_transitions_2024.pkl', 'rb') as f:
            trace = pickle.load(f)
    except FileNotFoundError:
        logger.warning("Skipping Figure 9: No trace file found")
        return

    logger.info("Creating Figure 9: Posterior distributions")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Extract transition matrix samples
    T_samples = trace.posterior['transition_matrix'].values
    # Shape: (chains, draws, origin, dest)

    # FA→FA (retention)
    fa_to_fa = T_samples[:, :, 0, 0].flatten() * 100
    axes[0].hist(fa_to_fa, bins=50, color=PARTY_COLORS['FA'],
                 alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[0].axvline(np.mean(fa_to_fa), color='#2C3E50', linestyle='--',
                    linewidth=2, label=f'Media: {np.mean(fa_to_fa):.1f}%')
    axes[0].axvline(np.percentile(fa_to_fa, 2.5), color='#E74C3C',
                    linestyle=':', linewidth=1.5, label='IC 95%')
    axes[0].axvline(np.percentile(fa_to_fa, 97.5), color='#E74C3C',
                    linestyle=':', linewidth=1.5)

    apply_tableau_style(axes[0],
                       title='Distribución Posterior: Retención FA (FA→FA)',
                       xlabel='Retención (%)',
                       ylabel='Densidad')
    axes[0].legend(loc='upper right')

    # FA→PN (defection)
    fa_to_pn = T_samples[:, :, 0, 1].flatten() * 100
    axes[1].hist(fa_to_pn, bins=50, color=PARTY_COLORS['PN'],
                 alpha=0.7, edgecolor='white', linewidth=0.5)
    axes[1].axvline(np.mean(fa_to_pn), color='#2C3E50', linestyle='--',
                    linewidth=2, label=f'Media: {np.mean(fa_to_pn):.1f}%')
    axes[1].axvline(np.percentile(fa_to_pn, 2.5), color='#E74C3C',
                    linestyle=':', linewidth=1.5, label='IC 95%')
    axes[1].axvline(np.percentile(fa_to_pn, 97.5), color='#E74C3C',
                    linestyle=':', linewidth=1.5)

    apply_tableau_style(axes[1],
                       title='Distribución Posterior: Defección FA (FA→PN)',
                       xlabel='Defección (%)',
                       ylabel='Densidad')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    save_publication_figure(fig, output_dir / 'fa_posterior_distributions')
    plt.close()
    logger.info("✓ Figure 9 saved")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERACIÓN DE FIGURAS FA")
    logger.info("="*70)

    # Load results
    results = load_results(config)

    # Output directory
    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    figure_1_forest_plot_retention(results, figures_dir)
    figure_2_sankey_national(results, figures_dir)
    figure_3_temporal_comparison(results, figures_dir)
    figure_4_urban_rural(results, figures_dir)
    figure_5_metro_interior(results, figures_dir)
    figure_6_ranking_top_bottom(results, figures_dir)
    figure_7_heatmap_departments(results, figures_dir)
    figure_8_change_magnitude(results, figures_dir)
    figure_9_posterior_distributions(results, figures_dir)

    logger.info("="*70)
    logger.info("GENERACIÓN DE FIGURAS FA COMPLETADA")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
