"""
Comparación Metodológica: Goodman Regression vs King's Ecological Inference
Valida resultados comparando baseline OLS con inferencia Bayesiana
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.models.goodman_regression import GoodmanRegression
from src.models.king_ei import KingEI
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


def load_and_prepare_data(config):
    """Load and prepare data for comparison."""
    logger.info("Cargando datos...")

    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])
    data_path = processed_dir / 'circuitos_merged.parquet'

    df = pd.read_parquet(data_path)
    logger.info(f"Datos cargados: {len(df)} circuitos")

    # Prepare for CA analysis (most interesting comparison)
    df = df.copy()
    df['blancos_ballotage'] = df.get('blancos_ballotage', 0) + df.get('anulados_ballotage', 0)

    # Use CA + others as origins
    origin_cols = ['ca_primera', 'otros_ca_recalc']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Recalculate others as complement of CA
    df['otros_ca_recalc'] = df['total_primera'] - df['ca_primera']

    # Filter valid circuits (minimum 30 votes)
    min_votes = 30
    valid = (df['total_primera'] >= min_votes) & (df['total_ballotage'] >= min_votes)
    df_clean = df[valid].copy()

    logger.info(f"Circuitos validos: {len(df_clean)} ({100*len(df_clean)/len(df):.1f}%)")

    return df_clean, origin_cols, destination_cols


def fit_goodman(df, origin_cols, destination_cols):
    """Fit Goodman Regression (OLS baseline)."""
    logger.info("Ajustando Goodman Regression (OLS)...")

    model = GoodmanRegression(add_constant=True)

    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage'
    )

    T = model.get_transition_matrix()
    uncertainty = model.get_uncertainty()

    logger.info("Goodman completado")

    return model, T, uncertainty


def fit_king(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Fit King's EI (Bayesian)."""
    logger.info(f"Ajustando King's EI (Bayesian: {num_samples} samples, {num_chains} chains)...")

    model = KingEI(
        num_samples=num_samples,
        num_chains=num_chains,
        num_warmup=num_samples // 2,
        random_seed=42
    )

    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(prob=0.95)
    diag = model.get_diagnostics()

    logger.info("King's EI completado")

    return model, T, ci, diag


def compare_results(T_goodman, T_king, ci_king, uncertainty_goodman, origin_cols, destination_cols):
    """Compare results from both methods."""
    logger.info("Comparando resultados...")

    results = []

    for i, origin in enumerate(origin_cols):
        for j, dest in enumerate(destination_cols):
            origin_name = origin.replace('_primera', '').replace('_recalc', '').upper()
            dest_name = dest.replace('_ballotage', '').upper()

            transition = f"{origin_name}→{dest_name}"

            results.append({
                'Transition': transition,
                'Origin': origin_name,
                'Destination': dest_name,
                'Goodman_Mean': T_goodman[i, j],
                'Goodman_SE': uncertainty_goodman['se'][i, j] if 'se' in uncertainty_goodman else np.nan,
                'King_Mean': T_king[i, j],
                'King_CI_Lower': ci_king['lower'][i, j],
                'King_CI_Upper': ci_king['upper'][i, j],
                'Difference': T_king[i, j] - T_goodman[i, j],
                'Abs_Difference': abs(T_king[i, j] - T_goodman[i, j])
            })

    df_comparison = pd.DataFrame(results)

    # Summary statistics
    summary = {
        'Mean_Abs_Difference': df_comparison['Abs_Difference'].mean(),
        'Max_Abs_Difference': df_comparison['Abs_Difference'].max(),
        'RMSE': np.sqrt((df_comparison['Difference']**2).mean()),
        'Correlation': np.corrcoef(
            df_comparison['Goodman_Mean'],
            df_comparison['King_Mean']
        )[0, 1]
    }

    logger.info(f"Diferencia absoluta media: {summary['Mean_Abs_Difference']:.4f}")
    logger.info(f"RMSE: {summary['RMSE']:.4f}")
    logger.info(f"Correlacion: {summary['Correlation']:.4f}")

    return df_comparison, summary


def save_results(df_comparison, summary, goodman_model, king_model, config):
    """Save comparison results."""
    logger.info("Guardando resultados...")

    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save comparison table (CSV)
    csv_path = tables_dir / 'goodman_vs_king_comparison.csv'
    df_comparison.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Tabla CSV guardada: {csv_path}")

    # 2. Save comparison table (LaTeX)
    df_latex = df_comparison[['Transition', 'Goodman_Mean', 'King_Mean', 'Difference']].copy()
    df_latex.columns = ['Transicion', 'Goodman (OLS)', 'King (Bayesian)', 'Diferencia']
    df_latex['Goodman (OLS)'] = df_latex['Goodman (OLS)'].apply(lambda x: f"{x:.1%}")
    df_latex['King (Bayesian)'] = df_latex['King (Bayesian)'].apply(lambda x: f"{x:.1%}")
    df_latex['Diferencia'] = df_latex['Diferencia'].apply(lambda x: f"{x:+.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lccc',
        caption='Comparacion metodologica: Goodman vs King',
        label='tab:goodman_vs_king'
    )

    latex_path = latex_dir / 'goodman_vs_king_comparison.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    logger.info(f"Tabla LaTeX guardada: {latex_path}")

    # 3. Save summary statistics
    summary_path = tables_dir / 'goodman_vs_king_summary.csv'
    pd.DataFrame([summary]).to_csv(summary_path, index=False, float_format='%.4f')
    logger.info(f"Resumen guardado: {summary_path}")


def generate_figures(df_comparison, config):
    """Generate comparison figures."""
    logger.info("Generando figuras...")

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Scatter plot Goodman vs King
    fig, ax = plt.subplots(figsize=(10, 10))

    x = df_comparison['Goodman_Mean'].values * 100
    y = df_comparison['King_Mean'].values * 100

    # Scatter
    ax.scatter(x, y, s=100, alpha=0.7, color=PARTY_COLORS['CA'], edgecolors='white', linewidth=2)

    # Diagonal y=x
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (acuerdo perfecto)')

    # Labels
    for i, row in df_comparison.iterrows():
        ax.annotate(row['Transition'], (x[i], y[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    apply_tableau_style(ax,
                       title='Comparacion Metodologica: Goodman (OLS) vs King (Bayesian)',
                       xlabel='Goodman Regression (%)',
                       ylabel='King\'s EI (%)')

    ax.legend(loc='upper left', frameon=True, fontsize=10)
    ax.set_xlim(min_val - 5, max_val + 5)
    ax.set_ylim(min_val - 5, max_val + 5)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'scatter_goodman_vs_king')
    plt.close()
    logger.info("Figura 1 guardada: scatter_goodman_vs_king")

    # Figure 2: Differences bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    transitions = df_comparison['Transition'].values
    differences = df_comparison['Difference'].values * 100

    colors = ['#E74C3C' if d < 0 else '#27AE60' for d in differences]

    bars = ax.bar(range(len(transitions)), differences, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=1.5)

    ax.axhline(0, color='#2C3E50', linestyle='-', linewidth=1.5)

    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels(transitions, rotation=45, ha='right', fontsize=9)

    # Value labels
    for bar, val in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}pp',
               ha='center', va='bottom' if val > 0 else 'top',
               fontsize=9, fontweight='normal')

    apply_tableau_style(ax,
                       title='Diferencias: King - Goodman (puntos porcentuales)',
                       xlabel='Transicion',
                       ylabel='Diferencia (pp)')

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'differences_king_minus_goodman')
    plt.close()
    logger.info("Figura 2 guardada: differences_king_minus_goodman")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Comparacion Goodman vs King')
    parser.add_argument('--samples', type=int, default=2000, help='Muestras MCMC para King')
    parser.add_argument('--chains', type=int, default=4, help='Cadenas MCMC')
    args = parser.parse_args()

    config = get_config()

    logger.info("="*70)
    logger.info("COMPARACION METODOLOGICA: GOODMAN vs KING")
    logger.info("="*70)

    # Load data
    df, origin_cols, destination_cols = load_and_prepare_data(config)

    # Fit both models
    goodman_model, T_goodman, uncertainty_goodman = fit_goodman(df, origin_cols, destination_cols)
    king_model, T_king, ci_king, diag_king = fit_king(df, origin_cols, destination_cols,
                                                       args.samples, args.chains)

    # Compare
    df_comparison, summary = compare_results(T_goodman, T_king, ci_king, uncertainty_goodman,
                                            origin_cols, destination_cols)

    # Save results
    save_results(df_comparison, summary, goodman_model, king_model, config)

    # Generate figures
    generate_figures(df_comparison, config)

    logger.info("="*70)
    logger.info("COMPARACION COMPLETADA")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
