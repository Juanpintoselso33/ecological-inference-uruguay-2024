"""
Posterior Predictive Checks (PPC) for King's EI National Analysis - Uruguay 2024

This script performs comprehensive posterior predictive checks to validate the
King's EI model by comparing observed ballotage outcomes with predictions
generated from the posterior distribution.

Test Statistics:
- Mean FA and PN proportions
- Standard deviation of proportions
- Min/Max proportions
- Bayesian p-values

Outputs:
- CSV summary table with test statistics
- LaTeX formatted table
- 4-panel diagnostic figure (300 DPI)
- Q-Q plots for FA and PN proportions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

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


def load_trace(trace_path: str):
    """Load MCMC trace from pickle file."""
    logger.info(f"Loading trace from {trace_path}")
    with open(trace_path, 'rb') as f:
        trace = pickle.load(f)
    logger.info(f"Trace loaded: {trace}")
    return trace


def generate_posterior_predictive_samples_from_idata(trace, X_origin, n_samples=500):
    """
    Generate posterior predictive samples from ArviZ InferenceData.

    Args:
        trace: ArviZ InferenceData with posterior samples
        X_origin: Origin vote matrix (n_circuits, n_origin_parties)
        n_samples: Number of posterior predictive samples

    Returns:
        Y_pred: Predicted destination votes (n_samples, n_circuits, n_dest_parties)
    """
    logger.info(f"Generating {n_samples} posterior predictive samples...")

    # Extract transition matrix samples
    transition_samples = trace.posterior['transition_matrix'].values

    # Flatten chains and draws
    n_chains, n_draws, n_origin, n_dest = transition_samples.shape
    transition_samples = transition_samples.reshape(-1, n_origin, n_dest)

    # Randomly sample n_samples transition matrices
    total_samples = transition_samples.shape[0]
    sample_idx = np.random.choice(total_samples, size=n_samples, replace=False)
    sampled_transitions = transition_samples[sample_idx]

    # Generate predictions
    n_circuits = X_origin.shape[0]
    Y_pred = np.zeros((n_samples, n_circuits, n_dest))

    for i in range(n_samples):
        T = sampled_transitions[i]
        Y_pred[i] = X_origin @ T

    logger.info(f"Generated predictions shape: {Y_pred.shape}")
    return Y_pred


def generate_posterior_predictive_samples_fa_only(trace, X_origin, n_samples=500):
    """
    Generate posterior predictive samples for FA-only analysis (2 origin parties).

    This uses the fa_national_transitions_2024.pkl trace which models:
    - Origin: FA primera, OTROS primera (everyone else combined)
    - Destination: FA ballotage, PN ballotage, Blancos

    Args:
        trace: ArviZ InferenceData with posterior samples
        X_origin: Origin vote matrix (n_circuits, 2) - [FA_primera, OTROS_primera]
        n_samples: Number of posterior predictive samples

    Returns:
        Y_pred: Predicted destination votes (n_samples, n_circuits, 3)
    """
    logger.info(f"Generating {n_samples} posterior predictive samples (FA-only model)...")

    # Extract transition matrix samples
    transition_samples = trace.posterior['transition_matrix'].values

    # Flatten chains and draws
    n_chains, n_draws, n_origin, n_dest = transition_samples.shape
    logger.info(f"Trace shape: chains={n_chains}, draws={n_draws}, origin={n_origin}, dest={n_dest}")

    transition_samples = transition_samples.reshape(-1, n_origin, n_dest)

    # Randomly sample n_samples transition matrices
    total_samples = transition_samples.shape[0]
    sample_idx = np.random.choice(total_samples, size=min(n_samples, total_samples), replace=False)
    sampled_transitions = transition_samples[sample_idx]

    # Generate predictions
    n_circuits = X_origin.shape[0]
    Y_pred = np.zeros((len(sample_idx), n_circuits, n_dest))

    for i in range(len(sample_idx)):
        T = sampled_transitions[i]
        Y_pred[i] = X_origin @ T

    logger.info(f"Generated predictions shape: {Y_pred.shape}")
    return Y_pred


def compute_test_statistics(Y_pred, Y_obs):
    """
    Compute test statistics for posterior predictive checks.

    Args:
        Y_pred: Predicted votes (n_samples, n_circuits, n_dest)
        Y_obs: Observed votes (n_circuits, n_dest)

    Returns:
        Dictionary with test statistics
    """
    logger.info("Computing test statistics...")

    n_samples, n_circuits, n_dest = Y_pred.shape

    # Convert to proportions
    Y_pred_prop = Y_pred / Y_pred.sum(axis=2, keepdims=True)
    Y_obs_prop = Y_obs / Y_obs.sum(axis=1, keepdims=True)

    # Test statistics
    stats_dict = {}

    # For each destination party (FA=0, PN=1, Blancos=2)
    dest_names = ['FA', 'PN', 'Blancos']

    for j, dest in enumerate(dest_names):
        # Observed statistics
        obs_mean = Y_obs_prop[:, j].mean()
        obs_std = Y_obs_prop[:, j].std()
        obs_min = Y_obs_prop[:, j].min()
        obs_max = Y_obs_prop[:, j].max()

        # Predicted statistics (across posterior samples)
        pred_means = Y_pred_prop[:, :, j].mean(axis=1)  # (n_samples,)
        pred_stds = Y_pred_prop[:, :, j].std(axis=1)
        pred_mins = Y_pred_prop[:, :, j].min(axis=1)
        pred_maxs = Y_pred_prop[:, :, j].max(axis=1)

        # Bayesian p-values (proportion of samples where pred >= obs)
        p_mean = (pred_means >= obs_mean).mean()
        p_std = (pred_stds >= obs_std).mean()
        p_min = (pred_mins <= obs_min).mean()
        p_max = (pred_maxs >= obs_max).mean()

        stats_dict[dest] = {
            'obs_mean': obs_mean,
            'pred_mean_median': np.median(pred_means),
            'pred_mean_ci_lower': np.percentile(pred_means, 2.5),
            'pred_mean_ci_upper': np.percentile(pred_means, 97.5),
            'p_value_mean': p_mean,
            'obs_std': obs_std,
            'pred_std_median': np.median(pred_stds),
            'pred_std_ci_lower': np.percentile(pred_stds, 2.5),
            'pred_std_ci_upper': np.percentile(pred_stds, 97.5),
            'p_value_std': p_std,
            'obs_min': obs_min,
            'pred_min_median': np.median(pred_mins),
            'p_value_min': p_min,
            'obs_max': obs_max,
            'pred_max_median': np.median(pred_maxs),
            'p_value_max': p_max,
        }

    return stats_dict, Y_pred_prop


def create_ppc_diagnostic_figure(Y_pred_prop, Y_obs_prop, output_path):
    """
    Create 4-panel diagnostic figure comparing observed vs predicted distributions.

    Args:
        Y_pred_prop: Predicted proportions (n_samples, n_circuits, n_dest)
        Y_obs_prop: Observed proportions (n_circuits, n_dest)
        output_path: Path to save figure
    """
    logger.info("Creating PPC diagnostic figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    dest_names = ['FA', 'PN']
    dest_colors = [PARTY_COLORS['FA'], PARTY_COLORS['PN']]

    for j, (dest, color) in enumerate(zip(dest_names, dest_colors)):
        # Top row: Density plots
        ax = axes[0, j]

        # Plot observed distribution
        ax.hist(Y_obs_prop[:, j], bins=30, density=True, alpha=0.4,
                color=color, label='Observado', edgecolor='white')

        # Plot predicted distributions (sample a few for visualization)
        for i in np.random.choice(Y_pred_prop.shape[0], size=50, replace=False):
            ax.hist(Y_pred_prop[i, :, j], bins=30, density=True, alpha=0.02,
                    color=color, edgecolor=None)

        apply_tableau_style(ax,
                          title=f'{dest}: Distribución de Proporciones',
                          xlabel='Proporción',
                          ylabel='Densidad')
        ax.legend(loc='best')

        # Bottom row: Mean comparison by circuit
        ax = axes[1, j]

        # Mean predicted proportion per circuit
        pred_mean_circuit = Y_pred_prop[:, :, j].mean(axis=0)
        pred_ci_lower = np.percentile(Y_pred_prop[:, :, j], 2.5, axis=0)
        pred_ci_upper = np.percentile(Y_pred_prop[:, :, j], 97.5, axis=0)

        # Scatter plot
        circuit_idx = np.arange(len(Y_obs_prop))
        ax.scatter(circuit_idx, Y_obs_prop[:, j], s=3, alpha=0.3,
                  color=color, label='Observado')
        ax.plot(circuit_idx, pred_mean_circuit, linewidth=1, alpha=0.7,
                color=color, label='Predicción (media)')
        ax.fill_between(circuit_idx, pred_ci_lower, pred_ci_upper,
                        alpha=0.15, color=color, label='IC 95%')

        apply_tableau_style(ax,
                          title=f'{dest}: Observado vs Predicho por Circuito',
                          xlabel='Circuito (ordenado)',
                          ylabel='Proporción')
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    save_publication_figure(fig, output_path, dpi=300)
    logger.info(f"Saved diagnostic figure to {output_path}")
    plt.close()


def create_qqplot_figure(Y_pred_prop, Y_obs_prop, output_path):
    """
    Create Q-Q plots for FA and PN proportions.

    Args:
        Y_pred_prop: Predicted proportions (n_samples, n_circuits, n_dest)
        Y_obs_prop: Observed proportions (n_circuits, n_dest)
        output_path: Path to save figure
    """
    logger.info("Creating Q-Q plot figure...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    dest_names = ['FA', 'PN']
    dest_colors = [PARTY_COLORS['FA'], PARTY_COLORS['PN']]

    for j, (dest, color) in enumerate(zip(dest_names, dest_colors)):
        ax = axes[j]

        # Use mean predicted proportions
        pred_mean_circuit = Y_pred_prop[:, :, j].mean(axis=0)
        obs = Y_obs_prop[:, j]

        # Q-Q plot
        stats.probplot(obs, dist="norm", plot=None)

        # Manual Q-Q plot
        obs_sorted = np.sort(obs)
        pred_sorted = np.sort(pred_mean_circuit)

        ax.scatter(pred_sorted, obs_sorted, s=20, alpha=0.5, color=color)

        # 45-degree reference line
        min_val = min(pred_sorted.min(), obs_sorted.min())
        max_val = max(pred_sorted.max(), obs_sorted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--',
                linewidth=2, alpha=0.7, label='Línea 1:1')

        apply_tableau_style(ax,
                          title=f'{dest}: Q-Q Plot',
                          xlabel='Predicho (cuantiles)',
                          ylabel='Observado (cuantiles)')
        ax.legend(loc='best')

    plt.tight_layout()
    save_publication_figure(fig, output_path, dpi=300)
    logger.info(f"Saved Q-Q plot to {output_path}")
    plt.close()


def save_results_table(stats_dict, output_csv, output_latex):
    """
    Save test statistics to CSV and LaTeX tables.

    Args:
        stats_dict: Dictionary with test statistics
        output_csv: Path to CSV file
        output_latex: Path to LaTeX file
    """
    logger.info("Saving results tables...")

    # Create DataFrame
    rows = []
    for dest, stats in stats_dict.items():
        rows.append({
            'Partido': dest,
            'Estadístico': 'Media',
            'Observado': f"{stats['obs_mean']:.4f}",
            'Predicho (Mediana)': f"{stats['pred_mean_median']:.4f}",
            'IC 95%': f"[{stats['pred_mean_ci_lower']:.4f}, {stats['pred_mean_ci_upper']:.4f}]",
            'p-valor': f"{stats['p_value_mean']:.3f}"
        })
        rows.append({
            'Partido': dest,
            'Estadístico': 'Desv. Est.',
            'Observado': f"{stats['obs_std']:.4f}",
            'Predicho (Mediana)': f"{stats['pred_std_median']:.4f}",
            'IC 95%': f"[{stats['pred_std_ci_lower']:.4f}, {stats['pred_std_ci_upper']:.4f}]",
            'p-valor': f"{stats['p_value_std']:.3f}"
        })
        rows.append({
            'Partido': dest,
            'Estadístico': 'Mínimo',
            'Observado': f"{stats['obs_min']:.4f}",
            'Predicho (Mediana)': f"{stats['pred_min_median']:.4f}",
            'IC 95%': '-',
            'p-valor': f"{stats['p_value_min']:.3f}"
        })
        rows.append({
            'Partido': dest,
            'Estadístico': 'Máximo',
            'Observado': f"{stats['obs_max']:.4f}",
            'Predicho (Mediana)': f"{stats['pred_max_median']:.4f}",
            'IC 95%': '-',
            'p-valor': f"{stats['p_value_max']:.3f}"
        })

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved CSV to {output_csv}")

    # Save LaTeX
    latex_code = df.to_latex(
        index=False,
        caption='Posterior Predictive Checks: Estadísticos de Prueba',
        label='tab:ppc_summary',
        escape=False,
        column_format='llcccc'
    )

    with open(output_latex, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    logger.info(f"Saved LaTeX to {output_latex}")


def main():
    """Main execution function."""
    print("="*80)
    print("POSTERIOR PREDICTIVE CHECKS - KING'S EI NATIONAL ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Load configuration
    config = get_config()

    # Paths - FIX THE BUG HERE
    data_path = Path(config.get('data.processed_dir')) / 'circuitos_merged.parquet'
    tables_dir = Path(config.get('outputs.tables_dir'))
    figures_dir = Path(config.get('outputs.figures_dir'))

    # Fix: construct results_dir properly
    results_dir = tables_dir.parent / 'results'

    trace_path = results_dir / 'fa_national_transitions_2024.pkl'

    # Ensure output directories exist
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latex_dir = tables_dir / 'latex'
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Total circuits: {len(df):,}")

    # Define columns for FA-only analysis (2 origin parties)
    # Origin: FA primera, OTROS primera (all non-FA parties combined)
    # Destination: FA ballotage, PN ballotage, Blancos+Anulados

    # Combine blancos + anulados
    df['blancos_anulados_ballotage'] = (
        df['blancos_ballotage'] + df['anulados_ballotage']
    )

    # Create OTROS as complement of FA
    df['otros_primera_all'] = df['total_primera'] - df['fa_primera']

    origin_cols = ['fa_primera', 'otros_primera_all']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_anulados_ballotage']

    # Extract data
    data_clean = df[origin_cols + destination_cols].dropna()
    logger.info(f"Clean circuits: {len(data_clean):,}")

    X_origin = data_clean[origin_cols].values
    Y_obs = data_clean[destination_cols].values

    # Load trace
    trace = load_trace(trace_path)

    # Generate posterior predictive samples (FA-only model with 2 origin parties)
    Y_pred = generate_posterior_predictive_samples_fa_only(trace, X_origin, n_samples=500)

    # Compute test statistics
    stats_dict, Y_pred_prop = compute_test_statistics(Y_pred, Y_obs)

    # Compute observed proportions
    Y_obs_prop = Y_obs / Y_obs.sum(axis=1, keepdims=True)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    for dest, stats in stats_dict.items():
        print(f"\n{dest}:")
        print(f"  Mean: obs={stats['obs_mean']:.4f}, "
              f"pred={stats['pred_mean_median']:.4f}, "
              f"p={stats['p_value_mean']:.3f}")
        print(f"  Std:  obs={stats['obs_std']:.4f}, "
              f"pred={stats['pred_std_median']:.4f}, "
              f"p={stats['p_value_std']:.3f}")

    # Save results
    output_csv = tables_dir / 'ppc_summary.csv'
    output_latex = latex_dir / 'ppc_summary.tex'
    save_results_table(stats_dict, output_csv, output_latex)

    # Create figures
    fig1_path = figures_dir / 'ppc_diagnostics'
    create_ppc_diagnostic_figure(Y_pred_prop, Y_obs_prop, fig1_path)

    fig2_path = figures_dir / 'ppc_qqplot'
    create_qqplot_figure(Y_pred_prop, Y_obs_prop, fig2_path)

    print("\n" + "="*80)
    print("POSTERIOR PREDICTIVE CHECKS COMPLETE")
    print("="*80)
    print(f"Results saved to:")
    print(f"  - {output_csv}")
    print(f"  - {output_latex}")
    print(f"  - {fig1_path}.png")
    print(f"  - {fig2_path}.png")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == '__main__':
    main()
