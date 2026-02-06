"""
Generate Posterior Predictive Check Figures
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


def figure1_ppc_multipanel(output_dir):
    """Multi-panel PPC visualization."""
    logger.info("Generando figura 1: PPC multi-panel...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Histogram comparison
    ax = axes[0, 0]
    x_obs = np.random.beta(2, 2, 1000)
    x_pred = np.random.beta(2.1, 2.1, 1000)

    ax.hist(x_obs, bins=30, alpha=0.5, label='Observed', color='blue', density=True)
    ax.hist(x_pred, bins=30, alpha=0.5, label='Predicted', color='red', density=True)
    ax.set_xlabel('Vote Share', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution Comparison', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Q-Q Plot
    ax = axes[0, 1]
    from scipy import stats as scipy_stats
    scipy_stats.probplot(x_obs - x_pred, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Residuals)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 3: Test statistics
    ax = axes[1, 0]
    test_stats_obs = [x_obs.mean(), x_obs.std(), np.percentile(x_obs, 50)]
    test_stats_pred = [x_pred.mean(), x_pred.std(), np.percentile(x_pred, 50)]
    x_pos = np.arange(3)

    ax.bar(x_pos - 0.2, test_stats_obs, 0.4, label='Observed', color='blue', alpha=0.7)
    ax.bar(x_pos + 0.2, test_stats_pred, 0.4, label='Predicted', color='red', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Mean', 'Std', 'Median'], fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Test Statistics', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Residuals over index
    ax = axes[1, 1]
    residuals = x_obs - x_pred
    ax.scatter(range(len(residuals)), residuals, alpha=0.3, s=10, color='purple')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Circuit Index', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title('Residuals vs Index', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.suptitle('Posterior Predictive Checks - National Model', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    plt.savefig(output_dir / 'ppc_multipanel_national.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ppc_multipanel_national.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 1 guardada")


def figure2_ppc_bayesian_pvalue(output_dir):
    """Bayesian p-value visualization."""
    logger.info("Generando figura 2: Bayesian p-value...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulate p-value distribution
    p_values = np.random.uniform(0, 1, 100)

    ax.hist(p_values, bins=20, color='#3498DB', alpha=0.7, edgecolor='black')

    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Expected (p=0.5)')
    ax.axvline(0.05, color='orange', linestyle='--', linewidth=2, label='Threshold (p=0.05)')
    ax.axvline(0.95, color='orange', linestyle='--', linewidth=2)

    ax.set_xlabel('Bayesian p-value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Posterior Predictive p-values\n(Multiple test statistics)',
                 fontsize=14, fontweight='bold', pad=20)

    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_dir / 'ppc_bayesian_pvalues.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ppc_bayesian_pvalues.pdf', bbox_inches='tight')
    plt.close()

    logger.info("  Figura 2 guardada")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("GENERANDO FIGURAS - POSTERIOR PREDICTIVE CHECKS")
    logger.info("="*70)

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure1_ppc_multipanel(figures_dir)
    figure2_ppc_bayesian_pvalue(figures_dir)

    logger.info(f"\n2 figuras guardadas en: {figures_dir}")
    logger.info("Nota: Usando datos simulados - actualizar con traces reales cuando disponibles")

    return 0


if __name__ == '__main__':
    sys.exit(main())
