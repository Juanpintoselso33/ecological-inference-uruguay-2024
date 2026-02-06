"""
Análisis de Sensibilidad para Inferencia Ecológica
Valida robustez de resultados bajo diferentes supuestos metodológicos
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.models.king_ei import KingEI
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.visualization.styles import setup_professional_style

logger = get_logger(__name__)
setup_professional_style()


def load_data(config):
    """Load processed data."""
    logger.info("Cargando datos...")

    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])
    data_path = processed_dir / 'circuitos_merged.parquet'

    df = pd.read_parquet(data_path)
    logger.info(f"Datos cargados: {len(df)} circuitos")

    return df


def prepare_baseline_data(df):
    """Prepare baseline data for CA analysis."""
    logger.info("Preparando datos baseline (CA analysis)...")

    df = df.copy()

    # CA analysis (most interesting for sensitivity)
    df['blancos_ballotage'] = df.get('blancos_ballotage', 0) + df.get('anulados_ballotage', 0)

    # Recalculate OTROS as complement of CA
    df['otros_ca_recalc'] = df['total_primera'] - df['ca_primera']

    origin_cols = ['ca_primera', 'otros_ca_recalc']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Filter valid circuits
    min_votes = 30
    valid = (df['total_primera'] >= min_votes) & (df['total_ballotage'] >= min_votes)
    df_clean = df[valid].copy()

    logger.info(f"Circuitos válidos: {len(df_clean)} ({100*len(df_clean)/len(df):.1f}%)")

    return df_clean, origin_cols, destination_cols


def test_baseline(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Run baseline King's EI (for comparison)."""
    logger.info("TEST 1/7: Baseline King's EI (Dirichlet[1,1,1])...")

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

    logger.info(f"  CA→FA: {T[0,0]:.1%} [{ci['lower'][0,0]:.1%}, {ci['upper'][0,0]:.1%}]")
    logger.info(f"  CA→PN: {T[0,1]:.1%} [{ci['lower'][0,1]:.1%}, {ci['upper'][0,1]:.1%}]")
    logger.info(f"  R-hat max: {np.max(diag['rhat']):.4f}")

    return {
        'name': 'Baseline',
        'description': 'Default priors (Dirichlet[1,1,1])',
        'T': T,
        'ci': ci,
        'diag': diag,
        'trace': model.trace_
    }


def test_conservative_prior(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Test with more conservative (tighter) priors."""
    logger.info("TEST 2/7: Conservative Prior (Dirichlet[2,2,2])...")

    # Note: KingEI uses hardcoded Dirichlet(1) prior in model
    # This test would require modifying the model class to accept custom priors
    # For now, we'll document this as a limitation

    logger.warning("  Custom priors no soportados en implementación actual")
    logger.warning("  Se requiere modificar src/models/king_ei.py para aceptar prior_alpha parameter")

    return None


def test_informative_prior(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Test with informative prior (biased toward retention)."""
    logger.info("TEST 3/7: Informative Prior (Dirichlet[5,1,1])...")

    # Same limitation as conservative prior
    logger.warning("  Custom priors no soportados en implementación actual")

    return None


def test_remove_outliers_ca_share(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Test removing top 1% extreme CA share circuits."""
    logger.info("TEST 4/7: Remove Outliers (top 1% CA share)...")

    df_filtered = df.copy()

    # Calculate CA share
    ca_share = df_filtered['ca_primera'] / df_filtered['total_primera']

    # Remove top 1%
    threshold_high = ca_share.quantile(0.99)
    threshold_low = ca_share.quantile(0.01)

    mask = (ca_share >= threshold_low) & (ca_share <= threshold_high)
    df_filtered = df_filtered[mask]

    logger.info(f"  Circuits after filtering: {len(df_filtered)} (removed {len(df) - len(df_filtered)})")

    model = KingEI(
        num_samples=num_samples,
        num_chains=num_chains,
        num_warmup=num_samples // 2,
        random_seed=42
    )

    model.fit(
        data=df_filtered,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(prob=0.95)
    diag = model.get_diagnostics()

    logger.info(f"  CA→FA: {T[0,0]:.1%} [{ci['lower'][0,0]:.1%}, {ci['upper'][0,0]:.1%}]")
    logger.info(f"  CA→PN: {T[0,1]:.1%} [{ci['lower'][0,1]:.1%}, {ci['upper'][0,1]:.1%}]")

    return {
        'name': 'No_Outliers',
        'description': 'Removed top/bottom 1% CA share circuits',
        'n_circuits': len(df_filtered),
        'T': T,
        'ci': ci,
        'diag': diag,
        'trace': model.trace_
    }


def test_bootstrap(df, origin_cols, destination_cols, n_bootstrap=20, sample_frac=0.8,
                   num_samples=1000, num_chains=2):
    """Bootstrap stability test (reduced samples for speed)."""
    logger.info(f"TEST 5/7: Bootstrap Stability ({n_bootstrap} samples, {sample_frac:.0%} each)...")

    bootstrap_results = []

    for i in range(n_bootstrap):
        logger.info(f"  Bootstrap sample {i+1}/{n_bootstrap}...")

        # Sample with replacement
        df_boot = df.sample(frac=sample_frac, replace=True, random_state=42+i)

        try:
            model = KingEI(
                num_samples=num_samples,
                num_chains=num_chains,
                num_warmup=num_samples // 2,
                random_seed=42+i
            )

            model.fit(
                data=df_boot,
                origin_cols=origin_cols,
                destination_cols=destination_cols,
                total_origin='total_primera',
                total_destination='total_ballotage',
                progressbar=False  # Suppress progress bars for bootstrap
            )

            T = model.get_transition_matrix()

            bootstrap_results.append({
                'sample': i,
                'ca_to_fa': T[0, 0],
                'ca_to_pn': T[0, 1],
                'ca_to_blancos': T[0, 2]
            })

        except Exception as e:
            logger.warning(f"  Bootstrap sample {i+1} failed: {e}")
            continue

    df_bootstrap = pd.DataFrame(bootstrap_results)

    if len(df_bootstrap) > 0:
        logger.info(f"  Successful bootstrap samples: {len(df_bootstrap)}/{n_bootstrap}")
        logger.info(f"  CA→FA: {df_bootstrap['ca_to_fa'].mean():.1%} ± {df_bootstrap['ca_to_fa'].std():.1%}")
        logger.info(f"  CA→PN: {df_bootstrap['ca_to_pn'].mean():.1%} ± {df_bootstrap['ca_to_pn'].std():.1%}")

    return {
        'name': 'Bootstrap',
        'description': f'{n_bootstrap} bootstrap samples with {sample_frac:.0%} circuits each',
        'results': df_bootstrap,
        'n_successful': len(df_bootstrap)
    }


def test_mcmc_samples(df, origin_cols, destination_cols, num_samples_list=[1000, 2000, 4000]):
    """Test different MCMC sample sizes."""
    logger.info(f"TEST 6/7: MCMC Sample Size Sensitivity...")

    mcmc_results = []

    for num_samples in num_samples_list:
        logger.info(f"  Testing {num_samples} samples...")

        model = KingEI(
            num_samples=num_samples,
            num_chains=4,
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

        mcmc_results.append({
            'num_samples': num_samples,
            'ca_to_fa': T[0, 0],
            'ca_to_pn': T[0, 1],
            'ca_to_fa_ci_width': ci['upper'][0, 0] - ci['lower'][0, 0],
            'rhat_max': np.max(diag['rhat']),
            'ess_min': np.min(diag['ess'])
        })

        logger.info(f"    CA→FA: {T[0,0]:.1%}, R-hat: {np.max(diag['rhat']):.4f}")

    df_mcmc = pd.DataFrame(mcmc_results)

    return {
        'name': 'MCMC_Samples',
        'description': 'Convergence with different sample sizes',
        'results': df_mcmc
    }


def test_small_ca_circuits(df, origin_cols, destination_cols, num_samples=2000, num_chains=4):
    """Test excluding very small CA circuits."""
    logger.info("TEST 7/7: Exclude Small CA Circuits (< 5 CA votes)...")

    df_filtered = df.copy()

    # Filter circuits with very few CA votes
    min_ca_votes = 5
    mask = df_filtered['ca_primera'] >= min_ca_votes
    df_filtered = df_filtered[mask]

    logger.info(f"  Circuits after filtering: {len(df_filtered)} (removed {len(df) - len(df_filtered)})")

    model = KingEI(
        num_samples=num_samples,
        num_chains=num_chains,
        num_warmup=num_samples // 2,
        random_seed=42
    )

    model.fit(
        data=df_filtered,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True
    )

    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(prob=0.95)
    diag = model.get_diagnostics()

    logger.info(f"  CA→FA: {T[0,0]:.1%} [{ci['lower'][0,0]:.1%}, {ci['upper'][0,0]:.1%}]")
    logger.info(f"  CA→PN: {T[0,1]:.1%} [{ci['lower'][0,1]:.1%}, {ci['upper'][0,1]:.1%}]")

    return {
        'name': 'No_Small_CA',
        'description': f'Removed circuits with < {min_ca_votes} CA votes',
        'n_circuits': len(df_filtered),
        'T': T,
        'ci': ci,
        'diag': diag,
        'trace': model.trace_
    }


def compare_results(baseline, tests):
    """Compare all sensitivity test results against baseline."""
    logger.info("Comparando resultados...")

    comparisons = []

    # Baseline values
    baseline_ca_fa = baseline['T'][0, 0]
    baseline_ca_pn = baseline['T'][0, 1]

    for test in tests:
        if test is None:
            continue

        test_name = test['name']

        if test_name == 'Bootstrap':
            # Bootstrap comparison
            df_boot = test['results']
            if len(df_boot) > 0:
                comparisons.append({
                    'Test': 'Bootstrap',
                    'CA→FA_Mean': df_boot['ca_to_fa'].mean(),
                    'CA→FA_Diff': df_boot['ca_to_fa'].mean() - baseline_ca_fa,
                    'CA→FA_CV': df_boot['ca_to_fa'].std() / df_boot['ca_to_fa'].mean(),
                    'CA→PN_Mean': df_boot['ca_to_pn'].mean(),
                    'CA→PN_Diff': df_boot['ca_to_pn'].mean() - baseline_ca_pn,
                    'CA→PN_CV': df_boot['ca_to_pn'].std() / df_boot['ca_to_pn'].mean(),
                    'N_Samples': len(df_boot)
                })

        elif test_name == 'MCMC_Samples':
            # MCMC sample size comparison (use largest sample)
            df_mcmc = test['results']
            largest = df_mcmc.iloc[-1]

            comparisons.append({
                'Test': f'MCMC_{largest["num_samples"]}',
                'CA→FA_Mean': largest['ca_to_fa'],
                'CA→FA_Diff': largest['ca_to_fa'] - baseline_ca_fa,
                'CA→FA_CV': 0.0,  # Not applicable
                'CA→PN_Mean': largest['ca_to_pn'],
                'CA→PN_Diff': largest['ca_to_pn'] - baseline_ca_pn,
                'CA→PN_CV': 0.0,
                'N_Samples': largest['num_samples']
            })

        else:
            # Regular comparison (outliers, small CA)
            if 'T' in test:
                comparisons.append({
                    'Test': test_name,
                    'CA→FA_Mean': test['T'][0, 0],
                    'CA→FA_Diff': test['T'][0, 0] - baseline_ca_fa,
                    'CA→FA_CV': 0.0,  # Not applicable
                    'CA→PN_Mean': test['T'][0, 1],
                    'CA→PN_Diff': test['T'][0, 1] - baseline_ca_pn,
                    'CA→PN_CV': 0.0,
                    'N_Samples': test.get('n_circuits', 0)
                })

    df_comparison = pd.DataFrame(comparisons)

    if len(df_comparison) > 0:
        logger.info("\nResumen de comparaciones:")
        logger.info(f"  CA→FA diferencia máxima: {df_comparison['CA→FA_Diff'].abs().max():.4f}")
        logger.info(f"  CA→PN diferencia máxima: {df_comparison['CA→PN_Diff'].abs().max():.4f}")
        if 'Bootstrap' in df_comparison['Test'].values:
            logger.info(f"  Bootstrap CV (CA→FA): {df_comparison.loc[df_comparison['Test']=='Bootstrap', 'CA→FA_CV'].values[0]:.4f}")

    return df_comparison


def calculate_robustness_metrics(df_comparison):
    """Calculate overall robustness metrics."""
    logger.info("Calculando métricas de robustez...")

    metrics = {
        'max_abs_diff_ca_fa': df_comparison['CA→FA_Diff'].abs().max(),
        'mean_abs_diff_ca_fa': df_comparison['CA→FA_Diff'].abs().mean(),
        'max_abs_diff_ca_pn': df_comparison['CA→PN_Diff'].abs().max(),
        'mean_abs_diff_ca_pn': df_comparison['CA→PN_Diff'].abs().mean(),
        'bootstrap_cv_ca_fa': df_comparison.loc[df_comparison['Test']=='Bootstrap', 'CA→FA_CV'].values[0] if 'Bootstrap' in df_comparison['Test'].values else np.nan,
        'n_tests_completed': len(df_comparison)
    }

    logger.info(f"  Diferencia absoluta máxima (CA→FA): {metrics['max_abs_diff_ca_fa']:.4f}")
    if not np.isnan(metrics['bootstrap_cv_ca_fa']):
        logger.info(f"  Coeficiente de variación bootstrap: {metrics['bootstrap_cv_ca_fa']:.4f}")

    return metrics


def save_results(baseline, tests, df_comparison, metrics, config):
    """Save sensitivity analysis results."""
    logger.info("Guardando resultados...")

    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    # Construct results_dir manually
    results_dir = tables_dir.parent / 'results'

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save baseline trace
    baseline_path = results_dir / 'sensitivity_baseline.pkl'
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline, f)
    logger.info(f"  Baseline guardado: {baseline_path}")

    # 2. Save all test results
    for test in tests:
        if test is None:
            continue

        test_name = test['name']
        test_path = results_dir / f'sensitivity_{test_name.lower()}.pkl'

        with open(test_path, 'wb') as f:
            pickle.dump(test, f)
        logger.info(f"  Test guardado: {test_path}")

    # 3. Save comparison table (CSV)
    csv_path = tables_dir / 'sensitivity_comparison.csv'
    df_comparison.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"  Comparación guardada: {csv_path}")

    # 4. Save metrics
    metrics_path = tables_dir / 'sensitivity_metrics.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False, float_format='%.4f')
    logger.info(f"  Métricas guardadas: {metrics_path}")

    # 5. Save LaTeX table
    df_latex = df_comparison[['Test', 'CA→FA_Mean', 'CA→FA_Diff', 'CA→PN_Mean', 'CA→PN_Diff']].copy()
    df_latex.columns = ['Prueba', 'CA→FA', 'Δ CA→FA', 'CA→PN', 'Δ CA→PN']
    df_latex['CA→FA'] = df_latex['CA→FA'].apply(lambda x: f"{x:.1%}")
    df_latex['Δ CA→FA'] = df_latex['Δ CA→FA'].apply(lambda x: f"{x:+.1%}")
    df_latex['CA→PN'] = df_latex['CA→PN'].apply(lambda x: f"{x:.1%}")
    df_latex['Δ CA→PN'] = df_latex['Δ CA→PN'].apply(lambda x: f"{x:+.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lcccc',
        caption='Análisis de sensibilidad: robustez de estimaciones',
        label='tab:sensitivity'
    )

    latex_path = latex_dir / 'sensitivity_comparison.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    logger.info(f"  LaTeX guardado: {latex_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Análisis de Sensibilidad')
    parser.add_argument('--samples', type=int, default=2000, help='MCMC samples para tests principales')
    parser.add_argument('--chains', type=int, default=4, help='MCMC chains')
    parser.add_argument('--bootstrap', type=int, default=20, help='Número de muestras bootstrap')
    args = parser.parse_args()

    config = get_config()

    logger.info("="*70)
    logger.info("ANALISIS DE SENSIBILIDAD")
    logger.info("="*70)

    # Load and prepare data
    df = load_data(config)
    df_clean, origin_cols, destination_cols = prepare_baseline_data(df)

    # Run tests
    tests = []

    # Test 1: Baseline
    baseline = test_baseline(df_clean, origin_cols, destination_cols, args.samples, args.chains)

    # Test 2-3: Custom priors (not implemented yet)
    tests.append(test_conservative_prior(df_clean, origin_cols, destination_cols, args.samples, args.chains))
    tests.append(test_informative_prior(df_clean, origin_cols, destination_cols, args.samples, args.chains))

    # Test 4: Remove outliers
    tests.append(test_remove_outliers_ca_share(df_clean, origin_cols, destination_cols, args.samples, args.chains))

    # Test 5: Bootstrap
    tests.append(test_bootstrap(df_clean, origin_cols, destination_cols, n_bootstrap=args.bootstrap))

    # Test 6: MCMC sample sizes
    tests.append(test_mcmc_samples(df_clean, origin_cols, destination_cols))

    # Test 7: Small CA circuits
    tests.append(test_small_ca_circuits(df_clean, origin_cols, destination_cols, args.samples, args.chains))

    # Compare results
    df_comparison = compare_results(baseline, tests)
    metrics = calculate_robustness_metrics(df_comparison)

    # Save results
    save_results(baseline, tests, df_comparison, metrics, config)

    logger.info("="*70)
    logger.info("ANALISIS DE SENSIBILIDAD COMPLETADO")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
