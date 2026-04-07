"""
King's Ecological Inference implementation using PyMC.
Bayesian method that respects bounds [0,1] for transition probabilities.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

import pymc as pm
import arviz as az

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_ei import BaseEIModel
from src.utils import get_logger, get_config

logger = get_logger(__name__)


class KingEI(BaseEIModel):
    """
    King's Ecological Inference model using PyMC.

    Implements a Bayesian hierarchical model to estimate transition probabilities
    from origin election to destination election while respecting bounds [0,1].

    For each origin party i and destination party j, estimates:
        p_ij = P(vote for j in destination | voted for i in origin)

    Subject to: Σ_j p_ij = 1 for each i (rows sum to 1)
                0 ≤ p_ij ≤ 1 for all i,j

    Uses Dirichlet prior to ensure row sums = 1.
    """

    def __init__(self,
                 num_samples: int = 2000,
                 num_chains: int = 4,
                 num_warmup: int = 1000,
                 target_accept: float = 0.9,
                 random_seed: int = 42,
                 likelihood: str = 'normal'):
        """
        Initialize King's EI model.

        Args:
            num_samples: Number of MCMC samples per chain
            num_chains: Number of parallel MCMC chains
            num_warmup: Number of warmup/burn-in samples
            target_accept: Target acceptance rate for NUTS sampler
            random_seed: Random seed for reproducibility
            likelihood: Observation likelihood — 'normal' (default, backward-compatible)
                or 'dirichlet_multinomial' (statistically correct for count data)
        """
        super().__init__(name="KingEI")

        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.likelihood = likelihood

        self.model_ = None
        self.trace_ = None

    def fit(self, data: pd.DataFrame,
            origin_cols: List[str],
            destination_cols: List[str],
            total_origin: str,
            total_destination: str,
            progressbar: bool = True,
            covariate_cols: Optional[List[str]] = None) -> 'KingEI':
        """
        Fit King's EI model using MCMC.

        Args:
            data: DataFrame with circuit-level vote data
            origin_cols: Column names for origin election parties
            destination_cols: Column names for destination election parties
            total_origin: Column name for total votes in origin election
            total_destination: Column name for total votes in destination election
            progressbar: Show MCMC progress bar

        Returns:
            Self (fitted model)
        """
        # Validate inputs
        self.validate_inputs(data, origin_cols, destination_cols, total_origin, total_destination)

        # Store column names
        self.origin_cols_ = origin_cols
        self.destination_cols_ = destination_cols
        self.origin_party_names_ = [col.replace('_primera', '').upper() for col in origin_cols]
        self.destination_party_names_ = [col.replace('_ballotage', '').upper() for col in destination_cols]

        # Drop rows with NaN values — include all needed columns
        all_cols = origin_cols + destination_cols + [total_origin, total_destination]
        if covariate_cols:
            all_cols = all_cols + covariate_cols
        data_clean = data[all_cols].dropna()
        logger.info(f"Fitting on {len(data_clean)} circuits (dropped {len(data) - len(data_clean)} with NaN)")

        # Extract data
        X = data_clean[origin_cols].values  # Origin votes (n_circuits, n_origin)
        Y = data_clean[destination_cols].values  # Destination votes (n_circuits, n_dest)

        n_circuits, n_origin = X.shape
        n_dest = Y.shape[1]

        # Extract covariates if provided
        Z = None
        if covariate_cols:
            Z = data_clean[covariate_cols].values.astype(float)

        logger.info(f"Building Bayesian model: {n_origin} origin parties → {n_dest} destination parties "
                    f"[likelihood={self.likelihood}]")

        # Build PyMC model — dispatch on likelihood type
        if self.likelihood == 'normal':
            model = self._build_model_normal(X, Y, n_origin, n_dest)
        elif self.likelihood == 'dirichlet_multinomial':
            # Use sum of observed destination cols as n (NOT total_destination, which
            # includes nulos/observados not in our destination categories)
            N2 = Y.sum(axis=1).astype(int)
            model = self._build_model_dirichlet_multinomial(X, Y, N2, n_origin, n_dest, Z=Z)
        else:
            raise ValueError(
                f"Unknown likelihood '{self.likelihood}'. "
                "Choose 'normal' or 'dirichlet_multinomial'."
            )

        # Sample from posterior
        logger.info(f"Starting MCMC sampling: {self.num_chains} chains, {self.num_samples} samples, {self.num_warmup} warmup")
        with model:
            trace = pm.sample(
                draws=self.num_samples,
                tune=self.num_warmup,
                chains=self.num_chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=progressbar,
                return_inferencedata=True
            )

        # Store results
        self.model_ = model
        self.trace_ = trace
        self.data_ = data.copy()
        self.is_fitted = True

        # Compute diagnostics
        self._compute_diagnostics()

        logger.info("✓ Model fitted successfully")

        return self

    def _build_model_normal(self, X, Y, n_origin, n_dest):
        """Build PyMC model with Normal (Gaussian) observation likelihood.

        This is the original ad-hoc likelihood — retained for backward
        compatibility. Uses sqrt(Y_pred + 1) as observation noise.
        """
        with pm.Model() as model:
            T = pm.Dirichlet(
                'transition_matrix',
                a=np.ones(n_dest),
                shape=(n_origin, n_dest)
            )
            Y_pred_mean = pm.math.dot(X, T)
            Y_pred_std = pm.math.sqrt(Y_pred_mean + 1.0)
            Y_obs = pm.Normal(
                'destination_votes',
                mu=Y_pred_mean,
                sigma=Y_pred_std,
                observed=Y
            )
        return model

    def _build_model_dirichlet_multinomial(self, X, Y, N, n_origin, n_dest, Z=None):
        """Build PyMC model with DirichletMultinomial observation likelihood.

        Statistically correct for compositional count data. Accounts for
        overdispersion via a concentration parameter phi ~ HalfNormal(100).

        When Z (covariates) are provided, concentration varies by circuit:
            log(phi_i) = alpha + Z_i @ gamma

        Args:
            X: Origin vote counts, shape (n_circuits, n_origin)
            Y: Destination vote counts, shape (n_circuits, n_dest)
            N: Total destination votes per circuit, shape (n_circuits,)
            n_origin: Number of origin parties
            n_dest: Number of destination parties
            Z: Optional covariate array, shape (n_circuits, n_covariates)
        """
        with pm.Model() as model:
            T = pm.Dirichlet(
                'transition_matrix',
                a=np.ones(n_dest),
                shape=(n_origin, n_dest),
            )

            if Z is not None and Z.shape[1] > 0:
                n_cov = Z.shape[1]
                alpha_conc = pm.Normal('alpha_concentration', mu=3.0, sigma=1.0)
                gamma_conc = pm.Normal('gamma_concentration', mu=0.0, sigma=1.0, shape=n_cov)
                log_phi = alpha_conc + pm.math.dot(Z, gamma_conc)
                concentration = pm.Deterministic('concentration', pm.math.exp(log_phi))
            else:
                concentration = pm.HalfNormal('concentration', sigma=100.0)

            X_prop = X / np.maximum(X.sum(axis=1, keepdims=True), 1)
            pi = pm.math.dot(X_prop, T)

            if Z is not None and Z.shape[1] > 0:
                alpha_pred = pi * concentration[:, None]
            else:
                alpha_pred = pi * concentration

            Y_obs = pm.DirichletMultinomial(
                'destination_votes',
                n=N.astype(int),
                a=alpha_pred,
                observed=Y.astype(int),
            )
        return model

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get posterior mean of transition matrix.

        Returns:
            Transition matrix (posterior mean)
            Shape: (n_origin_parties, n_destination_parties)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        # Extract posterior samples
        posterior = self.trace_.posterior['transition_matrix']

        # Compute mean across chains and draws
        T_mean = posterior.mean(dim=['chain', 'draw']).values

        return T_mean

    def get_transition_matrix_samples(self) -> np.ndarray:
        """
        Get all posterior samples of transition matrix.

        Returns:
            Transition matrix samples
            Shape: (n_chains * n_draws, n_origin, n_dest)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        # Extract posterior samples
        posterior = self.trace_.posterior['transition_matrix']

        # Stack chains and draws
        samples = posterior.stack(sample=['chain', 'draw']).values.T

        return samples

    def get_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates from posterior.

        Returns:
            Dictionary with:
            - 'mean': Posterior mean
            - 'std': Posterior standard deviation
            - 'lower_bound': 2.5th percentile (95% credible interval)
            - 'upper_bound': 97.5th percentile
            - 'median': 50th percentile
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        posterior = self.trace_.posterior['transition_matrix']

        uncertainty = {
            'mean': posterior.mean(dim=['chain', 'draw']).values,
            'std': posterior.std(dim=['chain', 'draw']).values,
            'median': posterior.median(dim=['chain', 'draw']).values,
            'lower_bound': posterior.quantile(0.025, dim=['chain', 'draw']).values,
            'upper_bound': posterior.quantile(0.975, dim=['chain', 'draw']).values,
        }

        return uncertainty

    def get_credible_intervals(self, prob: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Get credible intervals for transition probabilities.

        Args:
            prob: Probability level (default 0.95 for 95% CI)

        Returns:
            Dictionary with 'lower' and 'upper' bounds
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        alpha = 1 - prob
        lower_q = alpha / 2
        upper_q = 1 - alpha / 2

        posterior = self.trace_.posterior['transition_matrix']

        return {
            'lower': posterior.quantile(lower_q, dim=['chain', 'draw']).values,
            'upper': posterior.quantile(upper_q, dim=['chain', 'draw']).values
        }

    def get_bounds(self, origin_cols, dest_cols, total_origin, total_destination):
        """Compute Duncan-Davis deterministic bounds for this model's data.

        Returns
        -------
        dict mapping origin party to {'lower': array(n,d), 'upper': array(n,d)}
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")
        from src.diagnostics.bounds import compute_duncan_davis_bounds
        return compute_duncan_davis_bounds(
            self.data_, origin_cols, dest_cols, total_origin, total_destination
        )

    def get_loo(self):
        """Compute PSIS/LOO-CV for the fitted model.

        Returns
        -------
        LOOResult with elpd_loo, p_loo, looic, se, pareto_k, n_bad_k, warning
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")
        from src.diagnostics.loo import compute_loo
        return compute_loo(self)

    def _compute_diagnostics(self) -> None:
        """Compute MCMC diagnostics (R-hat, ESS, etc.)."""
        # Compute R-hat (Gelman-Rubin statistic)
        rhat = az.rhat(self.trace_, var_names=['transition_matrix'])
        self.rhat_ = rhat['transition_matrix'].values

        # Compute effective sample size
        ess = az.ess(self.trace_, var_names=['transition_matrix'])
        self.ess_ = ess['transition_matrix'].values

        # Check convergence
        max_rhat = np.max(self.rhat_)
        min_ess = np.min(self.ess_)

        if max_rhat > 1.01:
            logger.warning(f"Poor convergence: max R-hat = {max_rhat:.4f} (should be < 1.01)")
        else:
            logger.info(f"✓ Convergence: max R-hat = {max_rhat:.4f}")

        if min_ess < 1000:
            logger.warning(f"Low effective sample size: min ESS = {min_ess:.0f} (should be > 1000)")
        else:
            logger.info(f"✓ Effective sample size: min ESS = {min_ess:.0f}")

    def get_diagnostics(self) -> Dict[str, np.ndarray]:
        """
        Get MCMC diagnostics.

        Returns:
            Dictionary with:
            - 'rhat': Gelman-Rubin statistic (should be < 1.01)
            - 'ess': Effective sample size (should be > 1000)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        return {
            'rhat': self.rhat_,
            'ess': self.ess_
        }

    def plot_trace(self, save_path: Optional[str] = None) -> None:
        """
        Plot MCMC trace plots for diagnostics.

        Args:
            save_path: Path to save figure (optional)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        import matplotlib.pyplot as plt

        # Plot trace
        axes = az.plot_trace(
            self.trace_,
            var_names=['transition_matrix'],
            compact=True,
            figsize=(12, 8)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trace plot saved to: {save_path}")

        plt.show()

    def get_results_summary(self) -> str:
        """
        Get detailed summary with uncertainty and diagnostics.

        Returns:
            Formatted string with results
        """
        if not self.is_fitted:
            return f"{self.name} model not fitted yet."

        summary = []
        summary.append(f"\n{'='*70}")
        summary.append(f"{self.name} - Transition Matrix (Posterior Mean)")
        summary.append(f"{'='*70}\n")

        # Transition matrix
        T_mean = self.get_transition_matrix()
        uncertainty = self.get_uncertainty()

        df = pd.DataFrame(
            T_mean,
            index=self.origin_party_names_,
            columns=self.destination_party_names_
        )
        summary.append(df.to_string())

        # Uncertainty
        summary.append(f"\n{'='*70}")
        summary.append("Posterior Standard Deviation:")
        summary.append(f"{'='*70}\n")

        df_std = pd.DataFrame(
            uncertainty['std'],
            index=self.origin_party_names_,
            columns=self.destination_party_names_
        )
        summary.append(df_std.to_string())

        # Credible intervals
        summary.append(f"\n{'='*70}")
        summary.append("95% Credible Intervals:")
        summary.append(f"{'='*70}\n")

        ci = self.get_credible_intervals(0.95)
        for i, origin in enumerate(self.origin_party_names_):
            summary.append(f"\n{origin}:")
            for j, dest in enumerate(self.destination_party_names_):
                mean_val = T_mean[i, j]
                lower = ci['lower'][i, j]
                upper = ci['upper'][i, j]
                summary.append(f"  → {dest}: {mean_val:.4f} [{lower:.4f}, {upper:.4f}]")

        # Diagnostics
        summary.append(f"\n{'='*70}")
        summary.append("MCMC Diagnostics:")
        summary.append(f"{'='*70}")

        diagnostics = self.get_diagnostics()
        summary.append(f"  Max R-hat: {np.max(diagnostics['rhat']):.4f} (should be < 1.01)")
        summary.append(f"  Min ESS: {np.min(diagnostics['ess']):.0f} (should be > 1000)")
        summary.append(f"  Chains: {self.num_chains}")
        summary.append(f"  Samples per chain: {self.num_samples}")

        return "\n".join(summary)


def analyze_coalition_losses(data: pd.DataFrame,
                             model: Optional[KingEI] = None,
                             **model_kwargs) -> Dict:
    """
    Analyze where the coalition (PN+PC+CA) lost votes between elections.

    Args:
        data: Merged electoral data
        model: Pre-fitted KingEI model (optional)
        **model_kwargs: Arguments for KingEI if model not provided

    Returns:
        Dictionary with coalition transfer analysis
    """
    logger.info("\n" + "="*70)
    logger.info("COALITION TRANSFER ANALYSIS (PN + PC + CA → FA vs PN)")
    logger.info("="*70)

    # Define origin (first round) and destination (runoff) columns
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Fit model if not provided
    if model is None:
        logger.info("Fitting King's EI model...")
        model = KingEI(**model_kwargs)
        model.fit(
            data=data,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=True
        )

    # Get transition matrix
    T = model.get_transition_matrix()
    uncertainty = model.get_uncertainty()
    ci_95 = model.get_credible_intervals(0.95)

    # Analyze coalition parties
    coalition_parties = ['CA', 'PC', 'PN']
    coalition_analysis = {}

    for i, party in enumerate(model.origin_party_names_):
        if party in coalition_parties:
            to_fa = T[i, 0]  # → FA
            to_pn = T[i, 1]  # → PN (coalition)
            to_blank = T[i, 2]  # → Blancos

            to_fa_ci = (ci_95['lower'][i, 0], ci_95['upper'][i, 0])
            to_pn_ci = (ci_95['lower'][i, 1], ci_95['upper'][i, 1])

            coalition_analysis[party] = {
                'to_fa': to_fa,
                'to_fa_ci': to_fa_ci,
                'to_pn': to_pn,
                'to_pn_ci': to_pn_ci,
                'to_blank': to_blank,
                'loyalty_to_coalition': to_pn,  # PN represents coalition in runoff
                'defection_to_fa': to_fa
            }

    # Print analysis
    print("\n" + "="*70)
    print("COALITION LOSSES ANALYSIS")
    print("="*70)

    for party, stats in coalition_analysis.items():
        print(f"\n{party} (October) →")
        print(f"  To PN (Coalition loyalty): {stats['to_pn']*100:.2f}% [{stats['to_pn_ci'][0]*100:.2f}%, {stats['to_pn_ci'][1]*100:.2f}%]")
        print(f"  To FA (Defection):        {stats['to_fa']*100:.2f}% [{stats['to_fa_ci'][0]*100:.2f}%, {stats['to_fa_ci'][1]*100:.2f}%]")
        print(f"  To Blancos:               {stats['to_blank']*100:.2f}%")

    return {
        'model': model,
        'transition_matrix': T,
        'uncertainty': uncertainty,
        'credible_intervals': ci_95,
        'coalition_analysis': coalition_analysis
    }


def main():
    """Test King's EI on electoral data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Run King\'s Ecological Inference on electoral data'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged.parquet',
        help='Path to merged data'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of MCMC samples'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Number of MCMC chains'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results (optional)'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    # Run coalition analysis
    results = analyze_coalition_losses(
        data=df,
        num_samples=args.samples,
        num_chains=args.chains
    )

    # Print full results
    print(results['model'].get_results_summary())

    # Save if output path provided
    if args.output:
        results['model'].save_results(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
