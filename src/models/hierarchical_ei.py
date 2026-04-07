# src/models/hierarchical_ei.py
"""Hierarchical Bayesian ecological inference model.

Uses partial pooling across geographic groups (departments): each group
gets its own transition matrix T_g, drawn from a shared Dirichlet prior
whose concentration hyperparameter is learned from data.

This implements the partial-pooling structure described in King (1997) and
Rosen et al. (2001), in contrast to the complete-pooling KingEI model
which estimates a single national T.
"""
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_ei import BaseEIModel
from src.utils import get_logger

logger = get_logger(__name__)


class HierarchicalEI(BaseEIModel):
    """Hierarchical EI with partial pooling across geographic groups.

    Each group (e.g., department) gets its own transition matrix, drawn
    from a shared Dirichlet hyperprior. This captures geographic variation
    while sharing statistical strength across groups.

    Parameters
    ----------
    num_samples : int, default 2000
    num_chains : int, default 4
    num_warmup : int, default 1000
    target_accept : float, default 0.9
    random_seed : int, default 42
    """

    def __init__(
        self,
        num_samples: int = 2000,
        num_chains: int = 4,
        num_warmup: int = 1000,
        target_accept: float = 0.9,
        random_seed: int = 42,
        trace_dir: str = 'outputs/results/traces',
    ):
        super().__init__(name="HierarchicalEI")
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.trace_path_ = None

    def fit(
        self,
        data: pd.DataFrame,
        origin_cols: List[str],
        destination_cols: List[str],
        total_origin: str,
        total_destination: str,
        group_col: str = 'departamento',
        progressbar: bool = True,
    ) -> 'HierarchicalEI':
        """Fit hierarchical EI model with partial pooling across groups.

        Parameters
        ----------
        data : DataFrame with circuit-level data
        origin_cols : origin party vote count columns
        destination_cols : destination vote count columns
        total_origin : total primera vuelta column
        total_destination : total ballotage column
        group_col : column with group labels (e.g., 'departamento')
        progressbar : show MCMC progress bar
        """
        self.validate_inputs(data, origin_cols, destination_cols, total_origin, total_destination)

        self.origin_cols_ = origin_cols
        self.destination_cols_ = destination_cols
        self.group_col_ = group_col

        all_cols = origin_cols + destination_cols + [total_origin, total_destination, group_col]
        data_clean = data[all_cols].dropna()
        logger.info(f"Fitting HierarchicalEI on {len(data_clean)} circuits")

        groups = data_clean[group_col].unique()
        self.groups_ = groups
        group_map = {g: i for i, g in enumerate(groups)}
        group_idx = data_clean[group_col].map(group_map).values

        n_groups = len(groups)
        n_origin = len(origin_cols)
        n_dest = len(destination_cols)

        N1 = data_clean[total_origin].values.astype(float)
        X = data_clean[origin_cols].values.astype(float)
        X_prop = X / np.maximum(N1[:, None], 1)

        N2 = data_clean[total_destination].values.astype(int)
        Y = data_clean[destination_cols].values.astype(int)

        logger.info(f"Building hierarchical model: {n_origin} origins, {n_dest} dests, "
                    f"{n_groups} groups")

        with pm.Model() as model:
            # Hyperprior: shared Dirichlet concentration for each row of T
            alpha_hyper = pm.HalfNormal('alpha_hyper', sigma=10.0, shape=n_dest)

            # Group-level transition matrices via partial pooling
            # T_g[g, i, j] = P(dest j | origin i, group g)
            T_g = pm.Dirichlet(
                'transition_matrix_groups',
                a=alpha_hyper,
                shape=(n_groups, n_origin, n_dest),
            )

            # Index into group-level T for each circuit
            T_circuit = T_g[group_idx]  # (n_circuits, n_origin, n_dest)

            # Predicted destination proportions: batched matmul
            # X_prop[c] @ T_circuit[c] for each circuit
            pi = pm.math.sum(X_prop[:, :, None] * T_circuit, axis=1)  # (n_circuits, n_dest)

            concentration = pm.HalfNormal('concentration', sigma=100.0)
            alpha_pred = pi * concentration

            Y_obs = pm.DirichletMultinomial(
                'destination_votes',
                n=N2,
                a=alpha_pred,
                observed=Y,
            )

        logger.info(f"Starting MCMC: {self.num_chains} chains, {self.num_samples} draws")
        with model:
            trace = pm.sample(
                draws=self.num_samples,
                chains=self.num_chains,
                tune=self.num_warmup,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=progressbar,
                return_inferencedata=True,
            )

        self.model_ = model
        self.trace_ = trace
        self.data_ = data.copy()
        self.is_fitted = True

        # Auto-save trace before diagnostics
        self._save_trace(origin_cols, destination_cols, n_groups)
        self._compute_diagnostics()
        logger.info("HierarchicalEI fitted successfully")
        return self

    def _save_trace(self, origin_cols, destination_cols, n_groups):
        """Save InferenceData trace to pickle."""
        if self.trace_dir is None:
            return
        try:
            import pickle
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            fname = (f"hierarchical_{len(origin_cols)}x{len(destination_cols)}"
                     f"_{n_groups}g_{self.num_samples}s_{self.num_chains}c.pkl")
            self.trace_path_ = self.trace_dir / fname
            with open(self.trace_path_, 'wb') as f:
                pickle.dump(self.trace_, f)
            logger.info("Trace guardado: %s", self.trace_path_)
        except Exception as e:
            logger.warning("No se pudo guardar el trace: %s", e)

    @classmethod
    def load_trace(cls, trace_path: str, **kwargs) -> 'HierarchicalEI':
        """Load a previously saved trace without re-running MCMC."""
        import pickle
        model = cls(**kwargs)
        with open(trace_path, 'rb') as f:
            model.trace_ = pickle.load(f)
        model.trace_path_ = Path(trace_path)
        model.is_fitted = True
        model._compute_diagnostics()
        logger.info("Trace cargado desde: %s", trace_path)
        return model

    def get_transition_matrix(self) -> np.ndarray:
        """Returns national-level T (mean over groups and posterior)."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")
        T_group = self.trace_.posterior['transition_matrix_groups']
        # Mean over groups dimension (dim 0 of the variable) then chains/draws
        return T_group.mean(dim=['chain', 'draw', 'transition_matrix_groups_dim_0']).values

    def get_group_transition_matrices(self) -> Dict[str, np.ndarray]:
        """Returns T per group: dict mapping group name -> (n_origin, n_dest) array."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")
        T_group = self.trace_.posterior['transition_matrix_groups']
        T_mean = T_group.mean(dim=['chain', 'draw']).values  # (n_groups, n_origin, n_dest)
        return {g: T_mean[i] for i, g in enumerate(self.groups_)}

    def get_uncertainty(self, prob: float = 0.95) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")
        alpha = 1 - prob
        T = self.trace_.posterior['transition_matrix_groups'].mean(
            dim='transition_matrix_groups_dim_0'
        )
        lower = T.quantile(alpha / 2, dim=['chain', 'draw']).values
        upper = T.quantile(1 - alpha / 2, dim=['chain', 'draw']).values
        return {'lower': lower, 'upper': upper}

    def _compute_diagnostics(self) -> None:
        """Compute MCMC diagnostics for group-level transition matrices."""
        rhat = az.rhat(self.trace_, var_names=['transition_matrix_groups'])
        ess = az.ess(self.trace_, var_names=['transition_matrix_groups'])
        self.rhat_ = float(rhat['transition_matrix_groups'].values.max())
        self.ess_ = float(ess['transition_matrix_groups'].values.min())
        if self.rhat_ > 1.01:
            logger.warning(f"Poor convergence: max R-hat = {self.rhat_:.4f}")
        else:
            logger.info(f"Convergence OK: max R-hat = {self.rhat_:.4f}")
