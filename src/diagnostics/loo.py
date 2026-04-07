# src/diagnostics/loo.py
"""PSIS/LOO-CV diagnostics for KingEI models.

Computes Pareto-Smoothed Importance Sampling Leave-One-Out
cross-validation using ArviZ. Pareto k values > 0.7 indicate
circuits that are highly influential; k >= 1.0 means unreliable.

Reference: Vehtari et al. (2017) "Practical Bayesian model evaluation
using leave-one-out cross-validation and WAIC."
"""
from dataclasses import dataclass
import numpy as np
import arviz as az


@dataclass
class LOOResult:
    elpd_loo: float       # Expected log pointwise predictive density
    p_loo: float          # Effective number of parameters
    looic: float          # LOO information criterion = -2 * elpd_loo
    se: float             # Standard error of elpd_loo
    pareto_k: np.ndarray  # Per-observation Pareto k values, shape (n_circuits,)
    n_bad_k: int          # Count of circuits with k > 0.7
    warning: bool         # True if any k > 0.7


def compute_loo(model) -> LOOResult:
    """Compute PSIS/LOO-CV for a fitted KingEI model.

    Parameters
    ----------
    model : KingEI instance (must be fitted)

    Returns
    -------
    LOOResult with elpd, p_loo, looic, se, pareto_k, n_bad_k, warning
    """
    if not model.is_fitted:
        raise ValueError("Model is not fitted. Call fit() before compute_loo().")

    idata = model.trace_
    loo = az.loo(idata, pointwise=True)

    pareto_k = loo.pareto_k.values if hasattr(loo.pareto_k, 'values') else np.array(loo.pareto_k)
    n_bad_k = int((pareto_k > 0.7).sum())

    return LOOResult(
        elpd_loo=float(loo.elpd_loo),
        p_loo=float(loo.p_loo),
        looic=float(-2 * loo.elpd_loo),
        se=float(loo.se),
        pareto_k=pareto_k,
        n_bad_k=n_bad_k,
        warning=n_bad_k > 0,
    )


def loo_summary(result: LOOResult) -> str:
    """Return a human-readable LOO summary string."""
    lines = [
        "PSIS-LOO-CV Summary",
        f"  ELPD LOO:  {result.elpd_loo:.2f} +/- {result.se:.2f}",
        f"  p_LOO:     {result.p_loo:.2f}",
        f"  LOOIC:     {result.looic:.2f}",
        f"  Bad k (>0.7): {result.n_bad_k} circuits",
    ]
    if result.warning:
        lines.append("  WARNING: Some Pareto k values > 0.7 -- LOO estimates may be unreliable.")
    return "\n".join(lines)
