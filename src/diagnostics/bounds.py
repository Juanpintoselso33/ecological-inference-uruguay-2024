# src/diagnostics/bounds.py
"""Duncan-Davis deterministic bounds on transition probabilities per circuit.

For each circuit i and origin party k, computes lower and upper bounds on
the transition proportion beta_{k,j} (fraction of k-voters going to destination j).

Reference: King (1997), Chapter 5; Duncan & Davis (1953).
"""
import numpy as np
import pandas as pd


def compute_duncan_davis_bounds(
    df: pd.DataFrame,
    origin_cols: list,
    dest_cols: list,
    total_primera_col: str,
    total_ballotage_col: str,
) -> dict:
    """Compute per-circuit Duncan-Davis bounds.

    Parameters
    ----------
    df : DataFrame with circuit-level data
    origin_cols : column names for origin party vote counts
    dest_cols : column names for destination vote counts
    total_primera_col : column name for total primera vuelta votes
    total_ballotage_col : column name for total ballotage votes

    Returns
    -------
    dict mapping each origin party name to {'lower': array(n,d), 'upper': array(n,d)}
    where n = number of circuits, d = number of destinations
    """
    n = len(df)
    d = len(dest_cols)

    N1 = df[total_primera_col].values.astype(float)
    N2 = df[total_ballotage_col].values.astype(float)

    X = np.column_stack([df[c].values for c in origin_cols]).astype(float)
    X_prop = X / np.maximum(N1[:, None], 1)

    Y = np.column_stack([df[c].values for c in dest_cols]).astype(float)
    Y_prop = Y / np.maximum(N2[:, None], 1)

    bounds = {}
    for idx, party in enumerate(origin_cols):
        x = X_prop[:, idx]  # shape (n,)
        lower = np.zeros((n, d))
        upper = np.ones((n, d))

        for j in range(d):
            t = Y_prop[:, j]  # aggregate outcome proportion
            safe_x = np.where(x > 1e-10, x, np.nan)
            other = 1.0 - x
            lb = np.where(x > 1e-10,
                          np.maximum(0.0, (t - other) / safe_x),
                          0.0)
            ub = np.where(x > 1e-10,
                          np.minimum(1.0, t / safe_x),
                          1.0)
            lower[:, j] = np.nan_to_num(lb, nan=0.0)
            upper[:, j] = np.nan_to_num(ub, nan=1.0)

        bounds[party] = {'lower': lower, 'upper': upper}

    return bounds


def bounds_to_dataframe(
    bounds: dict,
    origin_cols: list,
    dest_cols: list,
) -> pd.DataFrame:
    """Convert bounds dict to a tidy DataFrame for reporting."""
    rows = []
    for party in origin_cols:
        lower = bounds[party]['lower']
        upper = bounds[party]['upper']
        for j, dest in enumerate(dest_cols):
            rows.append({
                'origin': party,
                'destination': dest,
                'lower_mean': lower[:, j].mean(),
                'upper_mean': upper[:, j].mean(),
                'lower_min': lower[:, j].min(),
                'upper_max': upper[:, j].max(),
            })
    return pd.DataFrame(rows)
