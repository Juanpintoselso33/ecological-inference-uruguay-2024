# src/diagnostics/leverage.py
"""Circuit-level leverage and influence diagnostics for ecological inference.

Leverage measures how much each circuit's X-values (origin party proportions)
deviate from the mean — circuits far from the mean exert more influence on
the global transition matrix estimate.

Reference: King (1997), pp. 101-105.
"""
import numpy as np
import pandas as pd


def compute_circuit_leverage(
    df: pd.DataFrame,
    origin_cols: list,
    total_col: str,
) -> pd.DataFrame:
    """Compute hat-matrix diagonal (leverage) for each circuit.

    Uses the standard OLS hat matrix: h_ii = x_i^T (X^T X)^{-1} x_i
    where x_i is the vector of origin party proportions for circuit i.

    Parameters
    ----------
    df : DataFrame with circuit data
    origin_cols : origin party vote count columns
    total_col : column for total primera vuelta votes

    Returns
    -------
    DataFrame with columns: 'leverage', 'high_leverage' (bool, > 2*mean)
    """
    total = df[total_col].values.astype(float)
    X = np.column_stack([df[c].values for c in origin_cols]).astype(float)
    X_prop = X / np.maximum(total[:, None], 1)

    # Add intercept
    X_design = np.column_stack([np.ones(len(df)), X_prop])

    XtX = X_design.T @ X_design
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX + 1e-8 * np.eye(XtX.shape[0]))

    # h_ii = x_i (X^T X)^{-1} x_i^T
    leverage = np.einsum('ij,jk,ik->i', X_design, XtX_inv, X_design)
    leverage = np.clip(leverage, 0.0, 1.0)

    mean_lev = leverage.mean()
    result = df[[]].copy()
    result['leverage'] = leverage
    result['high_leverage'] = leverage > 2 * mean_lev
    return result


def compute_dfbeta(
    df: pd.DataFrame,
    origin_cols: list,
    dest_col: str,
    total_col: str,
) -> pd.DataFrame:
    """Compute DFBETA (change in coefficient when circuit i is removed).

    Returns DataFrame with 'dfbeta_<origin>' columns for each origin party.
    """
    total = df[total_col].values.astype(float)
    X = np.column_stack([df[c].values for c in origin_cols]).astype(float)
    X_prop = X / np.maximum(total[:, None], 1)
    y = df[dest_col].values.astype(float) / np.maximum(total, 1)

    X_design = np.column_stack([np.ones(len(df)), X_prop])
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.pinv(XtX)
    beta_full = XtX_inv @ X_design.T @ y

    n = len(df)
    k = X_design.shape[1]
    dfbeta = np.zeros((n, k))

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        Xi = X_design[mask]
        yi = y[mask]
        XtXi = Xi.T @ Xi
        beta_i = np.linalg.pinv(XtXi) @ Xi.T @ yi
        dfbeta[i] = beta_full - beta_i

    cols = ['dfbeta_intercept'] + [f'dfbeta_{c}' for c in origin_cols]
    return pd.DataFrame(dfbeta, columns=cols, index=df.index)
