# tests/test_bounds.py
import numpy as np
import pandas as pd
import pytest
from src.diagnostics.bounds import compute_duncan_davis_bounds


def test_bounds_shape(synthetic_data_small):
    df = synthetic_data_small
    origin_cols = ['party_a', 'party_b']
    dest_cols = ['dest_x', 'dest_y']
    bounds = compute_duncan_davis_bounds(df, origin_cols, dest_cols, 'total_primera', 'total_ballotage')
    assert set(bounds.keys()) == set(origin_cols)
    for party in origin_cols:
        assert 'lower' in bounds[party]
        assert 'upper' in bounds[party]
        assert bounds[party]['lower'].shape == (len(df), len(dest_cols))
        assert bounds[party]['upper'].shape == (len(df), len(dest_cols))


def test_bounds_valid_range(synthetic_data_small):
    df = synthetic_data_small
    origin_cols = ['party_a', 'party_b']
    dest_cols = ['dest_x', 'dest_y']
    bounds = compute_duncan_davis_bounds(df, origin_cols, dest_cols, 'total_primera', 'total_ballotage')
    for party in origin_cols:
        lower = bounds[party]['lower']
        upper = bounds[party]['upper']
        assert np.all(lower >= 0.0 - 1e-10)
        assert np.all(upper <= 1.0 + 1e-10)
        assert np.all(lower <= upper + 1e-10)


def test_bounds_aggregate():
    """Aggregate bounds: known 2x2 case from King (1997) p.78.
    x=60/100=0.6, t_y=70/100=0.7 → lower = max(0,(0.7-0.4)/0.6) = 0.5
    """
    df = pd.DataFrame({
        'x': [60, 40, 50],
        'one_minus_x': [40, 60, 50],
        't_y': [70, 30, 50],
        'total': [100, 100, 100],
        'total_b': [100, 100, 100],
    })
    bounds = compute_duncan_davis_bounds(df, ['x', 'one_minus_x'], ['t_y'], 'total', 'total_b')
    # For party 'x': lower bound per circuit = max(0, (t_y - (1-x)) / x)
    # circuit 0: max(0, (0.7-0.4)/0.6) = 0.5; upper = min(1, 0.7/0.6) = 1.0
    assert abs(bounds['x']['lower'][0, 0] - 0.5) < 0.01
