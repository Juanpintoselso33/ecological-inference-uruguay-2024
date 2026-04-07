# tests/test_leverage.py
import numpy as np
import pandas as pd
import pytest
from src.diagnostics.leverage import compute_circuit_leverage


def test_leverage_shape(synthetic_data_small):
    df = synthetic_data_small
    origin_cols = ['party_a', 'party_b']
    result = compute_circuit_leverage(df, origin_cols, 'total_primera')
    assert 'leverage' in result.columns
    assert len(result) == len(df)


def test_leverage_range(synthetic_data_small):
    df = synthetic_data_small
    origin_cols = ['party_a', 'party_b']
    result = compute_circuit_leverage(df, origin_cols, 'total_primera')
    assert result['leverage'].between(0, 1 + 1e-10).all()


def test_leverage_high_for_extreme_x():
    """Circuit with extreme x (near 0 or 1) should have high leverage."""
    df = pd.DataFrame({
        'party_a': [500, 10, 250],
        'party_b': [10, 500, 250],
        'total': [510, 510, 500],
    })
    result = compute_circuit_leverage(df, ['party_a', 'party_b'], 'total')
    # First two circuits are more extreme — higher leverage than the centered one
    assert result['leverage'].iloc[0] > result['leverage'].iloc[2]
    assert result['leverage'].iloc[1] > result['leverage'].iloc[2]
