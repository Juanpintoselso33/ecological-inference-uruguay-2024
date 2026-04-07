# tests/test_king_ei_api.py
"""Smoke tests: existing KingEI public API must be unchanged."""
import pytest
import numpy as np
from src.models.king_ei import KingEI


def test_default_instantiation():
    model = KingEI()
    assert model.num_samples == 2000
    assert model.num_chains == 4
    assert not model.is_fitted


def test_likelihood_default_is_normal():
    """Default likelihood='normal' preserves backward compatibility."""
    model = KingEI()
    assert model.likelihood == 'normal'


def test_fit_returns_self(synthetic_data_small, quick_mcmc_params):
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params)
    result = model.fit(df, origin_cols=['party_a', 'party_b'],
                       destination_cols=['dest_x', 'dest_y'],
                       total_origin='total_primera', total_destination='total_ballotage')
    assert result is model
    assert model.is_fitted


def test_get_transition_matrix_shape(synthetic_data_small, quick_mcmc_params):
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params)
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    T = model.get_transition_matrix()
    assert T.shape == (2, 2)
    assert np.allclose(T.sum(axis=1), 1.0, atol=0.05)


def test_get_credible_intervals_shape(synthetic_data_small, quick_mcmc_params):
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params)
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    ci = model.get_credible_intervals(prob=0.95)
    assert 'lower' in ci and 'upper' in ci
    assert ci['lower'].shape == (2, 2)
    assert np.all(ci['lower'] <= ci['upper'])


def test_not_fitted_raises():
    model = KingEI()
    with pytest.raises(ValueError, match="not fitted"):
        model.get_transition_matrix()
