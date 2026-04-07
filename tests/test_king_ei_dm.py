# tests/test_king_ei_dm.py
"""Tests for KingEI with DirichletMultinomial likelihood."""
import numpy as np
import pandas as pd
import pytest
from src.models.king_ei import KingEI


def test_dm_fit_runs(synthetic_data_small, quick_mcmc_params):
    """DirichletMultinomial model fits without error."""
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params, likelihood='dirichlet_multinomial')
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    assert model.is_fitted


def test_dm_transition_matrix_shape(synthetic_data_small, quick_mcmc_params):
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params, likelihood='dirichlet_multinomial')
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    T = model.get_transition_matrix()
    assert T.shape == (2, 2)
    assert np.allclose(T.sum(axis=1), 1.0, atol=0.1)


def test_dm_recovers_true_T(synthetic_data_small, quick_mcmc_params):
    """DM model recovers T_true = [[0.8,0.2],[0.3,0.7]] within 0.15."""
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params, likelihood='dirichlet_multinomial')
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    T = model.get_transition_matrix()
    T_true = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert np.allclose(T, T_true, atol=0.15), f"Got T={T}, expected ~{T_true}"


def test_dm_invalid_likelihood_raises():
    model = KingEI(likelihood='invalid')
    df = pd.DataFrame({
        'a': [10], 'b': [10], 'c': [10], 'd': [10], 'tot': [20], 'tot2': [20]
    })
    with pytest.raises(ValueError, match="likelihood"):
        model.fit(df, ['a', 'b'], ['c', 'd'], 'tot', 'tot2')


def test_dm_has_concentration_parameter(synthetic_data_small, quick_mcmc_params):
    """DirichletMultinomial trace must contain 'concentration' parameter."""
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params, likelihood='dirichlet_multinomial')
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    assert 'concentration' in model.trace_.posterior


def test_covariates_urbanrural(synthetic_data_small, quick_mcmc_params):
    """Model accepts covariate column and runs without error."""
    df = synthetic_data_small.copy()
    df['urban'] = ([1.0] * 25 + [0.0] * 25)
    model = KingEI(**quick_mcmc_params, likelihood='dirichlet_multinomial')
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage',
              covariate_cols=['urban'])
    T = model.get_transition_matrix()
    assert T.shape == (2, 2)
