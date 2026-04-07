# tests/test_hierarchical_ei.py
import pytest
import numpy as np
from src.models.hierarchical_ei import HierarchicalEI


def test_hierarchical_instantiation():
    model = HierarchicalEI()
    assert not model.is_fitted


def test_hierarchical_fit_runs(synthetic_data_electoral, quick_mcmc_params):
    df = synthetic_data_electoral
    origin_cols = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    dest_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
    model = HierarchicalEI(**quick_mcmc_params)
    model.fit(df, origin_cols=origin_cols, destination_cols=dest_cols,
              total_origin='total_primera', total_destination='total_ballotage',
              group_col='departamento')
    assert model.is_fitted


def test_hierarchical_transition_matrix_shape(synthetic_data_electoral, quick_mcmc_params):
    df = synthetic_data_electoral
    origin_cols = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    dest_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
    model = HierarchicalEI(**quick_mcmc_params)
    model.fit(df, origin_cols=origin_cols, destination_cols=dest_cols,
              total_origin='total_primera', total_destination='total_ballotage',
              group_col='departamento')
    T = model.get_transition_matrix()
    assert T.shape == (6, 3)
    assert np.allclose(T.sum(axis=1), 1.0, atol=0.1)
