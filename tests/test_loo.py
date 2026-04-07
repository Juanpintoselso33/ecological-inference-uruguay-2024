# tests/test_loo.py
import numpy as np
import pytest
from src.diagnostics.loo import compute_loo, LOOResult


def test_loo_result_type():
    """LOOResult is a dataclass with expected fields."""
    r = LOOResult(elpd_loo=-100.0, p_loo=5.0, looic=200.0,
                  se=10.0, pareto_k=np.array([0.3, 0.5, 0.7]),
                  n_bad_k=1, warning=True)
    assert r.n_bad_k == 1
    assert r.warning is True


def test_loo_requires_fitted_model(synthetic_data_small, quick_mcmc_params):
    """compute_loo raises ValueError if model not fitted."""
    from src.models.king_ei import KingEI
    model = KingEI(**quick_mcmc_params)
    with pytest.raises(ValueError, match="not fitted"):
        compute_loo(model)


def test_loo_returns_result(synthetic_data_small, quick_mcmc_params):
    """compute_loo returns LOOResult with correct shapes after fitting."""
    from src.models.king_ei import KingEI
    df = synthetic_data_small
    model = KingEI(**quick_mcmc_params)
    model.fit(df, origin_cols=['party_a', 'party_b'],
              destination_cols=['dest_x', 'dest_y'],
              total_origin='total_primera', total_destination='total_ballotage')
    result = compute_loo(model)
    assert isinstance(result, LOOResult)
    assert result.pareto_k.shape == (len(df),)
    assert isinstance(result.elpd_loo, float)
    assert isinstance(result.n_bad_k, int)
