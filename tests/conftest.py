# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_data_small():
    """50 circuits, 2 origin parties, 2 destinations. T_true = [[0.8,0.2],[0.3,0.7]]."""
    rng = np.random.default_rng(42)
    n_circuits = 50
    total = rng.integers(200, 600, size=n_circuits)
    # Origin split: party A gets 40-60% of votes
    prop_a = rng.uniform(0.4, 0.6, size=n_circuits)
    x_a = (total * prop_a).astype(int)
    x_b = total - x_a
    T_true = np.array([[0.8, 0.2], [0.3, 0.7]])
    # Destination ballotage totals
    n_ballotage = total + rng.integers(-20, 20, size=n_circuits)
    n_ballotage = np.maximum(n_ballotage, 50)
    # Expected proportions
    prop_dest = (x_a[:, None] * T_true[0] + x_b[:, None] * T_true[1]) / total[:, None]
    y_raw = rng.multinomial(1, prop_dest[0].clip(0, 1))  # placeholder
    y_raw = np.array([rng.multinomial(n_ballotage[i], prop_dest[i].clip(0, 1)) for i in range(n_circuits)])
    df = pd.DataFrame({
        'party_a': x_a,
        'party_b': x_b,
        'dest_x': y_raw[:, 0],
        'dest_y': y_raw[:, 1],
        'total_primera': total,
        'total_ballotage': n_ballotage,
        'departamento': ['Montevideo'] * 25 + ['Canelones'] * 25,
    })
    return df


@pytest.fixture
def synthetic_data_electoral():
    """200 circuits, 6 origin parties (ca, fa, pc, pn, pi, otros), 3 destinations."""
    rng = np.random.default_rng(99)
    n = 200
    total = rng.integers(300, 800, size=n)
    # Proportions for 6 parties summing to 1
    raw = rng.dirichlet(np.ones(6) * 2, size=n)
    counts = (raw * total[:, None]).astype(int)
    counts[:, -1] = total - counts[:, :-1].sum(axis=1)  # fix rounding
    counts = np.maximum(counts, 0)
    T_true = np.array([
        [0.45, 0.50, 0.05],  # ca
        [0.95, 0.03, 0.02],  # fa
        [0.10, 0.85, 0.05],  # pc
        [0.05, 0.90, 0.05],  # pn
        [0.30, 0.60, 0.10],  # pi
        [0.40, 0.40, 0.20],  # otros
    ])
    n_bal = total + rng.integers(-30, 30, size=n)
    n_bal = np.maximum(n_bal, 100)
    prop_dest = (counts / total[:, None]) @ T_true
    prop_dest = prop_dest / prop_dest.sum(axis=1, keepdims=True)
    y_raw = np.array([rng.multinomial(n_bal[i], prop_dest[i]) for i in range(n)])
    cols_origin = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    cols_dest = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
    df = pd.DataFrame(counts, columns=cols_origin)
    for j, c in enumerate(cols_dest):
        df[c] = y_raw[:, j]
    df['total_primera'] = total
    df['total_ballotage'] = n_bal
    df['departamento'] = rng.choice(['Montevideo', 'Canelones', 'Rivera', 'Salto'], size=n)
    return df


@pytest.fixture
def quick_mcmc_params():
    return dict(num_samples=100, num_chains=2, num_warmup=50, random_seed=42)
