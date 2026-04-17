import numpy as np
import pytest

from spectralrl.baselines import (
    degree_proportional_weights,
    metropolis_weights,
    uniform_weights,
)
from spectralrl.graphs import erdos_renyi, ring


def test_uniform_respects_budget():
    W, _ = erdos_renyi(15, 0.3, seed=0)
    budget = 3.0
    Wu = uniform_weights(W, budget=budget, w_max=1.0)
    assert np.triu(Wu, k=1).sum() <= budget + 1e-9


def test_metropolis_row_sums_below_one():
    W, _ = ring(12)
    Wm = metropolis_weights(W)
    # Sum of neighbor weights per row should be strictly less than 1
    row_sums = Wm.sum(axis=1)
    assert np.all(row_sums < 1.0 + 1e-12)
    assert np.allclose(Wm, Wm.T)


def test_degree_proportional_respects_budget_and_bounds():
    W, _ = erdos_renyi(20, 0.25, seed=1)
    budget = 5.0
    Wd = degree_proportional_weights(W, budget=budget, w_max=1.0)
    assert np.triu(Wd, k=1).sum() <= budget + 1e-9
    assert Wd.max() <= 1.0 + 1e-9
    assert Wd.min() >= 0.0
