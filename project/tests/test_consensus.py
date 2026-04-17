import numpy as np
import pytest

from spectralrl.consensus import (
    convergence_time,
    disagreement_energy,
    rate_estimate,
    run_consensus,
    stable_step_size,
)
from spectralrl.graphs import complete, laplacian, ring


def test_consensus_converges_on_ring():
    W, _ = ring(12)
    L = laplacian(W)
    alpha = stable_step_size(L)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(12)
    traj = run_consensus(L, x0, alpha, T=800)
    J = disagreement_energy(traj)
    assert J[-1] / J[0] < 1e-4


def test_average_preserved_undirected():
    W, _ = complete(10)
    L = laplacian(W)
    rng = np.random.default_rng(1)
    x0 = rng.standard_normal(10)
    mean0 = x0.mean()
    alpha = stable_step_size(L)
    traj = run_consensus(L, x0, alpha, T=50)
    assert np.allclose(traj.mean(axis=1), mean0, atol=1e-10)


def test_convergence_time_monotone_in_lambda2():
    W_ring, _ = ring(20)
    W_complete, _ = complete(20)
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal(20)
    x0 -= x0.mean()

    def sim(W):
        L = laplacian(W)
        alpha = stable_step_size(L)
        traj = run_consensus(L, x0, alpha, T=500)
        return convergence_time(traj, eps=1e-3)

    tau_ring = sim(W_ring)
    tau_complete = sim(W_complete)
    assert tau_complete <= tau_ring


def test_rate_estimate_bounded():
    W, _ = complete(8)
    L = laplacian(W)
    alpha = stable_step_size(L)
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal(8)
    x0 -= x0.mean()
    traj = run_consensus(L, x0, alpha, T=100)
    rho = rate_estimate(traj)
    assert 0.0 <= rho < 1.0


def test_stable_step_size_respects_bound():
    W, _ = complete(5)
    L = laplacian(W)
    alpha = stable_step_size(L)
    from scipy.linalg import eigh

    lam_max = eigh(L, eigvals_only=True)[-1]
    # alpha must be strictly below 2 / lam_max for contractivity
    assert alpha * lam_max < 2.0
