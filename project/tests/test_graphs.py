import numpy as np
import pytest

from spectralrl.graphs import (
    complete,
    erdos_renyi,
    fiedler_value,
    grid,
    is_connected,
    laplacian,
    random_geometric,
    ring,
    watts_strogatz,
)


def _is_symmetric_nonneg_zero_diag(W):
    assert np.allclose(W, W.T)
    assert np.all(W >= 0)
    assert np.allclose(np.diag(W), 0)


def test_ring_spectrum_matches_closed_form():
    for n in [8, 12, 20]:
        W, _ = ring(n)
        _is_symmetric_nonneg_zero_diag(W)
        lam2 = fiedler_value(laplacian(W))
        expected = 2 * (1 - np.cos(2 * np.pi / n))
        assert lam2 == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_complete_graph_spectrum():
    for n in [4, 8, 16]:
        W, _ = complete(n)
        lam2 = fiedler_value(laplacian(W))
        # Complete graph K_n has all nonzero eigenvalues equal to n.
        assert lam2 == pytest.approx(float(n), rel=1e-6)


def test_generators_symmetric_and_connected():
    specs = [
        ("ring", lambda: ring(20)[0]),
        ("grid", lambda: grid(4, 5)[0]),
        ("erdos_renyi", lambda: erdos_renyi(20, 0.2, seed=1)[0]),
        ("random_geometric", lambda: random_geometric(20, 0.35, seed=2)[0]),
        ("watts_strogatz", lambda: watts_strogatz(20, 4, 0.1, seed=3)[0]),
        ("complete", lambda: complete(6)[0]),
    ]
    for name, make in specs:
        W = make()
        _is_symmetric_nonneg_zero_diag(W)
        assert is_connected(W), f"{name} should be connected"


def test_fiedler_nonnegative_and_small_for_sparse():
    W, _ = ring(50)
    lam2 = fiedler_value(laplacian(W))
    assert lam2 > 0
    W_c, _ = complete(50)
    assert fiedler_value(laplacian(W_c)) > lam2
