"""Perturbation models for robustness evaluation."""

from __future__ import annotations

import numpy as np


def random_edge_failure(
    W: np.ndarray, p: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Independently drop each edge with probability ``p``."""
    if rng is None:
        rng = np.random.default_rng()
    n = W.shape[0]
    mask = rng.random((n, n)) >= p
    mask = np.triu(mask, k=1)
    mask = mask + mask.T
    return W * mask


def node_dropout(
    W: np.ndarray, q: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Zero out rows and columns of ``W`` for nodes dropped with probability ``q``."""
    if rng is None:
        rng = np.random.default_rng()
    n = W.shape[0]
    keep = rng.random(n) >= q
    W_out = W.copy()
    W_out[~keep, :] = 0.0
    W_out[:, ~keep] = 0.0
    return W_out


def bernoulli_packet_loss(
    W: np.ndarray, p_loss: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Same as ``random_edge_failure`` but parameterized by loss probability.

    Kept as a separate name for readability at call sites that model per-step
    transmission failures rather than persistent topology damage.
    """
    return random_edge_failure(W, p_loss, rng=rng)
