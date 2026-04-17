"""Non-learning edge-weight baselines.

Each function takes an adjacency/support matrix ``A`` (1 where an edge is
allowed, 0 otherwise) and returns a weighted symmetric ``W`` on the same
support. Functions that accept a ``budget`` rescale to ``sum_{i<j} W_ij == budget``.
"""

from __future__ import annotations

import numpy as np


def _apply_budget(W: np.ndarray, budget: float | None, w_max: float) -> np.ndarray:
    W = np.clip(W, 0.0, w_max)
    if budget is None:
        return W
    total = float(np.triu(W, k=1).sum())
    if total <= 0:
        return W
    scale = min(1.0, budget / total)
    return W * scale


def uniform_weights(
    A: np.ndarray, budget: float | None = None, w_max: float = 1.0
) -> np.ndarray:
    support = (A > 0).astype(np.float64)
    np.fill_diagonal(support, 0.0)
    m = float(np.triu(support, k=1).sum())
    if m == 0:
        return support
    if budget is None:
        w = min(w_max, 1.0)
    else:
        w = min(w_max, budget / m)
    W = support * w
    return W


def metropolis_weights(A: np.ndarray) -> np.ndarray:
    """Metropolis-Hastings weights: ``w_ij = 1 / (1 + max(d_i, d_j))``.

    Gives a doubly-stochastic ``I - L`` when the graph is connected. Self-loops
    are not set here (the caller can add them if a Markov matrix is needed).
    """
    support = (A > 0).astype(np.float64)
    np.fill_diagonal(support, 0.0)
    d = support.sum(axis=1)
    n = support.shape[0]
    W = np.zeros_like(support)
    for i in range(n):
        for j in range(i + 1, n):
            if support[i, j] > 0:
                w = 1.0 / (1.0 + max(d[i], d[j]))
                W[i, j] = W[j, i] = w
    return W


def degree_proportional_weights(
    A: np.ndarray, budget: float, w_max: float = 1.0
) -> np.ndarray:
    """Allocate more weight to edges whose endpoints have higher combined degree.

    This is a simple heuristic meant to stress bottleneck edges.
    """
    support = (A > 0).astype(np.float64)
    np.fill_diagonal(support, 0.0)
    d = support.sum(axis=1)
    U = np.triu(support, k=1)
    scores = np.zeros_like(U)
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if U[i, j] > 0:
                scores[i, j] = d[i] + d[j]
    total = float(scores.sum())
    if total == 0:
        return _apply_budget(support, budget, w_max)
    W_upper = budget * scores / total
    W_upper = np.clip(W_upper, 0.0, w_max)
    W = W_upper + W_upper.T
    return _apply_budget(W, budget, w_max)
