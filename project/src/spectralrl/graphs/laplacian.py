"""Graph Laplacian utilities."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def laplacian(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    return np.diag(d) - W


def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    d = W.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt


def fiedler_value(L: np.ndarray) -> float:
    """Return the second-smallest eigenvalue of L (algebraic connectivity).

    Uses a dense symmetric eigensolver; robust and fast for n up to ~500.
    """
    L = np.asarray(L, dtype=np.float64)
    L_sym = 0.5 * (L + L.T)
    vals = eigh(L_sym, eigvals_only=True, subset_by_index=[0, 1])
    return float(vals[1])


def is_connected(W: np.ndarray) -> bool:
    n = W.shape[0]
    if n == 0:
        return True
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    adj = [np.flatnonzero(W[i] > 0) for i in range(n)]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(int(v))
    return bool(seen.all())
