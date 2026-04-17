"""Graph family generators returning (W, positions).

All returned weight matrices ``W`` are symmetric, zero on the diagonal, and
non-negative. ``positions`` is an ``(n, 2)`` array for geometric graphs and
``None`` otherwise.
"""

from __future__ import annotations

import numpy as np


def _empty(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def ring(n: int) -> tuple[np.ndarray, None]:
    if n < 3:
        raise ValueError("ring requires n >= 3")
    W = _empty(n)
    for i in range(n):
        j = (i + 1) % n
        W[i, j] = W[j, i] = 1.0
    return W, None


def complete(n: int) -> tuple[np.ndarray, None]:
    W = np.ones((n, n), dtype=np.float64) - np.eye(n)
    return W, None


def grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    n = rows * cols
    W = _empty(n)
    pos = np.zeros((n, 2), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            pos[idx] = (c / max(cols - 1, 1), r / max(rows - 1, 1))
            if c + 1 < cols:
                j = r * cols + (c + 1)
                W[idx, j] = W[j, idx] = 1.0
            if r + 1 < rows:
                j = (r + 1) * cols + c
                W[idx, j] = W[j, idx] = 1.0
    return W, pos


def erdos_renyi(
    n: int, p: float, seed: int | None = None, ensure_connected: bool = True
) -> tuple[np.ndarray, None]:
    from .laplacian import is_connected

    rng = np.random.default_rng(seed)
    for attempt in range(32):
        W = _empty(n)
        mask = rng.random((n, n)) < p
        mask = np.triu(mask, k=1)
        W[mask] = 1.0
        W = W + W.T
        if not ensure_connected or is_connected(W):
            return W, None
    # Fall through: graft a spanning path to guarantee connectivity.
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W, None


def random_geometric(
    n: int, radius: float, seed: int | None = None, ensure_connected: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    for attempt in range(32):
        pos = rng.random((n, 2))
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        W = (d <= radius).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        from .laplacian import is_connected

        if not ensure_connected or is_connected(W):
            return W, pos
    # Grow a minimum-spanning-style path by nearest neighbors to force connectivity.
    order = np.argsort(pos[:, 0])
    for a, b in zip(order[:-1], order[1:]):
        W[a, b] = W[b, a] = 1.0
    return W, pos


def watts_strogatz(
    n: int, k: int, p: float, seed: int | None = None
) -> tuple[np.ndarray, None]:
    if k % 2 != 0:
        raise ValueError("watts_strogatz requires even k")
    if k >= n:
        raise ValueError("watts_strogatz requires k < n")
    rng = np.random.default_rng(seed)
    W = _empty(n)
    for i in range(n):
        for step in range(1, k // 2 + 1):
            j = (i + step) % n
            W[i, j] = W[j, i] = 1.0
    for i in range(n):
        for step in range(1, k // 2 + 1):
            j = (i + step) % n
            if rng.random() < p:
                candidates = [
                    v for v in range(n) if v != i and W[i, v] == 0.0
                ]
                if not candidates:
                    continue
                new_j = int(rng.choice(candidates))
                W[i, j] = W[j, i] = 0.0
                W[i, new_j] = W[new_j, i] = 1.0
    return W, None
