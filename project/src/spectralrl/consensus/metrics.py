"""Convergence metrics for consensus trajectories."""

from __future__ import annotations

import numpy as np


def disagreement_energy(x: np.ndarray) -> np.ndarray:
    """Compute ``||x - mean(x)||^2`` per time step.

    Accepts a single state ``(n,)``, a trajectory ``(T, n)``, or a batched
    trajectory ``(T, n, B)``. For batched input, the mean is taken over the
    node axis for each batch and each step.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        e = x - x.mean()
        return float(np.dot(e, e))
    if x.ndim == 2:
        mean = x.mean(axis=1, keepdims=True)
        e = x - mean
        return np.sum(e * e, axis=1)
    if x.ndim == 3:
        mean = x.mean(axis=1, keepdims=True)
        e = x - mean
        return np.sum(e * e, axis=1).mean(axis=-1)
    raise ValueError(f"unexpected shape {x.shape}")


def convergence_time(trajectory: np.ndarray, eps: float = 1e-3) -> int:
    """First step ``k`` with ``J(k) / J(0) <= eps``.

    Returns ``len(J) - 1`` (horizon) when the threshold is never reached.
    """
    J = disagreement_energy(trajectory)
    if np.isscalar(J):
        raise ValueError("convergence_time expects a trajectory")
    J = np.asarray(J)
    if J[0] == 0:
        return 0
    ratio = J / J[0]
    hits = np.where(ratio <= eps)[0]
    return int(hits[0]) if hits.size else int(len(J) - 1)


def rate_estimate(trajectory: np.ndarray) -> float:
    """Geometric contraction rate from ``log J(k)`` slope.

    Returns ``rho`` in ``J(k) ~ J(0) * rho^{2k}`` — i.e. ``|1 - alpha * lambda_2|``
    in the linear-consensus regime. Falls back to 1.0 on degenerate input.
    """
    J = disagreement_energy(trajectory)
    if np.isscalar(J):
        raise ValueError("rate_estimate expects a trajectory")
    J = np.asarray(J)
    mask = J > 1e-300
    if mask.sum() < 3:
        return 1.0
    k = np.arange(len(J))[mask]
    y = np.log(J[mask])
    slope, _ = np.polyfit(k, y, 1)
    return float(np.exp(0.5 * slope))
