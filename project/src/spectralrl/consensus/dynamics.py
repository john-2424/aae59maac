"""Discrete-time consensus dynamics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def stable_step_size(L: np.ndarray, safety: float = 0.9) -> float:
    """Return a step size ``alpha`` that keeps ``I - alpha L`` contractive.

    Uses ``2 / lambda_max(L)`` as the stability boundary and scales by ``safety``.
    """
    L_sym = 0.5 * (L + L.T)
    vals = eigh(L_sym, eigvals_only=True)
    lam_max = float(vals[-1])
    if lam_max <= 0:
        return 0.0
    return safety * 2.0 / lam_max


def run_consensus(
    L: np.ndarray, x0: np.ndarray, alpha: float, T: int
) -> np.ndarray:
    """Run ``T`` steps of ``x(k+1) = x(k) - alpha L x(k)``.

    ``x0`` may be shape ``(n,)`` or ``(n, B)`` for a batch of ``B`` initial
    conditions. Returns a trajectory of shape ``(T + 1, n)`` or ``(T + 1, n, B)``.
    """
    x = np.asarray(x0, dtype=np.float64)
    if x.ndim == 1:
        traj = np.empty((T + 1, x.shape[0]), dtype=np.float64)
    else:
        traj = np.empty((T + 1, x.shape[0], x.shape[1]), dtype=np.float64)
    traj[0] = x
    M = np.eye(L.shape[0]) - alpha * L
    for k in range(T):
        x = M @ x
        traj[k + 1] = x
    return traj
