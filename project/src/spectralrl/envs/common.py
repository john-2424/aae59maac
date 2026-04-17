"""Shared helpers for RL environments that act on graph edge weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from ..graphs.laplacian import laplacian


def edge_index_from_support(A: np.ndarray) -> np.ndarray:
    """Return edges of the upper triangle of ``A`` as an ``(m, 2)`` int array."""
    i, j = np.nonzero(np.triu(A, k=1) > 0)
    return np.stack([i, j], axis=1).astype(np.int64)


def weights_vector_to_matrix(
    w: np.ndarray, edge_index: np.ndarray, n: int
) -> np.ndarray:
    W = np.zeros((n, n), dtype=np.float64)
    i = edge_index[:, 0]
    j = edge_index[:, 1]
    W[i, j] = w
    W[j, i] = w
    return W


def project_onto_budget(w: np.ndarray, budget: float, w_max: float) -> np.ndarray:
    w = np.clip(w, 0.0, w_max)
    total = float(w.sum())
    if budget is None or total <= budget or total == 0.0:
        return w
    return w * (budget / total)


def _top_k_eigs(L: np.ndarray, k: int) -> np.ndarray:
    L_sym = 0.5 * (L + L.T)
    n = L_sym.shape[0]
    k = min(k, n)
    vals = eigh(L_sym, eigvals_only=True, subset_by_index=[0, k - 1])
    out = np.zeros(k, dtype=np.float64)
    out[: vals.size] = vals
    return out


@dataclass
class RewardConfig:
    beta: float = 0.01
    gamma: float = 1.0


def compute_reward(
    L: np.ndarray,
    w: np.ndarray,
    budget: float | None,
    cfg: RewardConfig,
) -> tuple[float, dict[str, float]]:
    lam2 = float(_top_k_eigs(L, 2)[1])
    cost = float(w.sum())
    over_budget = 0.0 if budget is None else max(0.0, cost - budget)
    reward = lam2 - cfg.beta * cost - cfg.gamma * over_budget
    info = {
        "lambda2": lam2,
        "cost": cost,
        "violation": over_budget,
    }
    return reward, info


class FeatureExtractor:
    """Builds a fixed-size observation vector for a reweighting env."""

    def __init__(self, n: int, m: int, top_k: int = 4) -> None:
        self.n = n
        self.m = m
        self.top_k = top_k
        # 4 degree stats + top_k eigenvalues + m weights + 2 scalars (budget frac, step frac)
        self.dim = 4 + top_k + m + 2

    def __call__(
        self,
        W: np.ndarray,
        w: np.ndarray,
        budget_used_frac: float,
        step_frac: float,
    ) -> np.ndarray:
        d = W.sum(axis=1)
        dn = d / max(self.n - 1, 1)
        deg = np.array([dn.mean(), dn.std(), dn.max(), dn.min()], dtype=np.float64)
        L = laplacian(W)
        eigs = _top_k_eigs(L, self.top_k)
        # scale eigenvalues by n so they stay O(1) across graph sizes
        eigs = eigs / float(self.n)
        feats = np.concatenate(
            [
                deg,
                eigs,
                w.astype(np.float64),
                np.array([budget_used_frac, step_frac], dtype=np.float64),
            ]
        )
        return feats.astype(np.float32)
