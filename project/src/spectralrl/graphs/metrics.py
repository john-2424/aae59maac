"""Simple graph summary statistics used by envs and loggers."""

from __future__ import annotations

import numpy as np


def edge_count(W: np.ndarray, weighted: bool = False) -> float:
    U = np.triu(W, k=1)
    if weighted:
        return float(U.sum())
    return float((U > 0).sum())


def total_weight(W: np.ndarray) -> float:
    return float(np.triu(W, k=1).sum())


def degree_stats(W: np.ndarray) -> dict[str, float]:
    d = W.sum(axis=1)
    return {
        "mean": float(d.mean()),
        "std": float(d.std()),
        "max": float(d.max()),
        "min": float(d.min()),
    }
