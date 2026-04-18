"""Gymnasium env for discrete edge rewiring (M3a).

The agent toggles an edge ``(i, j)`` per step. An edge budget caps the number
of active edges; a degree cap optionally limits per-node connectivity. Actions
that would violate connectivity or the budget are rejected with a penalty and
the graph state is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..graphs.laplacian import fiedler_value, is_connected, laplacian
from .common import FeatureExtractor, RewardConfig


@dataclass
class RewireEnvConfig:
    n: int
    init_adj: np.ndarray  # (n, n) starting 0/1 adjacency
    edge_budget: int
    degree_cap: int | None = None
    w_max: float = 1.0
    episode_len: int = 64
    invalid_penalty: float = 0.5
    reward: RewardConfig = field(default_factory=RewardConfig)
    top_k_eigs: int = 4
    seed: int | None = None
    # When True, reset() draws a fresh ER(n, p_resample) graph each episode
    # instead of copying init_adj. Required for the actor to generalize across
    # graph topologies — without it, PPO overfits to one specific graph.
    resample_init: bool = False
    p_resample: float = 0.2


class RewireEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: RewireEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.n = int(cfg.n)
        self.num_pairs = self.n * (self.n - 1) // 2
        self.pair_index = np.array(
            [(i, j) for i in range(self.n) for j in range(i + 1, self.n)],
            dtype=np.int64,
        )
        self.action_space = spaces.Discrete(self.num_pairs)
        # Binary weights on all possible pairs, so features are fixed size.
        self.features = FeatureExtractor(self.n, self.num_pairs, top_k=cfg.top_k_eigs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.features.dim,), dtype=np.float32
        )
        self._rng = np.random.default_rng(cfg.seed)
        self._A = (cfg.init_adj > 0).astype(np.float64)
        np.fill_diagonal(self._A, 0.0)
        self._step = 0
        self._last_info: dict[str, float] = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.cfg.resample_init:
            # Reject disconnected draws (rare for p≥0.2, n=20) so eval metric is meaningful.
            for _ in range(64):
                A = self._sample_er(self.cfg.p_resample)
                if is_connected(A):
                    self._A = A
                    break
            else:
                self._A = A  # fall back to last draw even if not connected
        else:
            self._A = (self.cfg.init_adj > 0).astype(np.float64)
            np.fill_diagonal(self._A, 0.0)
        self._step = 0
        return self._observation(), {}

    def _sample_er(self, p: float) -> np.ndarray:
        n = self.n
        upper = (self._rng.random((n, n)) < p).astype(np.float64)
        upper = np.triu(upper, k=1)
        A = upper + upper.T
        return A

    def step(self, action: int):
        i, j = map(int, self.pair_index[int(action)])
        invalid = False
        proposed = self._A.copy()
        if proposed[i, j] > 0:
            proposed[i, j] = proposed[j, i] = 0.0
        else:
            proposed[i, j] = proposed[j, i] = 1.0

        active_edges = float(np.triu(proposed, k=1).sum())
        if active_edges > self.cfg.edge_budget:
            invalid = True
        if self.cfg.degree_cap is not None and proposed.sum(axis=1).max() > self.cfg.degree_cap:
            invalid = True
        if active_edges > 0 and not is_connected(proposed):
            invalid = True

        if not invalid:
            self._A = proposed

        L = laplacian(self._A)
        lam2 = fiedler_value(L) if active_edges > 0 else 0.0
        cost = float(np.triu(self._A, k=1).sum())
        reward = lam2 - self.cfg.reward.beta * cost
        if invalid:
            reward -= self.cfg.invalid_penalty
        info = {
            "lambda2": float(lam2),
            "cost": cost,
            "invalid": float(invalid),
        }
        self._last_info = info
        self._step += 1
        terminated = False
        truncated = self._step >= self.cfg.episode_len
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- internals
    def _observation(self) -> np.ndarray:
        # Weights vector is just the binary triu entries of A on all pairs.
        w = self._A[self.pair_index[:, 0], self.pair_index[:, 1]].astype(np.float64)
        budget_used = float(np.triu(self._A, k=1).sum()) / max(self.cfg.edge_budget, 1)
        step_frac = self._step / max(self.cfg.episode_len, 1)
        return self.features(self._A, w, budget_used, step_frac)

    @property
    def adjacency(self) -> np.ndarray:
        return self._A.copy()

    @property
    def last_info(self) -> dict[str, float]:
        return dict(self._last_info)
