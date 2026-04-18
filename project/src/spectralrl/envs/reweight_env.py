"""Gymnasium env for constrained continuous edge reweighting (M2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..graphs.laplacian import laplacian
from ..robustness.perturbations import random_edge_failure
from .common import (
    FeatureExtractor,
    RewardConfig,
    compute_reward,
    edge_index_from_support,
    project_onto_budget,
    weights_vector_to_matrix,
)


@dataclass
class ReweightEnvConfig:
    support: np.ndarray  # (n, n) 0/1 adjacency defining the allowed edge set
    w_max: float = 1.0
    budget: float | None = None  # None => |E| * w_max (effectively no scaling)
    episode_len: int = 32
    reward: RewardConfig = field(default_factory=RewardConfig)
    top_k_eigs: int = 4
    seed: int | None = None
    perturb_p: float = 0.0  # per-step edge-failure prob applied *before* reward (for robust training)


class ReweightEnv(gym.Env):
    """Continuous edge reweighting on a fixed edge support."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: ReweightEnvConfig):
        super().__init__()
        A = (cfg.support > 0).astype(np.float64)
        np.fill_diagonal(A, 0.0)
        self.A = A
        self.n = A.shape[0]
        self.edge_index = edge_index_from_support(A)
        self.m = int(self.edge_index.shape[0])
        if self.m == 0:
            raise ValueError("support must contain at least one edge")

        self.w_max = float(cfg.w_max)
        self.budget = float(cfg.budget) if cfg.budget is not None else float(self.m) * self.w_max
        self.episode_len = int(cfg.episode_len)
        self.reward_cfg = cfg.reward
        self.perturb_p = float(cfg.perturb_p)
        self.features = FeatureExtractor(self.n, self.m, top_k=cfg.top_k_eigs)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.m,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.dim,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(cfg.seed)
        self._step = 0
        self._w = np.zeros(self.m, dtype=np.float64)
        self._last_info: dict[str, float] = {}

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        # warm-start weights at the Metropolis-like value on the support
        d = self.A.sum(axis=1)
        w0 = np.zeros(self.m, dtype=np.float64)
        for e_idx, (i, j) in enumerate(self.edge_index):
            w0[e_idx] = 1.0 / (1.0 + max(d[i], d[j]))
        self._w = project_onto_budget(w0, self.budget, self.w_max)
        return self._observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(self.m)
        a = np.clip(a, -1.0, 1.0)
        w = 0.5 * self.w_max * (a + 1.0)
        w = project_onto_budget(w, self.budget, self.w_max)
        self._w = w

        W = weights_vector_to_matrix(w, self.edge_index, self.n)
        if self.perturb_p > 0.0:
            W_eff = random_edge_failure(W, self.perturb_p, rng=self._rng)
        else:
            W_eff = W
        L = laplacian(W_eff)
        reward, info = compute_reward(L, w, self.budget, self.reward_cfg)
        self._last_info = info

        self._step += 1
        terminated = False
        truncated = self._step >= self.episode_len
        obs = self._observation()
        return obs, float(reward), terminated, truncated, info

    # --------------------------------------------------------------- internals
    def _observation(self) -> np.ndarray:
        W = weights_vector_to_matrix(self._w, self.edge_index, self.n)
        budget_used_frac = float(self._w.sum()) / self.budget if self.budget > 0 else 0.0
        step_frac = self._step / max(self.episode_len, 1)
        return self.features(W, self._w, budget_used_frac, step_frac)

    # -------------------------------------------------------- eval conveniences
    @property
    def current_weights_matrix(self) -> np.ndarray:
        return weights_vector_to_matrix(self._w, self.edge_index, self.n)

    @property
    def last_info(self) -> dict[str, float]:
        return dict(self._last_info)
