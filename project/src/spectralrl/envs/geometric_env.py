"""Geometric swarm env: 2D point-mass agents moving to increase connectivity (M3c).

Edges form between agents within sensing radius ``r``. The policy outputs a
bounded velocity per agent; the reward uses ``lambda_2`` of the induced random
geometric graph, minus a small motion cost. Observation is the concatenated
flattened position field plus degree statistics — a simple, shape-stable form
suitable for MLP policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..graphs.laplacian import fiedler_value, laplacian
from .common import RewardConfig


@dataclass
class GeometricEnvConfig:
    n: int = 30
    radius: float = 0.25
    v_max: float = 0.05
    world: float = 1.0
    episode_len: int = 200
    motion_cost: float = 0.01
    reward: RewardConfig = field(default_factory=RewardConfig)
    seed: int | None = None


class GeometricSwarmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: GeometricEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.n = int(cfg.n)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n * 2,), dtype=np.float32
        )
        # Observation: flattened positions (2n) + 4 degree stats + 1 lambda2.
        self.obs_dim = 2 * self.n + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self._rng = np.random.default_rng(cfg.seed)
        self._pos = np.zeros((self.n, 2), dtype=np.float64)
        self._step = 0
        self._last_info: dict[str, float] = {}

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = self._rng.random((self.n, 2)) * self.cfg.world
        self._step = 0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        v = np.asarray(action, dtype=np.float64).reshape(self.n, 2)
        v = np.clip(v, -1.0, 1.0) * self.cfg.v_max
        self._pos = np.clip(self._pos + v, 0.0, self.cfg.world)

        W = self._adjacency()
        lam2 = fiedler_value(laplacian(W))
        motion = float(np.linalg.norm(v, axis=1).sum())
        reward = lam2 - self.cfg.motion_cost * motion
        info = {
            "lambda2": float(lam2),
            "motion": motion,
            "edges": float(np.triu(W, k=1).sum()),
        }
        self._last_info = info
        self._step += 1
        terminated = False
        truncated = self._step >= self.cfg.episode_len
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- internals
    def _adjacency(self) -> np.ndarray:
        diff = self._pos[:, None, :] - self._pos[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        W = (d <= self.cfg.radius).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        return W

    def _observation(self) -> np.ndarray:
        W = self._adjacency()
        deg = W.sum(axis=1)
        deg_stats = np.array(
            [deg.mean(), deg.std(), deg.max(), deg.min()], dtype=np.float64
        ) / max(self.n - 1, 1)
        lam2 = fiedler_value(laplacian(W)) if deg.sum() > 0 else 0.0
        flat_pos = (self._pos / self.cfg.world).reshape(-1)
        return np.concatenate([flat_pos, deg_stats, np.array([lam2], dtype=np.float64)]).astype(
            np.float32
        )

    @property
    def positions(self) -> np.ndarray:
        return self._pos.copy()

    @property
    def last_info(self) -> dict[str, float]:
        return dict(self._last_info)
