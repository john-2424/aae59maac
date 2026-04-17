import numpy as np
import pytest

from spectralrl.envs import ReweightEnv, ReweightEnvConfig
from spectralrl.graphs import erdos_renyi


def _make_env(seed=0, budget=3.0, episode_len=8):
    A, _ = erdos_renyi(10, 0.4, seed=seed)
    cfg = ReweightEnvConfig(
        support=A, w_max=1.0, budget=budget, episode_len=episode_len, seed=seed
    )
    return ReweightEnv(cfg)


def test_observation_and_action_shapes():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    obs2, r, done, trunc, info = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert np.isfinite(r)
    assert not done
    assert "lambda2" in info


def test_action_projection_respects_constraints():
    env = _make_env(budget=2.0)
    env.reset(seed=1)
    # Extreme positive action => all weights pushed to w_max, then scaled to budget.
    action = np.ones(env.m, dtype=np.float32)
    env.step(action)
    W = env.current_weights_matrix
    assert W.max() <= 1.0 + 1e-9
    assert W.min() >= -1e-12
    total_weight = np.triu(W, k=1).sum()
    assert total_weight <= 2.0 + 1e-6


def test_reset_deterministic_under_seed():
    env1 = _make_env(seed=7)
    env2 = _make_env(seed=7)
    o1, _ = env1.reset(seed=7)
    o2, _ = env2.reset(seed=7)
    assert np.allclose(o1, o2)


def test_episode_truncates_at_length():
    env = _make_env(episode_len=5)
    env.reset(seed=0)
    for i in range(5):
        _, _, done, trunc, _ = env.step(env.action_space.sample())
        if i < 4:
            assert not trunc
    assert trunc
