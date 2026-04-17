import numpy as np

from spectralrl.envs.common import (
    RewardConfig,
    compute_reward,
    edge_index_from_support,
    project_onto_budget,
    weights_vector_to_matrix,
)
from spectralrl.graphs import erdos_renyi, laplacian


def test_scaling_weights_up_increases_lambda2():
    A, _ = erdos_renyi(12, 0.35, seed=0)
    edge_index = edge_index_from_support(A)
    m = edge_index.shape[0]
    cfg = RewardConfig(beta=0.0, gamma=0.0)  # pure lambda2
    rewards = []
    for scale in [0.1, 0.3, 0.6, 1.0]:
        w = np.full(m, scale, dtype=np.float64)
        W = weights_vector_to_matrix(w, edge_index, A.shape[0])
        r, info = compute_reward(laplacian(W), w, budget=None, cfg=cfg)
        rewards.append(r)
    # Strictly increasing: scaling up improves lambda2 linearly (scalar L).
    for a, b in zip(rewards, rewards[1:]):
        assert b > a - 1e-9


def test_budget_violation_penalty_active():
    A, _ = erdos_renyi(10, 0.4, seed=1)
    edge_index = edge_index_from_support(A)
    m = edge_index.shape[0]
    cfg = RewardConfig(beta=0.0, gamma=1.0)
    w = np.full(m, 1.0, dtype=np.float64)  # sum == m
    W = weights_vector_to_matrix(w, edge_index, A.shape[0])
    budget = m - 2.0
    r, info = compute_reward(laplacian(W), w, budget=budget, cfg=cfg)
    assert info["violation"] > 0
    # With gamma=1, reward is lambda2 - violation; violation should subtract off at least 1.0
    assert info["violation"] == 2.0


def test_project_onto_budget_scales_down():
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    out = project_onto_budget(w, budget=2.0, w_max=1.0)
    assert out.sum() <= 2.0 + 1e-12
    assert np.all(out >= 0)
    assert np.all(out <= 1.0 + 1e-12)
