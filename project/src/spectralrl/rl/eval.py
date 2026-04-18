"""Evaluation utilities: policy vs baseline comparison.

The PPO actor is tied to a fixed observation/action dimension, which in the
current env equals ``4 + top_k + m + 2`` (obs) and ``m`` (action) for the
training graph's edge count ``m``. Evaluating on a graph with a different ``m``
would fail with a shape mismatch, so we always rebuild the exact training
support from the ckpt config and compare policy vs baselines on that support.
Statistical variation comes from resampling the consensus initial condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ..baselines.weights import (
    degree_proportional_weights,
    metropolis_weights,
    uniform_weights,
)
from ..consensus.dynamics import run_consensus, stable_step_size
from ..consensus.metrics import convergence_time, rate_estimate
from ..envs.reweight_env import ReweightEnv, ReweightEnvConfig
from ..graphs.laplacian import fiedler_value, laplacian


@dataclass
class EvalRecord:
    graph: str
    policy: str
    lambda2: float
    tau_eps: int
    rate: float
    cost: float
    x0_seed: int


def _consensus_metrics(W: np.ndarray, x0_seed: int, T: int = 500, eps: float = 1e-3) -> tuple[int, float]:
    rng = np.random.default_rng(x0_seed)
    L = laplacian(W)
    alpha = stable_step_size(L)
    n = W.shape[0]
    x0 = rng.standard_normal(n)
    x0 -= x0.mean()
    traj = run_consensus(L, x0, alpha, T)
    return convergence_time(traj, eps=eps), rate_estimate(traj)


def _policy_weights(actor, env: ReweightEnv, episode_seed: int) -> np.ndarray:
    obs, _ = env.reset(seed=episode_seed)
    done = truncated = False
    while not (done or truncated):
        td = TensorDict(
            {"observation": torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)},
            batch_size=[1],
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = actor(td)
            action = td["action"].squeeze(0).cpu().numpy()
        obs, _, done, truncated, _ = env.step(action)
    return env.current_weights_matrix


def evaluate_policy_vs_baselines(
    actor,
    support: np.ndarray,
    graph_name: str,
    w_max: float,
    budget: float | None,
    episode_len: int,
    x0_seeds: list[int],
    env_seed: int = 0,
) -> list[EvalRecord]:
    """Evaluate on a single fixed support. The policy runs one deterministic
    rollout to produce its weight matrix; baselines are computed once from the
    support. Consensus metrics (``tau_eps``, ``rho``) are averaged over multiple
    ``x0`` seeds; ``lambda_2`` is the same across seeds by construction.
    """
    out: list[EvalRecord] = []

    cfg = ReweightEnvConfig(
        support=support, w_max=w_max, budget=budget, episode_len=episode_len, seed=env_seed
    )
    env = ReweightEnv(cfg)
    W_policy = _policy_weights(actor, env, episode_seed=env_seed)
    budget_eff = env.budget
    lam2_policy = fiedler_value(laplacian(W_policy))
    cost_policy = float(np.triu(W_policy, k=1).sum())

    weights_by_policy: dict[str, np.ndarray] = {
        "ppo": W_policy,
        "uniform": uniform_weights(support, budget=budget_eff, w_max=w_max),
        "metropolis": metropolis_weights(support),
        "degree_prop": degree_proportional_weights(support, budget=budget_eff, w_max=w_max),
    }
    lambda2_by_policy = {k: fiedler_value(laplacian(v)) for k, v in weights_by_policy.items()}
    cost_by_policy = {k: float(np.triu(v, k=1).sum()) for k, v in weights_by_policy.items()}
    lambda2_by_policy["ppo"] = lam2_policy
    cost_by_policy["ppo"] = cost_policy

    for seed in x0_seeds:
        for name, W in weights_by_policy.items():
            tau, rho = _consensus_metrics(W, x0_seed=seed)
            out.append(
                EvalRecord(
                    graph=graph_name,
                    policy=name,
                    lambda2=lambda2_by_policy[name],
                    tau_eps=tau,
                    rate=rho,
                    cost=cost_by_policy[name],
                    x0_seed=seed,
                )
            )
    return out
