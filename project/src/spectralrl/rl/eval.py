"""Evaluation utilities: policy vs baseline comparison on held-out graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from ..baselines.weights import (
    degree_proportional_weights,
    metropolis_weights,
    uniform_weights,
)
from ..consensus.dynamics import run_consensus, stable_step_size
from ..consensus.metrics import convergence_time, disagreement_energy, rate_estimate
from ..envs.common import edge_index_from_support, weights_vector_to_matrix
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


def _consensus_metrics(W: np.ndarray, T: int = 200, eps: float = 1e-3, rng: np.random.Generator | None = None) -> tuple[int, float]:
    rng = rng or np.random.default_rng(0)
    L = laplacian(W)
    alpha = stable_step_size(L)
    n = W.shape[0]
    x0 = rng.standard_normal(n)
    x0 = x0 - x0.mean()  # zero-mean so disagreement energy is meaningful
    traj = run_consensus(L, x0, alpha, T)
    tau = convergence_time(traj, eps=eps)
    rho = rate_estimate(traj)
    return tau, rho


def evaluate_policy_vs_baselines(
    actor,
    supports: list[tuple[str, np.ndarray]],
    budget: float | None = None,
    w_max: float = 1.0,
    episode_len: int = 32,
    seed: int = 0,
) -> list[EvalRecord]:
    out: list[EvalRecord] = []
    rng = np.random.default_rng(seed)
    for name, A in supports:
        cfg = ReweightEnvConfig(
            support=A, w_max=w_max, budget=budget, episode_len=episode_len, seed=seed
        )
        env = ReweightEnv(cfg)
        # PPO policy rollout, deterministic mean action.
        obs, _ = env.reset(seed=seed)
        done = truncated = False
        while not (done or truncated):
            from tensordict import TensorDict

            td = TensorDict(
                {"observation": torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)},
                batch_size=[1],
            )
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                td = actor(td)
                action = td["action"].squeeze(0).cpu().numpy()
            obs, _, done, truncated, _ = env.step(action)
        W_policy = env.current_weights_matrix
        lam2 = fiedler_value(laplacian(W_policy))
        tau, rho = _consensus_metrics(W_policy, rng=rng)
        out.append(EvalRecord(name, "ppo", lam2, tau, rho, float(env.last_info.get("cost", 0.0))))

        # Baselines on the same support.
        budget_eff = env.budget
        baselines = {
            "uniform": uniform_weights(A, budget=budget_eff, w_max=w_max),
            "metropolis": metropolis_weights(A),
            "degree_prop": degree_proportional_weights(A, budget=budget_eff, w_max=w_max),
        }
        for bname, Wb in baselines.items():
            lam2_b = fiedler_value(laplacian(Wb))
            tau_b, rho_b = _consensus_metrics(Wb, rng=rng)
            cost_b = float(np.triu(Wb, k=1).sum())
            out.append(EvalRecord(name, bname, lam2_b, tau_b, rho_b, cost_b))
    return out
