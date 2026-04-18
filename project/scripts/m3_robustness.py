"""Milestone 3b: robustness evaluation under perturbations.

Sweeps edge-failure and node-dropout probabilities; for each, compares
``lambda_2`` and convergence time of the baseline weighting vs the PPO
policy's weighting. Produces degradation curves.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from spectralrl.baselines.weights import metropolis_weights, uniform_weights
from spectralrl.consensus import convergence_time, run_consensus, stable_step_size
from spectralrl.envs.reweight_env import ReweightEnv, ReweightEnvConfig
from spectralrl.graphs import erdos_renyi, fiedler_value, laplacian
from spectralrl.rl.policy import build_actor
from spectralrl.robustness import node_dropout, random_edge_failure
from spectralrl.utils import save_manifest, set_seed


def _policy_weights(actor, A: np.ndarray, w_max: float, episode_len: int, seed: int) -> np.ndarray:
    cfg = ReweightEnvConfig(support=A, w_max=w_max, budget=None, episode_len=episode_len, seed=seed)
    env = ReweightEnv(cfg)
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
            a = td["action"].squeeze(0).cpu().numpy()
        obs, _, done, truncated, _ = env.step(a)
    return env.current_weights_matrix


def _evaluate(W: np.ndarray, failure_p: float, n_trials: int, rng: np.random.Generator) -> tuple[float, float]:
    lam2s = []
    taus = []
    for _ in range(n_trials):
        Wp = random_edge_failure(W, failure_p, rng=rng)
        if Wp.sum() == 0:
            lam2s.append(0.0)
            taus.append(500)
            continue
        L = laplacian(Wp)
        lam2s.append(fiedler_value(L))
        alpha = stable_step_size(L)
        n = W.shape[0]
        x0 = rng.standard_normal(n)
        x0 -= x0.mean()
        traj = run_consensus(L, x0, alpha, T=500)
        taus.append(convergence_time(traj, eps=1e-3))
    return float(np.mean(lam2s)), float(np.mean(taus))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/m3"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--failure-grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    set_seed(int(config.get("seed", 0)))
    hidden = tuple(config["ppo"].get("hidden_sizes", [128, 128]))
    actor = build_actor(ckpt["obs_dim"], ckpt["act_dim"], hidden=hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    rng = np.random.default_rng(int(config.get("seed", 0)))
    rows: list[dict] = []
    for seed in args.seeds:
        A, _ = erdos_renyi(args.n, 0.25, seed=seed)
        W_uniform = uniform_weights(A, budget=float(args.n), w_max=1.0)
        W_metro = metropolis_weights(A)
        W_ppo = _policy_weights(
            actor, A, w_max=float(config["env"]["w_max"]),
            episode_len=int(config["env"]["episode_len"]), seed=seed,
        )
        weights_by_policy = {"uniform": W_uniform, "metropolis": W_metro, "ppo": W_ppo}
        for p in args.failure_grid:
            for name, W in weights_by_policy.items():
                lam2, tau = _evaluate(W, p, args.trials, rng)
                rows.append({"seed": seed, "failure_p": p, "policy": name, "lambda2": lam2, "tau": tau})
                print(f"seed={seed} p={p:.2f} policy={name:<10s} lam2={lam2:.4f} tau={tau:.1f}")

    csv_path = args.out / "robustness.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot_curve(rows, args.out / "robustness_lambda2.png", metric="lambda2", ylabel=r"$\lambda_2$")
    _plot_curve(rows, args.out / "robustness_tau.png", metric="tau", ylabel=r"$\tau_\epsilon$")
    save_manifest(args.out, {"ckpt": str(args.ckpt), "n": args.n})
    print(f"wrote {csv_path}")


def _plot_curve(rows, path: Path, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    policies = sorted({r["policy"] for r in rows})
    ps = sorted({r["failure_p"] for r in rows})
    for policy in policies:
        ys = []
        for p in ps:
            vals = [r[metric] for r in rows if r["policy"] == policy and r["failure_p"] == p]
            ys.append(float(np.mean(vals)))
        ax.plot(ps, ys, "o-", label=policy)
    ax.set_xlabel("edge failure probability")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
