"""Milestone 3a eval: PPO discrete rewiring vs greedy-swap vs random-rewire.

All three policies start from the same initial ER graph and take the same
``episode_len`` toggle actions under the same budget/connectivity rules. We
track lambda_2 after every step and compare final lambda_2 plus the
area-under-curve of the lambda_2 trace as a speed-of-improvement proxy.

Since the PPO categorical actor output is over ``n(n-1)/2`` pairs (fixed by
the training ``n``), eval graphs must have the same ``n`` as training. We
reconstruct the exact init_adj from the ckpt config and vary only
``x0_seeds``-like holdout seeds for the ER initial graph.
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
from tensordict import TensorDict
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, OneHotCategorical, ProbabilisticActor
from tensordict.nn import TensorDictModule

from spectralrl.envs.common import RewardConfig
from spectralrl.envs.rewire_env import RewireEnv, RewireEnvConfig
from spectralrl.graphs import erdos_renyi, fiedler_value, is_connected, laplacian
from spectralrl.utils import save_manifest, set_seed


def _build_rewire_actor(obs_dim: int, n_actions: int, hidden: tuple[int, ...]) -> ProbabilisticActor:
    actor_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=list(hidden), activation_class=nn.Tanh)
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module=actor_module,
        in_keys=["logits"],
        distribution_class=OneHotCategorical,
        return_log_prob=True,
    )


def _make_env(init_adj: np.ndarray, episode_len: int, beta: float, seed: int) -> RewireEnv:
    n = init_adj.shape[0]
    cfg = RewireEnvConfig(
        n=n,
        init_adj=init_adj,
        edge_budget=int(1.5 * np.triu(init_adj > 0, k=1).sum()),
        degree_cap=None,
        episode_len=episode_len,
        reward=RewardConfig(beta=beta, gamma=0.0),
        seed=seed,
    )
    return RewireEnv(cfg)


def _lam2(A: np.ndarray) -> float:
    if np.triu(A, k=1).sum() == 0:
        return 0.0
    return float(fiedler_value(laplacian(A)))


def ppo_rewire_rollout(actor, env: RewireEnv) -> tuple[np.ndarray, list[float]]:
    obs, _ = env.reset()
    trace = [_lam2(env.adjacency)]
    done = truncated = False
    while not (done or truncated):
        td = TensorDict(
            {"observation": torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)},
            batch_size=[1],
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            td = actor(td)
            action = td["action"].squeeze(0).cpu().numpy()
            a_idx = int(np.argmax(action))
        obs, _, done, truncated, info = env.step(a_idx)
        trace.append(float(info["lambda2"]))
    return env.adjacency, trace


def random_rewire_rollout(init_adj: np.ndarray, episode_len: int, rng: np.random.Generator) -> tuple[np.ndarray, list[float]]:
    env = _make_env(init_adj, episode_len, beta=0.0, seed=int(rng.integers(1_000_000)))
    env.reset()
    trace = [_lam2(env.adjacency)]
    for _ in range(episode_len):
        a = int(rng.integers(env.num_pairs))
        _, _, _, trunc, info = env.step(a)
        trace.append(float(info["lambda2"]))
        if trunc:
            break
    return env.adjacency, trace


def greedy_swap_rollout(init_adj: np.ndarray, episode_len: int) -> tuple[np.ndarray, list[float]]:
    """At each step, try every toggle, pick the one maximizing lambda_2 among feasible."""
    A = (init_adj > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    budget = int(1.5 * np.triu(A > 0, k=1).sum())
    trace = [_lam2(A)]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for _ in range(episode_len):
        best = -np.inf
        best_pair = None
        for (i, j) in pairs:
            B = A.copy()
            B[i, j] = B[j, i] = 0.0 if A[i, j] > 0 else 1.0
            active = int(np.triu(B > 0, k=1).sum())
            if active > budget or active == 0 or not is_connected(B):
                continue
            val = _lam2(B)
            if val > best:
                best = val
                best_pair = (i, j)
        if best_pair is None:
            trace.append(trace[-1])
            continue
        i, j = best_pair
        A[i, j] = A[j, i] = 0.0 if A[i, j] > 0 else 1.0
        trace.append(_lam2(A))
    return A, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/m3a"))
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    parser.add_argument("--skip-greedy", action="store_true", help="skip greedy baseline (slow on larger n)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    env_cfg = config["env"]
    n = int(env_cfg["n"])
    episode_len = int(env_cfg.get("episode_len", 64))
    beta = float(env_cfg.get("reward_beta", 0.0))
    hidden = tuple(config["ppo"].get("hidden_sizes", [128, 128]))
    set_seed(int(config.get("seed", 0)))

    # Dims from a single env instance (actor was trained with these sizes).
    sample_env = _make_env(erdos_renyi(n, 0.2, seed=int(config.get("seed", 0)))[0], episode_len, beta, 0)
    obs_dim = int(sample_env.observation_space.shape[0])
    n_actions = int(sample_env.action_space.n)
    actor = _build_rewire_actor(obs_dim, n_actions, hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    rows: list[dict] = []
    all_traces: dict[str, list[list[float]]] = {"ppo": [], "random": [], "greedy": []}
    for seed in args.holdout_seeds:
        A0, _ = erdos_renyi(n, 0.2, seed=int(seed))
        env = _make_env(A0, episode_len, beta, int(seed))
        _, trace_ppo = ppo_rewire_rollout(actor, env)
        all_traces["ppo"].append(trace_ppo)
        rng = np.random.default_rng(int(seed))
        _, trace_rand = random_rewire_rollout(A0, episode_len, rng)
        all_traces["random"].append(trace_rand)
        if not args.skip_greedy:
            _, trace_greedy = greedy_swap_rollout(A0, episode_len)
            all_traces["greedy"].append(trace_greedy)

        def _auc(t: list[float]) -> float:
            return float(np.trapz(np.asarray(t)))

        rows.append({"seed": seed, "policy": "ppo", "lambda2_final": trace_ppo[-1], "lambda2_auc": _auc(trace_ppo)})
        rows.append({"seed": seed, "policy": "random", "lambda2_final": trace_rand[-1], "lambda2_auc": _auc(trace_rand)})
        if not args.skip_greedy:
            rows.append({"seed": seed, "policy": "greedy", "lambda2_final": trace_greedy[-1], "lambda2_auc": _auc(trace_greedy)})
        print(
            f"seed={seed} ppo={trace_ppo[-1]:.3f} random={trace_rand[-1]:.3f}"
            + (f" greedy={trace_greedy[-1]:.3f}" if not args.skip_greedy else "")
        )

    csv_path = args.out / "eval.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot_curves(all_traces, args.out / "lambda2_curves.png")
    _plot_bars(rows, args.out / "lambda2_final_bar.png")
    save_manifest(args.out, {"ckpt": str(args.ckpt), "n": n, "episode_len": episode_len})
    print(f"wrote {csv_path}")


def _plot_curves(all_traces, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, traces in all_traces.items():
        if not traces:
            continue
        L = min(len(t) for t in traces)
        M = np.stack([np.asarray(t[:L]) for t in traces], axis=0)
        mu = M.mean(axis=0)
        sd = M.std(axis=0)
        x = np.arange(L)
        ax.plot(x, mu, label=name)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title("Rewiring: lambda_2 vs step (mean ± std over holdout seeds)")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_bars(rows, path: Path) -> None:
    by_policy: dict[str, list[float]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r["lambda2_final"])
    policies = sorted(by_policy.keys())
    means = [float(np.mean(by_policy[p])) for p in policies]
    stds = [float(np.std(by_policy[p])) for p in policies]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(policies, means, yerr=stds, capsize=4)
    ax.set_ylabel(r"final $\lambda_2$")
    ax.set_title("Final lambda_2 by rewiring policy")
    ax.grid(True, axis="y", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
