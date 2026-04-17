"""Milestone 2: evaluate a trained PPO policy against non-learning baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spectralrl.graphs import erdos_renyi, ring, watts_strogatz
from spectralrl.rl.eval import evaluate_policy_vs_baselines
from spectralrl.rl.policy import build_actor
from spectralrl.utils import save_manifest, set_seed


def _load_actor(ckpt_path: Path, obs_dim: int, act_dim: int, hidden):
    actor = build_actor(obs_dim, act_dim, hidden=hidden)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt


def _held_out_supports(n: int, seeds: list[int]) -> list[tuple[str, np.ndarray]]:
    out = []
    for s in seeds:
        out.append(("ring", ring(n)[0]))
        out.append(("erdos_renyi", erdos_renyi(n, 0.25, seed=1000 + s)[0]))
        out.append(("watts_strogatz", watts_strogatz(n, 4, 0.1, seed=2000 + s)[0]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/m2"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    hidden = tuple(config["ppo"].get("hidden_sizes", [128, 128]))
    set_seed(int(config.get("seed", 0)))

    supports = _held_out_supports(args.n, args.seeds)
    # For eval we instantiate the actor with the ckpt's known dims.
    actor = build_actor(ckpt["obs_dim"], ckpt["act_dim"], hidden=hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    records = evaluate_policy_vs_baselines(
        actor,
        supports,
        budget=None,
        w_max=float(config["env"]["w_max"]),
        episode_len=int(config["env"]["episode_len"]),
        seed=int(config.get("seed", 0)),
    )

    csv_path = args.out / "eval.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["graph", "policy", "lambda2", "tau_eps", "rate", "cost"])
        for r in records:
            w.writerow([r.graph, r.policy, r.lambda2, r.tau_eps, r.rate, r.cost])

    _boxplot(records, args.out / "lambda2_by_policy.png", metric="lambda2")
    _boxplot(records, args.out / "tau_by_policy.png", metric="tau_eps")
    save_manifest(args.out, {"ckpt": str(args.ckpt), "n": args.n})
    print(f"wrote {csv_path}")


def _boxplot(records, path: Path, metric: str) -> None:
    by_policy: dict[str, list[float]] = {}
    for r in records:
        by_policy.setdefault(r.policy, []).append(getattr(r, metric))
    policies = sorted(by_policy.keys())
    data = [by_policy[p] for p in policies]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, labels=policies, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by policy (held-out graphs)")
    ax.grid(True, axis="y", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
