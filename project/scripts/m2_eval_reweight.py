"""Milestone 2: evaluate a trained PPO policy against non-learning baselines.

The PPO actor's input/output dimensions are tied to the training graph's edge
count ``m`` (obs_dim = 4 + top_k + m + 2, act_dim = m), so evaluation must run
on the exact training support. We reconstruct it deterministically from the
ckpt's config and vary only the consensus initial-condition seed for stats.

Optional ``--sdp-json`` overlays the FDLA-SDP upper bound on lambda_2 and adds
a ``fraction_of_sdp`` column per eval row.
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _support(family: str, n: int, seed: int) -> np.ndarray:
    if family == "ring":
        W, _ = ring(n)
    elif family == "erdos_renyi":
        W, _ = erdos_renyi(n, 0.25, seed=seed)
    elif family == "watts_strogatz":
        W, _ = watts_strogatz(n, 4, 0.1, seed=seed)
    else:
        raise ValueError(f"unknown family: {family}")
    return W


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/m2"))
    parser.add_argument("--x0-seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--sdp-json", type=Path, default=None, help="JSON with lambda2_sdp to overlay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    hidden = tuple(config["ppo"].get("hidden_sizes", [128, 128]))
    set_seed(int(config.get("seed", 0)))

    env_cfg = config["env"]
    family = env_cfg["graph_families"][0]
    n = int(env_cfg["n"])
    seed = int(config.get("seed", 0))
    support = _support(family, n, seed)

    actor = build_actor(ckpt["obs_dim"], ckpt["act_dim"], hidden=hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    records = evaluate_policy_vs_baselines(
        actor,
        support=support,
        graph_name=family,
        w_max=float(env_cfg["w_max"]),
        budget=env_cfg.get("budget"),
        episode_len=int(env_cfg["episode_len"]),
        x0_seeds=list(args.x0_seeds),
        env_seed=seed,
    )

    lam2_sdp: float | None = None
    if args.sdp_json is not None and args.sdp_json.exists():
        with args.sdp_json.open() as f:
            lam2_sdp = float(json.load(f)["lambda2_sdp"])

    csv_path = args.out / "eval.csv"
    with csv_path.open("w", newline="") as f:
        header = ["graph", "policy", "lambda2", "tau_eps", "rate", "cost", "x0_seed"]
        if lam2_sdp is not None:
            header.append("fraction_of_sdp")
        w = csv.writer(f)
        w.writerow(header)
        for r in records:
            row = [r.graph, r.policy, r.lambda2, r.tau_eps, r.rate, r.cost, r.x0_seed]
            if lam2_sdp is not None:
                frac = r.lambda2 / lam2_sdp if lam2_sdp > 0 else float("nan")
                row.append(frac)
            w.writerow(row)

    _boxplot(records, args.out / "lambda2_by_policy.png", metric="lambda2", sdp_line=lam2_sdp)
    _boxplot(records, args.out / "tau_by_policy.png", metric="tau_eps", sdp_line=None)
    save_manifest(args.out, {"ckpt": str(args.ckpt), "family": family, "n": n, "lambda2_sdp": lam2_sdp})
    print(f"wrote {csv_path}")
    if lam2_sdp is not None:
        print(f"SDP upper bound: lambda2 = {lam2_sdp:.4f}")


def _boxplot(records, path: Path, metric: str, sdp_line: float | None = None) -> None:
    by_policy: dict[str, list[float]] = {}
    for r in records:
        by_policy.setdefault(r.policy, []).append(getattr(r, metric))
    policies = sorted(by_policy.keys())
    data = [by_policy[p] for p in policies]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(data, tick_labels=policies, showmeans=True)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by policy (training support, x0 seeds)")
    ax.grid(True, axis="y", linewidth=0.3)
    if sdp_line is not None:
        ax.axhline(sdp_line, linestyle="--", linewidth=1.2, label=f"SDP bound ({sdp_line:.3f})")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
