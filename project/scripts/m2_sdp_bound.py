"""Compute the SDP upper bound on lambda_2 for an M2 training support.

Reconstructs the training graph from a ckpt config (same helper as
``m2_eval_reweight.py``) and solves the Boyd-Ghosh FDLA SDP. Writes
``sdp.json`` with ``{lambda2_sdp, family, n, budget, method}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectralrl.baselines.sdp import fdla_upper_bound, fdla_upper_bound_subgradient
from spectralrl.graphs import erdos_renyi, ring, watts_strogatz


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
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--method", choices=["cvxpy", "subgradient"], default="cvxpy")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    env_cfg = config["env"]
    family = env_cfg["graph_families"][0]
    n = int(env_cfg["n"])
    seed = int(config.get("seed", 0))
    w_max = float(env_cfg["w_max"])
    A = _support(family, n, seed)
    m = int(np.triu(A > 0, k=1).sum())
    budget = env_cfg.get("budget")
    budget = float(m) * w_max if budget is None else float(budget)

    if args.method == "cvxpy":
        try:
            _, lam2 = fdla_upper_bound(A, w_max=w_max, budget=budget)
            method_used = "cvxpy+SCS"
        except ImportError:
            print("cvxpy unavailable; falling back to subgradient method")
            _, lam2 = fdla_upper_bound_subgradient(A, w_max=w_max, budget=budget)
            method_used = "subgradient"
    else:
        _, lam2 = fdla_upper_bound_subgradient(A, w_max=w_max, budget=budget)
        method_used = "subgradient"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "lambda2_sdp": lam2,
        "family": family,
        "n": n,
        "budget": budget,
        "w_max": w_max,
        "seed": seed,
        "method": method_used,
    }
    with args.out.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"SDP lambda2 = {lam2:.4f} ({method_used}); wrote {args.out}")


if __name__ == "__main__":
    main()
