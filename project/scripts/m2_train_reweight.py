"""Milestone 2: train PPO policy for edge reweighting on a chosen graph family."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from spectralrl.envs.common import RewardConfig
from spectralrl.envs.reweight_env import ReweightEnv, ReweightEnvConfig
from spectralrl.graphs import erdos_renyi, ring, watts_strogatz
from spectralrl.rl.train_ppo import train
from spectralrl.utils import set_seed


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


def make_env(config: dict):
    env_cfg = config["env"]
    family = env_cfg["graph_families"][0]
    n = int(env_cfg["n"])
    seed = int(config.get("seed", 0))
    A = _support(family, n, seed)
    m = int(np.triu(A > 0, k=1).sum())
    budget = env_cfg.get("budget")
    if budget is None:
        budget = 1.0 * m * float(env_cfg["w_max"])
    cfg = ReweightEnvConfig(
        support=A,
        w_max=float(env_cfg["w_max"]),
        budget=float(budget),
        episode_len=int(env_cfg["episode_len"]),
        reward=RewardConfig(
            beta=float(env_cfg["reward_beta"]), gamma=float(env_cfg["reward_gamma"])
        ),
        seed=seed,
        perturb_p=float(env_cfg.get("perturb_p", 0.0)),
    )
    return ReweightEnv(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/m2"))
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--total-frames", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.family is not None:
        config["env"]["graph_families"] = [args.family]
    if args.n is not None:
        config["env"]["n"] = int(args.n)
    if args.total_frames is not None:
        config["ppo"]["total_frames"] = int(args.total_frames)

    set_seed(int(config.get("seed", 0)))
    result = train(partial(make_env, config), config, args.out)
    print(f"best ckpt: {result.best_ckpt}")
    print(f"best mean reward: {result.final_reward:.4f}")


if __name__ == "__main__":
    main()
