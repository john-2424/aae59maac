"""Milestone 3c: train a PPO policy on the geometric swarm env.

Uses the same TorchRL continuous-actor pipeline (``rl/train_ppo.py``) as M2.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import yaml

from spectralrl.envs.geometric_env import GeometricEnvConfig, GeometricSwarmEnv
from spectralrl.rl.train_ppo import train
from spectralrl.utils import set_seed


def make_env(config: dict):
    env_cfg = config["env"]
    seed = int(config.get("seed", 0))
    cfg = GeometricEnvConfig(
        n=int(env_cfg["n"]),
        radius=float(env_cfg.get("radius", 0.25)),
        v_max=float(env_cfg.get("v_max", 0.02)),
        episode_len=int(env_cfg.get("episode_len", 100)),
        motion_cost=float(env_cfg.get("motion_cost", 0.01)),
        r_min=float(env_cfg.get("r_min", 0.08)),
        collision_penalty=float(env_cfg.get("collision_penalty", 1.0)),
        spread_bonus=float(env_cfg.get("spread_bonus", 0.1)),
        weight_kernel=str(env_cfg.get("weight_kernel", "binary")),
        init_connected=bool(env_cfg.get("init_connected", False)),
        seed=seed,
    )
    return GeometricSwarmEnv(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_m3c.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/m3c"))
    parser.add_argument("--total-frames", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.total_frames is not None:
        config["ppo"]["total_frames"] = int(args.total_frames)

    set_seed(int(config.get("seed", 0)))
    result = train(partial(make_env, config), config, args.out)
    print(f"best ckpt: {result.best_ckpt}")
    print(f"best mean reward: {result.final_reward:.4f}")


if __name__ == "__main__":
    main()
