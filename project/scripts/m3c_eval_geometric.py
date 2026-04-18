"""Milestone 3c eval: compare PPO swarm policy vs greedy-centroid vs random.

For each eval seed we run three controllers on the same initial RGG:
  - ppo: trained TanhNormal actor
  - centroid: the existing move-toward-centroid heuristic (from m3_geometric.py)
  - random: uniform random velocities in [-1, 1]^(2n)
and log per-step lambda_2 and minimum pairwise distance.
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
from torchrl.envs.utils import ExplorationType, set_exploration_type

from spectralrl.envs.geometric_env import GeometricEnvConfig, GeometricSwarmEnv
from spectralrl.rl.policy import build_actor
from spectralrl.utils import save_manifest, set_seed


def _make_env(config: dict, seed: int) -> GeometricSwarmEnv:
    env_cfg = config["env"]
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


def _centroid_action(env: GeometricSwarmEnv) -> np.ndarray:
    pos = env.positions
    centroid = pos.mean(axis=0)
    direction = centroid - pos
    norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
    return (direction / norms).reshape(-1).astype(np.float32)


def _ppo_action(actor, obs: np.ndarray) -> np.ndarray:
    td = TensorDict(
        {"observation": torch.tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)},
        batch_size=[1],
    )
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        td = actor(td)
        a = td["action"].squeeze(0).cpu().numpy()
    return a


def _random_action(env: GeometricSwarmEnv, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2 * env.n).astype(np.float32)


def _rollout(env: GeometricSwarmEnv, policy_fn) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
    obs, _ = env.reset()
    init_pos = env.positions.copy()
    lam2_trace: list[float] = []
    min_dist_trace: list[float] = []
    done = truncated = False
    while not (done or truncated):
        a = policy_fn(obs)
        obs, _, done, truncated, info = env.step(a)
        lam2_trace.append(float(info["lambda2"]))
        min_dist_trace.append(float(info["min_dist"]))
    return lam2_trace, min_dist_trace, init_pos, env.positions.copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/m3c"))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(100, 110)))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    hidden = tuple(config["ppo"].get("hidden_sizes", [128, 128]))
    set_seed(int(config.get("seed", 0)))

    actor = build_actor(ckpt["obs_dim"], ckpt["act_dim"], hidden=hidden)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    all_traces: dict[str, list[list[float]]] = {"ppo": [], "centroid": [], "random": []}
    all_min: dict[str, list[list[float]]] = {"ppo": [], "centroid": [], "random": []}
    rows: list[dict] = []
    sample_pos_ppo = None
    sample_pos_init = None
    for seed in args.eval_seeds:
        # PPO
        env = _make_env(config, int(seed))
        obs, _ = env.reset(seed=int(seed))
        lam2_t, min_t = [], []
        done = trunc = False
        while not (done or trunc):
            a = _ppo_action(actor, obs)
            obs, _, done, trunc, info = env.step(a)
            lam2_t.append(float(info["lambda2"]))
            min_t.append(float(info["min_dist"]))
        if sample_pos_ppo is None:
            sample_pos_ppo = env.positions.copy()
        all_traces["ppo"].append(lam2_t)
        all_min["ppo"].append(min_t)
        rows.append({"seed": seed, "policy": "ppo", "lambda2_final": lam2_t[-1],
                     "mean_min_dist": float(np.mean(min_t))})

        # Centroid
        env = _make_env(config, int(seed))
        obs, _ = env.reset(seed=int(seed))
        if sample_pos_init is None:
            sample_pos_init = env.positions.copy()
        lam2_t, min_t = [], []
        done = trunc = False
        while not (done or trunc):
            a = _centroid_action(env)
            obs, _, done, trunc, info = env.step(a)
            lam2_t.append(float(info["lambda2"]))
            min_t.append(float(info["min_dist"]))
        all_traces["centroid"].append(lam2_t)
        all_min["centroid"].append(min_t)
        rows.append({"seed": seed, "policy": "centroid", "lambda2_final": lam2_t[-1],
                     "mean_min_dist": float(np.mean(min_t))})

        # Random
        env = _make_env(config, int(seed))
        obs, _ = env.reset(seed=int(seed))
        rng = np.random.default_rng(int(seed))
        lam2_t, min_t = [], []
        done = trunc = False
        while not (done or trunc):
            a = _random_action(env, rng)
            obs, _, done, trunc, info = env.step(a)
            lam2_t.append(float(info["lambda2"]))
            min_t.append(float(info["min_dist"]))
        all_traces["random"].append(lam2_t)
        all_min["random"].append(min_t)
        rows.append({"seed": seed, "policy": "random", "lambda2_final": lam2_t[-1],
                     "mean_min_dist": float(np.mean(min_t))})
        print(
            f"seed={seed} ppo={rows[-3]['lambda2_final']:.3f} "
            f"centroid={rows[-2]['lambda2_final']:.3f} random={rows[-1]['lambda2_final']:.3f}"
        )

    csv_path = args.out / "eval.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    _plot_traces(all_traces, args.out / "lambda2_curves.png", r"$\lambda_2$")
    _plot_traces(all_min, args.out / "min_dist_curves.png", "min pairwise distance")
    _plot_positions(sample_pos_init, sample_pos_ppo, config["env"].get("radius", 0.25),
                    args.out / "positions_ppo.png")
    save_manifest(args.out, {"ckpt": str(args.ckpt)})
    print(f"wrote {csv_path}")


def _plot_traces(all_traces, path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, traces in all_traces.items():
        L = min(len(t) for t in traces)
        M = np.stack([np.asarray(t[:L]) for t in traces], axis=0)
        mu = M.mean(axis=0)
        sd = M.std(axis=0)
        x = np.arange(L)
        ax.plot(x, mu, label=name)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_positions(init_pos, final_pos, radius, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, pos, title in zip(axes, [init_pos, final_pos], ["initial (PPO seed)", "final (PPO)"]):
        ax.scatter(pos[:, 0], pos[:, 1], s=20)
        n = pos.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) <= radius:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "k-", linewidth=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
