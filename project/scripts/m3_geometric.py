"""Milestone 3c: geometric swarm demo.

Trains (briefly) or runs a random-motion baseline in the geometric env and
plots the lambda_2 trajectory plus a final-positions snapshot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectralrl.envs.geometric_env import GeometricEnvConfig, GeometricSwarmEnv
from spectralrl.graphs import laplacian, fiedler_value
from spectralrl.utils import save_manifest, set_seed


def _greedy_connectivity_action(env: GeometricSwarmEnv) -> np.ndarray:
    """Pick each agent's velocity as a step toward the swarm centroid.

    A cheap non-RL baseline that demonstrates the move-to-improve-connectivity
    idea without requiring training.
    """
    pos = env.positions
    centroid = pos.mean(axis=0)
    direction = centroid - pos
    norms = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
    direction = direction / norms
    return direction.reshape(-1).astype(np.float32)


def run(out_dir: Path, n: int, steps: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    cfg = GeometricEnvConfig(n=n, radius=0.25, v_max=0.03, episode_len=steps, seed=seed)
    env = GeometricSwarmEnv(cfg)
    obs, _ = env.reset(seed=seed)
    lam2_trace = []
    init_pos = env.positions.copy()
    for _ in range(steps):
        a = _greedy_connectivity_action(env)
        obs, r, done, trunc, info = env.step(a)
        lam2_trace.append(info["lambda2"])
        if done or trunc:
            break
    final_pos = env.positions.copy()

    # Plot: lambda_2 trace.
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(lam2_trace)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title("Spectral gap over episode (greedy move-to-centroid)")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lambda2_trace.png", dpi=150)
    fig.savefig(out_dir / "lambda2_trace.pdf")
    plt.close(fig)

    # Plot: initial and final positions.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, pos, title in zip(
        axes, [init_pos, final_pos], ["initial positions", "final positions"]
    ):
        ax.scatter(pos[:, 0], pos[:, 1], s=20)
        # draw edges within radius
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) <= cfg.radius:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], "k-", linewidth=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "positions.png", dpi=150)
    fig.savefig(out_dir / "positions.pdf")
    plt.close(fig)

    save_manifest(out_dir, {"n": n, "steps": steps, "seed": seed, "radius": cfg.radius})
    print(f"final lambda2: {lam2_trace[-1]:.4f} (initial ~ {lam2_trace[0]:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/geom"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.out, args.n, args.steps, args.seed)


if __name__ == "__main__":
    main()
