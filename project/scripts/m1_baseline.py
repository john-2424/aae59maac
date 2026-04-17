"""Milestone 1: baseline graph + consensus pipeline.

For each graph family x size, computes ``lambda_2``, edge count, convergence
time, and geometric rate; saves a CSV and two plots.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectralrl.consensus import (
    convergence_time,
    disagreement_energy,
    rate_estimate,
    run_consensus,
    stable_step_size,
)
from spectralrl.graphs import (
    erdos_renyi,
    fiedler_value,
    grid,
    laplacian,
    random_geometric,
    ring,
    watts_strogatz,
)
from spectralrl.utils import save_manifest, set_seed


def _rect_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    rows = int(np.round(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    return grid(rows, cols)


def _families(n: int, seed: int):
    return [
        ("ring", lambda: ring(n)),
        ("grid", lambda: _rect_grid(n)),
        ("erdos_renyi", lambda: erdos_renyi(n, 0.2, seed=seed)),
        ("random_geometric", lambda: random_geometric(n, 0.35, seed=seed)),
        ("watts_strogatz", lambda: watts_strogatz(n, 4, 0.1, seed=seed)),
    ]


def run(out_dir: Path, sizes: list[int], seed: int, quick: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(out_dir, {"sizes": sizes, "seed": seed, "quick": quick})
    rows: list[dict] = []
    curves: dict[tuple[str, int], np.ndarray] = {}

    rng = np.random.default_rng(seed)
    T = 100 if quick else 500

    for n in sizes:
        for name, make in _families(n, seed):
            W, _ = make()
            n_actual = W.shape[0]
            L = laplacian(W)
            lam2 = fiedler_value(L)
            e_count = int(np.triu(W > 0, k=1).sum())
            alpha = stable_step_size(L)
            x0 = rng.standard_normal(n_actual)
            x0 -= x0.mean()
            traj = run_consensus(L, x0, alpha, T=T)
            tau = convergence_time(traj, eps=1e-3)
            rho = rate_estimate(traj)
            J = disagreement_energy(traj)
            J_norm = J / max(J[0], 1e-30)
            curves[(name, n)] = J_norm
            rows.append(
                {
                    "family": name,
                    "n": n_actual,
                    "edges": e_count,
                    "lambda2": lam2,
                    "alpha": alpha,
                    "tau_eps": tau,
                    "rate_rho": rho,
                }
            )
            print(
                f"[{name:>18s} n={n:>3d}] lam2={lam2:.4f} edges={e_count:>4d} "
                f"tau(1e-3)={tau:>4d} rho={rho:.4f}"
            )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Plot 1: lambda_2 vs edges per family.
    _plot_lambda2_vs_edges(rows, out_dir / "lambda2_vs_edges.png")
    # Plot 2: J(k) curves for the largest n.
    _plot_J_curves(curves, max(sizes), out_dir / "J_vs_k.png")
    print(f"\nWrote {csv_path}")


def _plot_lambda2_vs_edges(rows, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    by_family: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append((r["edges"], r["lambda2"]))
    for family, pts in sorted(by_family.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", label=family)
    ax.set_xlabel("edges")
    ax.set_ylabel(r"$\lambda_2(L)$")
    ax.set_title("Algebraic connectivity vs edge count")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_J_curves(curves, n_target: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for (family, n), J in sorted(curves.items()):
        if n != n_target:
            continue
        ax.semilogy(J, label=family)
    ax.set_xlabel("step k")
    ax.set_ylabel(r"$J(k)/J(0)$")
    ax.set_title(f"Disagreement energy decay (n={n_target})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/m1"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    run(args.out, args.sizes, args.seed, args.quick)


if __name__ == "__main__":
    main()
