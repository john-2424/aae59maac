"""Aggregate M2 results across seeds.

Expects a runs directory shaped like ``runs/<name>/seed_{s}/{best.pt,train_log.csv}``
produced by ``run_multiseed.py``. Runs ``m2_eval_reweight`` per seed and then
aggregates both the training curves and the eval rows.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_train_log(path: Path) -> tuple[list[int], list[float]]:
    iters, rewards = [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            rewards.append(float(row["mean_reward"]))
    return iters, rewards


def _agg_training_curves(seed_dirs: list[Path], out_png: Path) -> None:
    curves = []
    for sd in seed_dirs:
        log = sd / "train_log.csv"
        if not log.exists():
            continue
        iters, rewards = _read_train_log(log)
        curves.append(np.array(rewards, dtype=np.float64))
    if not curves:
        return
    L = min(len(c) for c in curves)
    M = np.stack([c[:L] for c in curves], axis=0)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, mu, label=f"mean over {len(curves)} seeds")
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.25, label=r"$\pm 1\sigma$")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean reward")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def _run_eval(seed_dir: Path, out_dir: Path, sdp_json: Path | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/m2_eval_reweight.py",
        "--ckpt", str(seed_dir / "best.pt"),
        "--out", str(out_dir),
    ]
    if sdp_json is not None and sdp_json.exists():
        cmd += ["--sdp-json", str(sdp_json)]
    subprocess.run(cmd, check=True)
    return out_dir / "eval.csv"


def _aggregate_eval(eval_csvs: list[Path], out_csv: Path) -> None:
    rows = []
    for p in eval_csvs:
        with p.open() as f:
            for r in csv.DictReader(f):
                r["_src"] = str(p)
                rows.append(r)
    # group by (graph, policy)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["graph"], r["policy"])].append(r)

    out_fields = ["graph", "policy", "n_seeds"]
    metrics = ["lambda2", "tau_eps", "rate", "cost"]
    for m in metrics:
        out_fields += [f"{m}_mean", f"{m}_std"]
    if any("fraction_of_sdp" in r for r in rows):
        out_fields += ["fraction_of_sdp_mean", "fraction_of_sdp_std"]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for (graph, policy), rs in sorted(groups.items()):
            row = {"graph": graph, "policy": policy, "n_seeds": len(rs)}
            for m in metrics:
                vals = [float(x[m]) for x in rs if x.get(m) not in (None, "")]
                row[f"{m}_mean"] = float(mean(vals)) if vals else float("nan")
                row[f"{m}_std"] = float(stdev(vals)) if len(vals) > 1 else 0.0
            if "fraction_of_sdp_mean" in out_fields:
                vals = [float(x["fraction_of_sdp"]) for x in rs if x.get("fraction_of_sdp") not in (None, "")]
                row["fraction_of_sdp_mean"] = float(mean(vals)) if vals else float("nan")
                row["fraction_of_sdp_std"] = float(stdev(vals)) if len(vals) > 1 else 0.0
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True, help="contains seed_* subdirs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sdp-json", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted([d for d in args.runs_dir.glob("seed_*") if d.is_dir()])
    if not seed_dirs:
        raise SystemExit(f"no seed_* dirs under {args.runs_dir}")
    print(f"found {len(seed_dirs)} seed dirs")

    _agg_training_curves(seed_dirs, args.out / "training_curve.png")

    eval_csvs = []
    for sd in seed_dirs:
        per_seed_out = args.out / sd.name
        eval_csvs.append(_run_eval(sd, per_seed_out, args.sdp_json))

    _aggregate_eval(eval_csvs, args.out / "eval_agg.csv")
    print(f"wrote {args.out / 'eval_agg.csv'}")


if __name__ == "__main__":
    main()
