"""Spawn an existing training script N times with different seeds.

Writes a per-seed copy of the config with ``seed: <s>`` so the existing scripts
don't need a --seed flag. Each seed's run lands in ``<out>/seed_<s>/``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _patch_config(base_cfg: Path, seed: int, dst: Path) -> Path:
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = int(seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f)
    return dst


def train_one_seed(script: Path, base_cfg: Path, seed: int, out: Path, extra: list[str]) -> Path:
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    patched = _patch_config(base_cfg, seed, seed_dir / "config.yaml")
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    cmd = [sys.executable, str(script), "--config", str(patched), "--out", str(seed_dir), *extra]
    print(f"[seed {seed}] launching: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    return seed_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True, help="training script to run")
    parser.add_argument("--config", type=Path, required=True, help="base YAML to patch")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", type=Path, required=True)
    args, extra = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        train_one_seed(args.script, args.config, s, args.out, extra)
    print(f"\nDone. All seeds under {args.out}/seed_*")


if __name__ == "__main__":
    main()
