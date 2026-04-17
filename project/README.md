# spectralrl — RL for Spectral Gap Optimization in Multi-Agent Networks

Implementation for AAE 59000 / MAAC. Learns to maximize $\lambda_2(L)$ of a communication graph under bounded weights and a communication budget, to accelerate discrete-time consensus.

## Install

### Option 1: conda (recommended)

```bash
conda create -n spectralrl python=3.11 -y
conda activate spectralrl
conda install -c conda-forge numpy scipy matplotlib networkx pyyaml pytest -y
conda install -c pytorch pytorch cpuonly -y          # or: pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gymnasium "torchrl>=0.4" "tensordict>=0.4"
pip install -e .
```

Or via the equivalent env file:

```bash
conda env create -f environment.yml
conda activate spectralrl
pip install -e .
```

### Option 2: pip / venv

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Reproduce results

```bash
python scripts/m1_baseline.py        --out results/m1
python scripts/m2_train_reweight.py  --config configs/ppo_default.yaml --out runs/m2
python scripts/m2_eval_reweight.py   --ckpt runs/m2/best.pt --out results/m2
python scripts/m3_train_rewire.py    --config configs/ppo_default.yaml --out runs/m3a
python scripts/m3_robustness.py      --out results/m3
python scripts/m3_geometric.py       --out results/geom
```

## Tests

```bash
pytest -q
```

See `info/` for the proposal and midterm report.
