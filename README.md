# aae59maac

**Reinforcement Learning for Spectral Gap Optimization in Multi-Agent Networks.**

Course project for AAE 59000 / MAAC (Multi-Agent Autonomy and Control),
MS Autonomy program, Purdue University, Spring 2026. The project
learns to maximize the algebraic connectivity $\lambda_2(L)$ of a
communication graph, which is the quantity that controls how fast a
linear consensus protocol converges in a multi-agent network.

The policy is a PPO agent. The benchmark it is measured against is
the Fastest Distributed Linear Averaging (FDLA) convex semidefinite
program of Boyd and Ghosh. On the setting where the SDP applies, PPO
reaches **87.3% of the convex optimum** on a held-out Erdős–Rényi
graph and a mean $\lambda_2 = 0.795 \pm 0.146$ across five trained
seeds (1.31× the uniform baseline). The same policy class is then
carried over to three regimes where the SDP no longer applies:
stochastic edge failures, discrete rewiring, and geometric swarm
control.

Full write-up: `project/info/AAE59MAAC_Final_Report/FinalReport.tex`.

---

## Table of contents

- [Headline results](#headline-results)
- [Repository layout](#repository-layout)
- [Install](#install)
- [Quickstart: reproduce the paper](#quickstart-reproduce-the-paper)
- [Per-milestone reproduction](#per-milestone-reproduction)
  - [M1 — baseline topology / spectral gap / convergence](#m1--baseline-topology--spectral-gap--convergence)
  - [M2 — PPO reweighting vs. baselines vs. the SDP bound](#m2--ppo-reweighting-vs-baselines-vs-the-sdp-bound)
  - [M3a — discrete rewiring (negative result)](#m3a--discrete-rewiring-negative-result)
  - [M3b — perturbation-robust reweighting](#m3b--perturbation-robust-reweighting)
  - [M3c — geometric swarm](#m3c--geometric-swarm)
- [Running the tests](#running-the-tests)
- [Building the final report PDF](#building-the-final-report-pdf)
- [Configs, checkpoints, and outputs](#configs-checkpoints-and-outputs)
- [Key source files](#key-source-files)
- [Known limitations](#known-limitations)
- [References](#references)

---

## Headline results

| Milestone | Setup | Number that matters |
|---|---|---|
| **M2** | PPO vs. classical baselines, Erdős–Rényi $n{=}20$, $B{=}25$, 5 seeds | PPO $\lambda_2 = 0.795 \pm 0.146$ vs. uniform $0.607 \pm 0.119$, Metropolis $0.167 \pm 0.033$ |
| **M2** | PPO vs. FDLA SDP, seed 0 | $s^\star = 0.660$, PPO reaches $\lambda_2 = 0.576$, i.e. **87.3% of SDP** |
| **M3b** | Clean-trained PPO under i.i.d. edge failures | PPO dominates uniform and Metropolis at $p \in \{0, 0.1, 0.2, 0.3\}$ |
| **M3c** | Geometric swarm, 10 evaluation seeds | PPO mean $\lambda_2 = 0.34$ vs. constrained centroid $0.30$ vs. random $0.27$ |
| **M3a** | Discrete rewiring, flat-MLP categorical head | PPO loses to random rewiring on **10/10** held-out seeds (negative result, motivates GNN actor) |

All numbers trace back to CSVs in `project/results/`. See
`project/info/AAE59MAAC_Final_Report/README.md` for the per-claim
source-file table.

---

## Repository layout

```
aae59maac/
├── LICENSE
├── README.md                  <-- you are here
└── project/
    ├── pyproject.toml         # editable install: pip install -e project
    ├── requirements.txt       # pip-based install alternative
    ├── environment.yml        # conda env
    ├── README.md              # install / quickstart
    ├── configs/               # YAML configs (one per milestone variant)
    ├── scripts/               # entry points: train / eval / plot
    ├── src/spectralrl/        # the library
    │   ├── graphs/            # generators, Laplacian, Fiedler value
    │   ├── consensus/         # discrete-time consensus dynamics + metrics
    │   ├── baselines/         # uniform / Metropolis / degree-prop / FDLA SDP
    │   ├── envs/              # Gymnasium envs (reweight, rewire, geometric)
    │   ├── rl/                # PPO training loop, actor/critic, eval utilities
    │   ├── robustness/        # edge-failure perturbation wrappers
    │   └── utils/             # logging + seeding helpers
    ├── tests/                 # pytest suite
    ├── runs/                  # checkpoints from each training run
    ├── results/               # eval CSVs, plots, SDP JSONs
    └── info/
        ├── AAE59MAAC_Final_Report/   # final report LaTeX + refs + figures
        ├── AAE59MAAC_Midterm_Report/ # midterm report
        ├── AAE59MAAC_Project_Proposal/
        ├── final_report_drafts/      # inputs inlined into the final report
        └── Resources/                # syllabus, rubric, paper PDFs
```

---

## Install

The package lives under `project/` and installs as `spectralrl`.
Everything below assumes you are `cd`-ed into `project/`.

### Option 1 — conda (recommended)

```bash
cd project
conda env create -f environment.yml
conda activate spectralrl
pip install -e .
```

Or manually:

```bash
conda create -n spectralrl python=3.11 -y
conda activate spectralrl
conda install -c conda-forge numpy scipy matplotlib networkx pyyaml pytest -y
conda install -c pytorch pytorch cpuonly -y          # or: pytorch-cuda=12.1 -c pytorch -c nvidia
pip install gymnasium "torchrl>=0.4" "tensordict>=0.4" "cvxpy" "scs"
pip install -e .
```

### Option 2 — pip / venv

```bash
cd project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Verify the install

```bash
pytest -q
```

CVXPY + SCS are only needed for the FDLA SDP solver in
`scripts/m2_sdp_bound.py`. Everything else works without them.

---

## Quickstart: reproduce the paper

The minimum sequence to regenerate every number that appears in the
final report, assuming a working install and `cd project/`:

```bash
# M1: baseline topology / lambda2 / convergence
python scripts/m1_baseline.py --out results/m1

# M2: train 5 PPO seeds on the tight-budget ER setup, evaluate, and
#     compute the FDLA SDP bound on seed 0
python scripts/run_multiseed.py \
    --script scripts/m2_train_reweight.py \
    --config configs/ppo_m2_er_tight.yaml \
    --seeds 0 1 2 3 4 \
    --out runs/m2_er_tight
python scripts/m2_eval_multiseed.py \
    --runs-dir runs/m2_er_tight \
    --out results/m2_er_tight \
    --sdp-json results/m2_er_tight/sdp.json
python scripts/m2_sdp_bound.py \
    --ckpt runs/m2_er_tight/seed_0/best.pt \
    --out results/m2_er_tight/sdp.json

# M3a: discrete rewiring (negative result)
python scripts/m3_train_rewire.py --config configs/ppo_m3a_v3.yaml --out runs/m3a_v3
python scripts/m3a_eval_rewire.py --ckpt runs/m3a_v3/best.pt       --out results/m3a_v3

# M3b: perturbation-robust reweighting
python scripts/m2_train_reweight.py --config configs/ppo_m2_er.yaml         --out runs/m2_er_ms/seed_0
python scripts/m2_train_reweight.py --config configs/ppo_m2_er_perturb.yaml --out runs/m2_er_perturb
python scripts/m3_robustness.py --ckpt runs/m2_er_ms/seed_0/best.pt  --out results/m3b_clean   --label clean
python scripts/m3_robustness.py --ckpt runs/m2_er_perturb/best.pt    --out results/m3b_perturb --label perturb

# M3c: geometric swarm
python scripts/m3c_train_geometric.py --config configs/ppo_m3c_v3.yaml --out runs/m3c_v3
python scripts/m3c_eval_geometric.py  --ckpt runs/m3c_v3/best.pt      --out results/m3c_v3
```

Total training time on a CPU is ~20 minutes for each M2/M3b seed and
~5 minutes for M3a/M3c. GPU is not required and the configs default
to CPU.

---

## Per-milestone reproduction

Each script prints the CSVs and plot paths it writes. The numbers
quoted below are from `project/results/*` and are the ones used in
the final report.

### M1 — baseline topology / spectral gap / convergence

Establishes the topology $\to \lambda_2 \to \tau_\varepsilon$ chain
on five graph families: ring, grid, Erdős–Rényi, random geometric,
Watts–Strogatz (small-world), at sizes $n \in \{20, 50, 100\}$.

```bash
python scripts/m1_baseline.py --out results/m1
```

Outputs:
- `results/m1/summary.csv` — one row per (family, n)
- `results/m1/lambda2_vs_edges.pdf` — Figure 1 in the report
- `results/m1/J_vs_k.pdf` — disagreement-energy decay example

### M2 — PPO reweighting vs. baselines vs. the SDP bound

Trains a continuous-action PPO policy that reweights a fixed
Erdős–Rényi support subject to a tight budget $B = 25$ (with
$m \approx 42$ so uniform cannot fill every edge to the bound).

**Train 5 seeds:**

```bash
python scripts/run_multiseed.py \
    --script scripts/m2_train_reweight.py \
    --config configs/ppo_m2_er_tight.yaml \
    --seeds 0 1 2 3 4 \
    --out runs/m2_er_tight
```

This spawns five independent training runs into
`runs/m2_er_tight/seed_{0..4}/`, each producing a `best.pt`
checkpoint and a `train_log.csv`.

**Aggregate and evaluate:**

```bash
python scripts/m2_eval_multiseed.py \
    --runs-dir runs/m2_er_tight \
    --out results/m2_er_tight
```

Writes `results/m2_er_tight/eval_agg.csv` (Table 1 in the report)
and per-seed eval plots.

**Compute the FDLA SDP bound for seed 0:**

```bash
python scripts/m2_sdp_bound.py \
    --ckpt runs/m2_er_tight/seed_0/best.pt \
    --out results/m2_er_tight/sdp.json
```

Writes `sdp.json` with `lambda2_sdp = 0.6599`. Divide the seed-0 PPO
$\lambda_2$ by this to get the 87.3% claim.

**Single-seed variant** (no multi-seed overhead):

```bash
python scripts/m2_train_reweight.py --config configs/ppo_m2_er_tight.yaml --out runs/m2_single
python scripts/m2_eval_reweight.py  --ckpt runs/m2_single/best.pt         --out results/m2_single
```

### M3a — discrete rewiring (negative result)

Replaces continuous reweighting with a categorical action over
$\binom{n}{2} = 190$ edge pairs. Reported as a negative result: a
flat MLP categorical head loses to random rewiring on 10/10 held-out
seeds because the action index has no consistent meaning across
resampled ER supports.

```bash
python scripts/m3_train_rewire.py  --config configs/ppo_m3a_v3.yaml --out runs/m3a_v3
python scripts/m3a_eval_rewire.py  --ckpt runs/m3a_v3/best.pt       --out results/m3a_v3
```

Outputs:
- `results/m3a_v3/eval.csv` — per-seed final $\lambda_2$ for PPO / random / greedy
- `results/m3a_v3/lambda2_final_bar.pdf` — Figure 5 in the report

Pass `--skip-greedy` to drop the (slow) greedy-swap oracle.

### M3b — perturbation-robust reweighting

Evaluates two PPO policies under i.i.d. Bernoulli edge failures:
(i) a clean-trained policy and (ii) a policy trained with
perturbation-in-the-loop at $p = 0.15$. Both are trained on the
*relaxed* budget setup (`budget: null`, i.e. $B = m \cdot w_{\max}$),
not the tight $B = 25$ of M2.

**Train the two policies:**

```bash
python scripts/m2_train_reweight.py --config configs/ppo_m2_er.yaml         --out runs/m2_er_ms/seed_0
python scripts/m2_train_reweight.py --config configs/ppo_m2_er_perturb.yaml --out runs/m2_er_perturb
```

**Evaluate robustness at $p \in \{0, 0.1, 0.2, 0.3\}$:**

```bash
python scripts/m3_robustness.py --ckpt runs/m2_er_ms/seed_0/best.pt --out results/m3b_clean   --label clean
python scripts/m3_robustness.py --ckpt runs/m2_er_perturb/best.pt   --out results/m3b_perturb --label perturb
```

Each writes `robustness.csv` with rows `(label, failure_p, policy, lambda2, tau)` and the plots in Figure 3.

### M3c — geometric swarm

$n = 20$ agents in the unit square; edges weighted by a truncated
Gaussian kernel $W_{ij} = \exp(-(\|p_i - p_j\|/r)^2)$ with $r = 0.25$
and cutoff $3r$. Per-agent planar velocity action, $v_{\max} = 0.02$,
collision radius $r_{\min} = 0.08$.

```bash
python scripts/m3c_train_geometric.py --config configs/ppo_m3c_v3.yaml --out runs/m3c_v3
python scripts/m3c_eval_geometric.py  --ckpt runs/m3c_v3/best.pt      --out results/m3c_v3
```

Outputs:
- `results/m3c_v3/eval.csv` — per-seed final $\lambda_2$ and `mean_min_dist` for PPO / centroid / constrained-centroid / random
- `results/m3c_v3/lambda2_curves.pdf` — Figure 4 in the report
- `results/m3c_v3/positions_ppo.pdf` — example final configuration

The eval also reports the *unconstrained* centroid baseline, which
reaches $\lambda_2 \approx 19.8$ by collapsing agents to a single
point. `mean_min_dist` for that baseline is $\sim 0.004$, about $22\times$
below $r_{\min}$, so it is reported for completeness only.

---

## Running the tests

```bash
cd project
pytest -q
```

The suite covers:

- `test_graphs.py` — generators produce symmetric, nonnegative,
  connected supports; closed-form ring and complete eigenvalues.
- `test_consensus.py` — stable step size, geometric decay of the
  disagreement energy, convergence-time / rate metrics.
- `test_baselines.py` — uniform / Metropolis / degree-proportional
  weight matrices match expected formulas and respect the budget.
- `test_envs.py` — observation shapes, action clipping, budget
  projection, and reward computation for all three envs.
- `test_reward.py` — reward components (spectral gap, cost,
  violation) composed correctly.

---

## Building the final report PDF

```bash
cd project/info/AAE59MAAC_Final_Report
pdflatex FinalReport.tex
bibtex   FinalReport
pdflatex FinalReport.tex
pdflatex FinalReport.tex
```

Requires a LaTeX distribution with IEEEtran, amsmath, amssymb,
algorithm, algpseudocode, booktabs, graphicx, and hyperref
(standard TeX Live full install).

See `project/info/AAE59MAAC_Final_Report/README.md` for a table that
traces every numeric claim in the paper back to a CSV in
`project/results/`, and `presentation_qna.md` in the same folder for
presentation talking points and anticipated Q&A.

---

## Configs, checkpoints, and outputs

- **Configs** (`project/configs/*.yaml`) are the source of truth for
  every hyperparameter. The ones actually used in the final report:
  - `ppo_m2_er_tight.yaml` — §IV.B headline, tight budget $B{=}25$
  - `ppo_m2_er.yaml`       — §IV.C clean policy, relaxed budget
  - `ppo_m2_er_perturb.yaml` — §IV.C perturbation-trained
  - `ppo_m3a_v3.yaml`      — §IV.E discrete rewiring
  - `ppo_m3c_v3.yaml`      — §IV.D geometric swarm
- **Checkpoints** land under `project/runs/<milestone>/` as
  `best.pt` (lowest-eval-loss checkpoint) plus a `config.yaml` copy.
- **Results** land under `project/results/<milestone>/` as CSVs
  (data of record) plus PDFs/PNGs for the report figures.

The version suffixes (`_v2`, `_v3`) are iterations. `v3` is always the
one used in the final report; earlier versions are kept for history.

---

## Key source files

If you are reading the code rather than running it, start here:

| File | What it does |
|---|---|
| `src/spectralrl/envs/reweight_env.py` | Continuous-action reweighting env (M2, M3b) |
| `src/spectralrl/envs/rewire_env.py` | Discrete-action edge toggle env (M3a) |
| `src/spectralrl/envs/geometric_env.py` | Position-controlled swarm env (M3c) |
| `src/spectralrl/envs/common.py` | Feature extractor, reward composer, budget projection |
| `src/spectralrl/rl/train_ppo.py` | PPO training loop (TorchRL) |
| `src/spectralrl/rl/policy.py` | Actor + critic MLPs |
| `src/spectralrl/rl/eval.py` | Policy-vs-baselines evaluator |
| `src/spectralrl/baselines/sdp.py` | FDLA SDP solver (CVXPY + SCS) |
| `src/spectralrl/baselines/weights.py` | Uniform / Metropolis / degree-proportional |
| `src/spectralrl/graphs/laplacian.py` | Laplacian construction, Fiedler value |
| `src/spectralrl/graphs/generators.py` | Ring / grid / ER / RGG / Watts–Strogatz |
| `src/spectralrl/consensus/dynamics.py` | Discrete-time consensus iteration |
| `src/spectralrl/robustness/perturbations.py` | i.i.d. edge-failure wrapper |

---

## Known limitations

Stated in the paper and in `presentation_qna.md`; summarized here:

1. **Per-seed variance in M3c.** PPO's mean beats the constrained
   centroid, but it wins head-to-head on only 4/10 seeds. Reported
   honestly, not hidden.
2. **Evaluation graph count.** The headline $0.795 \pm 0.146$ is
   across 5 distinct graphs (one per trained seed), with each seed
   re-evaluated on 5 consensus initial conditions. $\lambda_2$ is
   identical within a seed by construction.
3. **M3b uses a separately trained policy.** Not the same checkpoint
   as the M2 headline; trained on the relaxed budget
   ($B = m \cdot w_{\max}$).
4. **M3a is a clean negative result.** PPO loses 10/10 to random
   rewiring. Framed as a motivating limitation for a GNN actor
   (future work item 1), not as a successful method.
5. **Global-spectrum observation.** The policy sees top-$\kappa$
   eigenvalues of the global Laplacian, which is a centralized
   assumption. Flagged in Remark 1 of the paper and listed as
   future-work item 2.

---

## References

Full bibliography in `project/info/AAE59MAAC_Final_Report/Reference.bib`.
The load-bearing citations are:

- Fiedler (1973) — algebraic connectivity.
- Olfati-Saber, Fax, Murray (2007) — consensus convergence rate.
- Boyd, Diaconis, Xiao (2004); Xiao & Boyd (2004) — Fastest Mixing
  Markov Chain / FDLA.
- Boyd (2006); Ghosh & Boyd (2006) — convex SDP for $\lambda_2$
  maximization (Theorem 1 in the report).
- Mosk-Aoyama (2008) — NP-hardness of $\lambda_2$ maximization over
  topologies.
- Schulman et al. (2017) — PPO; Schulman et al. (2016) — GAE.
- Kipf & Welling (2017); Dai et al. (2017); Darvariu et al. (2021,
  2024) — graph neural networks and graph RL for combinatorial design.
- Sabattini et al. (2013); Yang et al. (2010); Kim & Mesbahi (2006)
  — decentralized connectivity maintenance.
- Mesbahi & Egerstedt (2010) — graph-theoretic methods in multi-agent
  networks (textbook).

---

## License

MIT. See `LICENSE`.

## Author

Shrikrishna Rajule — MS Autonomy, Purdue University.
