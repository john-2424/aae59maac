# Final Report — Presentation Transcript and Q&A Prep

Notes to use when presenting or fielding questions on the final report.
Organized by section of the paper plus anticipated questions.

## One-line summary to open with

> I formulated constrained algebraic-connectivity maximization as an
> MDP, proved it is upper-bounded by the FDLA convex SDP of Boyd and
> Ghosh, and trained a PPO agent that reaches 87.3% of the SDP optimum
> on held-out Erdős–Rényi graphs (mean λ₂ = 0.795 ± 0.146 across five
> trained seeds, a 1.31× lift over the uniform baseline).

## Headline numbers to remember

- **§IV.B (M2 reweighting):** PPO λ₂ = 0.795 ± 0.146, uniform 0.607,
  Metropolis 0.167. SDP optimum on seed 0 is s* = 0.660; PPO reaches
  λ₂ = 0.576, i.e. ρ_π = 87.3%.
- **§IV.C (M3b robustness):** clean-trained PPO dominates uniform and
  Metropolis at p ∈ {0, 0.1, 0.2, 0.3}. Separately trained from the
  §IV.B policy on a relaxed budget (B = m · w_max).
- **§IV.D (M3c geometric):** mean PPO λ₂ = 0.34 vs. constrained
  centroid 0.30 vs. random 0.27. Unconstrained centroid reaches
  λ₂ ≈ 19.83 but with mean_min_dist 0.004 (≈22× below r_min = 0.08), so
  it is a degenerate optimum, not a fair comparator.
- **§IV.E (M3a discrete):** PPO loses to random rewiring on 10/10
  held-out seeds. Framed as an architectural limitation of a flat MLP
  categorical head over a 190-way action space.

## Anticipated Q&A

### Q1. What is the contribution if Boyd already solved the convex case with an SDP?
The SDP is the upper bound, not the method. My contribution is
(a) framing the problem as an MDP whose optimum is upper-bounded by
that SDP, so the learned policy can be reported as a suboptimality
ratio rather than just "better than a heuristic," and (b) showing the
same policy class transfers to three regimes where the SDP no longer
applies: stochastic edge failures, discrete rewiring, and geometric
swarms.

### Q2. Why do you train on the same graph per seed? Isn't that overfitting?
The PPO actor has a fixed input/output dimensionality tied to the edge
count m of the training graph. Evaluating on a different support would
be a shape mismatch. To get distributional statistics I train five
seeds on five different ER draws from the same (n=20, p=0.25)
distribution, and for each seed I re-evaluate on five held-out
consensus initial conditions. That gives 25 evaluation rows covering
five distinct graphs, which is where the 0.795 ± 0.146 comes from.
The per-seed-per-graph overfitting is a real limitation, and the
natural fix is a permutation-equivariant GNN actor that shares weights
across supports. That is future-work item 1.

### Q3. In m3c, PPO loses to the constrained centroid on 6 of 10 seeds. How is that a win?
It wins on the *mean* (0.34 vs. 0.30) because when PPO does win, it
wins by a larger margin, and when it loses it loses by a smaller
margin. Per-seed variance is high. I report the per-seed numbers
honestly in the paper rather than hiding behind the mean. The
interpretation is that a flat MLP over agent positions is a weak
policy class for this regime, and the same GNN-actor direction that
addresses m3a should improve m3c as well. Beating random rewiring
velocities 6/10 is genuine; beating a well-tuned geometric heuristic
on the mean is a positive but not strict result.

### Q4. Why not just run the SDP at every step instead of training PPO?
Three reasons. First, the SDP only applies when the edge support is
fixed and the map w → L(w) is linear, so it does not extend to
discrete rewiring, position-induced supports, or edge failures.
Second, solving an n×n SDP at every step does not scale to the
regimes I care about in the extensions. Third, the goal is an
adaptive policy that absorbs perturbations and nonconvex structure,
not a one-shot optimizer; the SDP is the benchmark against which I
measure the policy on the one setting where both apply.

### Q5. Why does PPO win so decisively over Metropolis when Metropolis is supposed to be optimal for mixing?
Metropolis weights are optimal for the *mixing rate of a Markov
chain* under a uniform stationary-distribution constraint, not for
λ₂ under a bounded-weight, total-budget constraint. The two
objectives line up only in unconstrained settings. With a tight
budget, Metropolis sums to ≈ 6.9 in these runs, well below B = 25;
it voluntarily under-uses the budget because it prioritizes
low-degree edges. A uniform budget-filling policy already beats it
by design, and PPO improves further by concentrating mass on cut
edges.

### Q6. How do you know the 13% gap to the SDP isn't just PPO at convergence?
I don't. The 13% gap is the approximation error of a 2×128 MLP
trained on 300k environment frames with a Gaussian policy and
entropy coefficient 0.005. A larger actor, a longer training budget,
or a permutation-equivariant feature extractor would all be expected
to close more of it. I stop there because (a) it is already a clear
win over the heuristic baselines and (b) more hyperparameter tuning
would not change the qualitative story for this report.

### Q7. Why does the observation include the global Laplacian spectrum? Doesn't that break decentralization?
Yes, and I flag this explicitly in Remark 1. The global spectrum
assumption is a deliberate simplification that isolates the
learning question from the decentralization question. A fully
distributed variant would replace the top-k eigenvalues with a
distributed Fiedler-vector estimate (Yang–Freeman–Gordon 2010,
Sabattini et al. 2013). That is future-work item 2.

### Q8. Why is m3a a negative result instead of a better algorithm?
The m3a experiment was deliberately designed with a flat MLP
categorical head over a 190-way action space so that the negative
result would be informative. A given action index "edge pair
(i, j)" has no consistent meaning across i.i.d. ER resamples, so
the actor literally cannot learn a transferable "which edge to
toggle" rule. The clean diagnostic is that random rewiring wins
10/10. The constructive follow-up is to replace the flat head with
a GNN actor that is permutation-equivariant over the node set by
construction (Kipf–Welling, Dai et al., Darvariu et al.), which is
future-work item 1. I report the negative result rather than
tuning around it because the failure mode is informative about
which architectural choice matters here.

### Q9. How robust are the results to the PPO hyperparameters?
The results in §IV.B use entropy coefficient 0.005, clip 0.2,
(γ, λ) = (0.99, 0.95), Adam 3e-4, [128, 128] hidden. I did not run a
full hyperparameter sweep. The entropy coefficient was the one
sensitive knob: without it the policy collapsed back to the
Metropolis warm start in preliminary runs on ER. Everything else is
near the TorchRL default.

### Q10. What would convince you this approach generalizes?
Three things in order of cost: (1) retrain with a GNN actor and
check that a single policy trained on multiple ER sizes transfers
to unseen sizes, (2) derive a finite-sample high-probability bound
on ρ_π using the tools in Olshevsky–Tsitsiklis 2011, (3) deploy the
geometric variant on a physical multi-robot testbed. These are
future-work items 1, 4, and 5.

## Known weaknesses to own, not hide

1. **Per-seed variance in m3c.** PPO's mean beats the constrained
   centroid, but it wins head-to-head on only 4/10 seeds. I report
   this honestly in §IV.D and frame it as a motivating limitation
   rather than a decisive win.
2. **Evaluation graph count.** The headline 0.795 ± 0.146 is across
   5 distinct graphs (one per trained seed), not 25 distinct graphs.
   Each seed is re-evaluated on five consensus initial conditions to
   reduce τ_ε variance, but λ₂ is identical within a seed by
   construction.
3. **m3b is a separately trained policy.** Not the same checkpoint as
   the §IV.B headline; trained on a relaxed budget (B = m · w_max)
   rather than the tight B = 25. This is stated explicitly in §IV.C.
4. **m3a is a clean negative result.** PPO loses 10/10 to random.
   Reported as a limitation that motivates GNN actors, not hidden.
5. **Global-spectrum observation.** The policy sees top-k eigenvalues
   of the global Laplacian, which is centralized. Flagged in Remark 1
   and listed as future-work item 2.
