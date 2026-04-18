# Final Report Alignment Review

Reviewed against `project/info/Resources/`: course syllabus, Mou's Tips PDF, the
`FinalReport.pdf` template, and the two example final reports
(`Example_FinalReport1.pdf` cooperative manipulation; `Example_FinalReprot2.pdf`
UAV / D-UCB).

## 1. Syllabus

Course requires: midterm (problem formulation + intro) + final as a technical
paper with **Motivation, Problem Formulation, Key Idea, Main Results,
Simulations (or Proofs)**. We have material for all five.

## 2. Mou's Tips PDF — process checklist

- Read broadly — proposal/midterm cite Boyd-Ghosh, Mosk-Aoyama,
  Olfati-Saber. ✓
- Picked a topic with open challenges — RL for spectral gap is genuinely
  under-studied. ✓
- **Question existing assumptions** — the tips explicitly call out "do they
  require global information?" Both example reports stress
  *decentralized/implicit* communication. **Our PPO uses global graph state.**
  This is a real critique a grader will raise. Mitigations:
  - (a) acknowledge as a limitation + future-work item, **or**
  - (b) restrict the policy to a per-node MLP that only sees local features
    (cheap to add, big credibility gain).

## 3. Template — point allocation

| Section | Pts | Status |
|---|---|---|
| Abstract | 10 | Need 2–3 sharp sentences |
| Intro & Motivation | 10 | Have it |
| Problem Formulation | 15 | Have eq. (10): max λ₂ − β·cost − γ·violations |
| **Main Results** | **40** | Weakest — see below |
| Simulations | 20 | Lots of CSV/plots; m3a/m3c pending re-runs |
| Conclusion | 5 | Easy to write |

The 40-pt Main Results section dominates the grade. Currently it reads as
engineering, not as a research claim. **The SDP upper bound (Boyd-Ghosh FDLA)
is the theoretical anchor**: state it as a proposition, show PPO reaches 87.7%
of it. Converts a benchmarking exercise into a quantitative optimality claim.

## 4. Example 1 — Cooperative Manipulation (6 pp)

Heavy theory: kinematics, rigidity matrix, **stability proof** (exponential
convergence). One simulation figure. 16 refs. Future-work section enumerates 5
applications.

**Lesson:** strong theoretical narrative carries the report even with one
figure. We have the opposite shape — many figures, weak theory. Borrow the
structure: state the spectral-gap optimization problem, cite the Fiedler bound
on consensus rate `J(k) ≤ ρᵏ J(0)` with `ρ = 1 − αλ₂`, then SDP optimality,
then PPO as approximation.

## 5. Example 2 — UAV / D-UCB (8 pp)

Builds on prior work [25] and improves it (online vs offline). **Propositions
1 & 2 with proofs** (feasibility, sub-optimality). Algorithm 1 pseudocode. 37
refs. **No simulations** — student writes "research still in progress" and
presents only theory + algorithm.

**Lesson:** even with zero simulations this reads as publishable-feeling
because the *key idea* (D-UCB element-wise L∞ bound vs standard L2 ellipsoidal
UCB) is articulated crisply in one paragraph. **Our key-idea sentence is
currently weak.** Draft:

> We frame algebraic-connectivity maximization as a constrained MDP whose
> optimum is upper-bounded by the convex FDLA SDP, and show that a
> policy-gradient agent learns a near-optimal weighting that generalizes to
> discrete rewiring and geometric swarm settings where the SDP no longer
> applies.

## Concrete gaps to close before the final

1. **Add a "Key Idea" paragraph** distilling the contribution in 4–5 sentences
   (use draft above as starting point).
2. **Promote the SDP bound from a baseline to a theorem** in §III. State
   Boyd-Ghosh, prove (or cite) `λ₂ ≤ s*`, position PPO as approximating `s*`.
3. **Acknowledge global-info limitation explicitly.** Optional but cheap
   mitigation: train a per-node MLP that sees only its row of the Laplacian +
   degree neighborhood. Even one ablation row showing "local PPO loses ~10%
   λ₂ vs global" is a valuable result.
4. **Bulk up references** to ~25–30. The midterm's bib has ~10 — light by
   Example 2's standard.
5. **Pseudocode block** like Example 2's "Algorithm 1" — give PPO + SDP a
   numbered algorithm box. Visually signals rigor.
6. **Future Work** with 3–5 concrete directions (decentralized variant,
   dynamic graphs, packet-loss MDP, rate guarantees).

## What we already do well vs the examples

- Multi-seed statistics (neither example reports variance — we're stronger).
- Reproducibility (both examples' code isn't released; we have full repo with
  manifests).
- Multiple environments (M2 reweight, M3a rewire, M3c geometric) — broader
  than either example.
- Convex upper bound as quantitative optimality measure — neither example has
  this.

## Overall verdict

Structurally we'll satisfy the rubric. The risk is the 40-point Main Results
section reading as "we ran PPO" instead of "we made a research claim."
Reframing around the SDP-bound + sub-optimality argument fixes that without
new code.
