# Final Report Drafts

Drop-in content for the final report (technical paper). Each file is
independent; combine as needed with the IEEEtran template used for the
midterm.

## Files

- `key_idea.tex` — one-paragraph "Key Idea" subsection. Place between
  Problem Formulation and Main Results. Frames the PPO effort as
  convex-relaxation approximation rather than pure benchmarking.
- `sdp_theorem.tex` — theorem-with-proof subsection promoting the
  FDLA SDP bound from "baseline" to theoretical anchor. Include in
  Main Results. Provides the 40-point section's research claim.
- `additional_refs.bib` — BibTeX entries needed by the two `.tex`
  files above. Append to the existing `Reference.bib`. All entries
  are real published works; spot-check page numbers and volume
  numbers against publisher PDFs before final submission.

## Citation accuracy notes

Verified against standard sources:

- **Boyd 2006** — "Convex Optimization of Graph Laplacian Eigenvalues,"
  ICM 2006. The canonical reference for the SDP formulation used in
  `sdp_theorem.tex`.
- **Ghosh & Boyd 2006** — "Growing Well-Connected Graphs," CDC 2006.
  The specific SDP for maximizing algebraic connectivity.
- **Xiao & Boyd 2004** — "Fast Linear Iterations for Distributed
  Averaging," Systems & Control Letters. Foundational FDLA paper.
- **Schulman et al. 2017** — PPO, arXiv:1707.06347.
- **Schulman et al. 2015 (TRPO) / 2016 (GAE)** — the training loop used
  in this work.
- **Kim & Mesbahi 2006** — IEEE TAC. Directly on maximizing lambda_2
  as a state-dependent control problem; strongest parallel to our
  MDP formulation.
- **Fiedler 1973** — already in midterm bib.
- **Mesbahi & Egerstedt 2010** — standard Laplacian-in-MAS textbook.

Citations marked with a `note` field in the `.bib` (e.g.
`Wang2020Fiedler`) are placeholder entries for recent related work;
verify the exact author list and venue before including.

## Target reference count

Midterm: ~10 refs. Final target: ~25–30 (per alignment review,
matching Example 2). Current new entries: 19 additional candidates in
`additional_refs.bib`. Trim to the ones actually cited in the text.
