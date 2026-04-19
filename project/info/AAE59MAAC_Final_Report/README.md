# Final Report — AAE59MAAC

Reinforcement Learning for Spectral Gap Optimization in Multi-Agent Networks.

## Files

- `FinalReport.tex` — main document (IEEEtran, journal twoside).
- `Reference.bib` — merged bibliography (midterm refs + FDLA/SDP/PPO refs).
- `figures/` — PDFs pulled from `project/results/`.

## Compile

```bash
pdflatex FinalReport.tex
bibtex   FinalReport
pdflatex FinalReport.tex
pdflatex FinalReport.tex
```

Produces `FinalReport.pdf`. Requires a LaTeX distribution with the
`IEEEtran`, `amsmath`, `amssymb`, `algorithm`, `algpseudocode`,
`booktabs`, `graphicx`, and `hyperref` packages (standard TeX Live
full install).

## Numeric claims and their sources

All numbers in the report trace back to CSVs in `project/results/`.

| Claim | Source |
|---|---|
| PPO $\lambda_2 = 0.795 \pm 0.146$ vs. uniform $0.607 \pm 0.119$ (25 held-out graphs) | `results/m2_er_tight/eval_agg.csv` |
| PPO reaches $87.3\%$ of the FDLA SDP on seed 0 ($s^\star = 0.660$) | `results/m2_er_tight/sdp.json`, `results/m2_er_tight/seed_0/eval.csv` |
| Clean-trained PPO beats baselines at $p \in \{0, 0.1, 0.2, 0.3\}$ | `results/m3b_clean/robustness.csv` |
| PPO $\lambda_2 = 0.34$ vs. constrained centroid $0.30$, random $0.27$ | `results/m3c_v3/eval.csv` |
| PPO loses to random rewiring on 10/10 held-out seeds | `results/m3a_v3/eval.csv` |

## Notes

- The Key Idea paragraph (§III preamble) and the SDP theorem with proof (§III.B) were drafted in `project/info/final_report_drafts/` and are inlined here. That directory can be deleted once this report is finalized.
- The midterm template at `project/info/AAE59MAAC_Midterm_Report/WeeklyReadingReport/MidtermReport_Template.tex` was used as the structural scaffold for the preamble.
