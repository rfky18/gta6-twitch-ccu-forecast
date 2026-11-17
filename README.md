# GTA VI Twitch CCU Forecast — make.fun bounty entry

Forecasts hourly Twitch concurrent viewers (CCU) of Grand Theft Auto VI for
hours 2–24 of launch day (19 Nov 2026, UTC), using only the hour-1 print.

## Model (v2)

Ensemble in log space (weights 0.4 / 0.3 / 0.3), fitted offline on 37 historical
big-game launch days from the competition kit's comparables. Parameters are
hardcoded constants in `main.py`; at runtime the program reads only the case
file — no network, no randomness. Components B and C refit at runtime from
an embedded table of the 37 launches' log-viewer curves (training data from
the kit, not practice answers).

- **Component A (0.4): the v1 model.** Per-hour piecewise log-log regression
  (knots at 20k / 200k hour-1 viewers) with tier-scaled residual quantiles,
  averaged 70/30 with a tiered median log-ratio model; median capped at
  2500× hour 1, q95 at 20000× hour 1. The hinges capture the dominant
  structure in the data: a small hour-1 print (most regional unlocks still
  pending) precedes explosive growth — median peak ≈ 200× hour 1 — while a
  large hour-1 print (NZ / CET / UK already unlocked) precedes a near-flat
  day, median peak ≈ 1.2×.
- **Component B (0.3): kernel-weighted local-linear log-ratio forecaster.**
  Per hour, a Gaussian-kernel (bandwidth 1.5 in ln hour-1) weighted
  local-linear fit of `ln(v_h)` on `ln(v1)`; quantiles are a 50/50 shrink of
  weighted empirical residual quantiles toward Gaussian quantiles, which
  stabilises the tails with only ~37 points. Falls back to the global OLS
  fit when the effective sample is tiny.
- **Component C (0.3): 2-D local-linear forecaster pooling hours.** One
  weighted fit per target hour over all 37 launches × 23 hours, Gaussian
  kernels in ln hour-1 (bandwidth 2.0) and in hour (bandwidth 3.0);
  quantiles from weighted residual quantiles.
- **Combination:** the three components' quantiles are averaged in log
  space, then the spread is widened ×1.05 around the median. Quantiles are
  clamped to (0, 1e9] and forced non-decreasing. Any malformed case still
  yields a valid persistence-style forecast instead of a crash.

## Backtest (leave-one-launch-out, 37 launches, exact server pinball loss on the log scale)

| Model | Loss |
|---|---|
| Persistence (house baseline) | 0.9297 |
| Global median-ratio | 0.7139 |
| Log-log regression | 0.4377 |
| Tiered median-ratio | 0.4676 |
| Piecewise log-log | 0.4025 |
| Piecewise + tier-spread | 0.3824 |
| **v1 ensemble** | **0.3785** |
| Kernel local-linear (bw 1.5, shrink 0.5) | 0.3739 |
| Hour-pooled 2-D local-linear (2.0 / 3) | 0.3764 |
| **v2 ensemble (this entry)** | **0.3686** |

In the large-hour-1 regime (n=16, the most plausible for GTA VI): v2 0.2126
vs v1 0.2231 vs persistence 0.93+. **Candidate status: the v2 gate (LOLO <
0.30) was NOT met** — 0.3686 is only 2.6% better than v1, inside backtest
noise — so this entry is kept as a ready candidate, not a validated
improvement. Full tables and validation results: see `VALIDATION.md` in the
contest folder (kept out of this repo).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, ~0.15 s per case.

## License

MIT — see `LICENSE`.
