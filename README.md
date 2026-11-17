# GTA VI Twitch CCU Forecast — make.fun bounty entry

Forecasts hourly Twitch concurrent viewers (CCU) of Grand Theft Auto VI for
hours 2–24 of launch day (19 Nov 2026, UTC), using only the hour-1 print.

## Model

Ensemble in log space (weights 0.7 / 0.3), fitted offline on 37 historical
big-game launch days from the competition kit's comparables. Parameters are
hardcoded constants in `main.py`; at runtime the program reads only the case
file — no network, no comparables, no randomness.

- **Component A (0.7): piecewise log-log regression, per hour.**
  `ln(v_h) ~ b0 + b1·ln(v1) + b2·max(0, ln(v1)−ln(20000)) + b3·max(0, ln(v1)−ln(200000))`
  plus empirical residual quantiles. The hinges capture the dominant
  structure in the data: a small hour-1 print (most regional unlocks still
  pending) precedes explosive growth — median peak ≈ 200× hour 1 — while a
  large hour-1 print (NZ / CET / UK already unlocked) precedes a near-flat
  day, median peak ≈ 1.2×. Residual spread is scaled per hour-1 tier
  (<10k / 10k–100k / >100k): small-hour-1 launches are far more volatile.
- **Component B (0.3): tiered median log-ratio.** Median `ln(v_h/v1)` per
  hour-1 tier plus empirical residual quantiles of the log-ratio.
- **Guards:** median capped at 2500× hour 1, q95 at 20000× hour 1 (max
  observed peak/hour-1 ratio was ~2120×); quantiles clamped to (0, 1e9] and
  forced non-decreasing. Any malformed case still yields a valid
  persistence-style forecast instead of a crash.

## Backtest (leave-one-launch-out, 37 launches, exact server pinball loss on the log scale)

| Model | Loss |
|---|---|
| Persistence (house baseline) | 0.9297 |
| Global median-ratio | 0.7139 |
| Log-log regression | 0.4377 |
| Tiered median-ratio | 0.4676 |
| Piecewise log-log | 0.4025 |
| Piecewise + tier-spread | 0.3824 |
| **Final ensemble (this entry)** | **0.3785** |

In the large-hour-1 regime (n=16, the most plausible for GTA VI): final
0.2231 vs persistence 0.93+. Full tables and validation results:
see `VALIDATION.md` in the contest folder (kept out of this repo).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, deterministic, ~0.05 s per case.

## License

MIT — see `LICENSE`.
