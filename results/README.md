# What is in `results/`, and which parts are stale

These files were produced before the 2026-08-24 audit. The input data is not
committed and requires an Alpaca key, so they **cannot be regenerated here** —
they are kept as the record of what was actually run, not as output of the
current code. Two columns therefore no longer mean what the code that reads
their name would produce.

## `backtest_results.csv` — two columns are stale, and are named accordingly

| Column | Status |
|---|---|
| `payoff_ratio_stale` | Was committed as `profit_factor`, but the value is `avg_win / avg_loss` — a **payoff ratio**, not a profit factor. Profit factor is gross profit over gross loss, which also needs the win and loss *counts*. At a 24.6% win rate the two differ by roughly 3×: SPY 1-min reads **0.908** here, where the true profit factor is **0.296**. The mislabelled version flattered the strategy. |
| `bh_sharpe_stale` | Buy-and-hold Sharpe computed with a hardcoded 390 bars/day. This sample includes extended hours — about 774 bars/day — so the benchmark's volatility was understated by √(774/390) ≈ 1.41×, inflating the magnitude of this figure by the same factor and handicapping the control the strategy is measured against. Corrected values would be roughly **−0.176** (QQQ) and **−0.084** (SPY), against the −0.249 / −0.119 recorded here. |

Both were renamed rather than deleted so the committed record stays intact and
cannot be quoted under a name that implies something else. `src/backtest.py`
now emits `profit_factor` (correct definition) and `payoff_ratio` as separate
columns, and computes buy-and-hold volatility from the actual bar count — so a
re-run produces different column names from the ones in this file. That
mismatch is intentional and is the point of the `_stale` suffix.

## Columns that are NOT stale

`ann_return_pct` and `sharpe` still match what the current code produces.
The simple (uncompounded) annualization that generates figures like −884% was
deliberately **kept**, because the committed results were produced with it and
changing it silently would make this file unreadable against its own history.
It is a units artifact, not a return — a long/short book cannot lose more than
everything — and `src/backtest.py` now reports
`annualized_return_compounded`, `per_trade_sharpe` and an
`annualization_implausible` flag alongside it so it cannot be quoted alone.
Those three columns, and `payoff_ratio`, do not exist in this file; a re-run
would add them.

`total_return_pct`, `win_rate_pct`, `avg_bps`, `random_sharpe_mean` and
`random_sharpe_p95` are unaffected. These are the columns the top-level README
quotes.

## `summary_stats.csv`

Unaffected by the audit. `src/signal_analysis.py` was not changed.

Note that the signal every row is computed on, `ofi_norm`, reduces
algebraically to `(close − open) / (high − low)` — the volume term cancels. See
the top of the repository README.
