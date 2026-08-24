# Order Flow Imbalance (OFI) Signal Research

Empirical analysis of Order Flow Imbalance as a short-term price predictor for US equities, using minute-level data from Alpaca Markets.

> ## Read this before the results
>
> **The signal every headline number below is computed on contains no order-flow
> information.** `ofi_raw = (close − open) / (high − low + ε) × volume`, and
> `ofi_norm = ofi_raw / volume`. The volume term cancels exactly, so
>
> ```
> ofi_norm ≡ (close − open) / (high − low)
> ```
>
> — the candlestick body-to-range ratio. It is scale-free and it is a
> reasonable microstructure feature, but it is not an order-flow imbalance and
> it does not use volume at all. `ofi_norm` is nonetheless what the analysis
> selected as the best variant (`src/signal_analysis.py:216`,
> `src/backtest.py:207`) and what every table in this README reports. See
> `src/ofi_calculator.py` for the identity written out.
>
> The variants that *do* retain volume — `ofi_roll5/10/20` — are the ones with
> IC near zero and p > 0.05 across most of `results/summary_stats.csv`. So the
> honest one-line summary of this repository is: **a candlestick body/range
> ratio has a small negative IC at minute frequency; the volume-weighted OFI
> proxies this project set out to test do not.**
>
> This was found on 2026-08-24 during a self-audit and is stated here rather
> than quietly repaired, because the results below cannot be re-run — the input
> data is not committed (see Reproduce) and requires an Alpaca key.

## Key Findings

### Signal has a small, statistically significant negative IC

The OFI signal shows a **mean reversion** relationship with future returns — the opposite of the momentum effect typically documented in academic literature:

| Symbol | Signal | Horizon | IC | p-value (nominal) | n |
|--------|--------|---------|----:|--------:|----:|
| SPY | ofi_norm | 1 min | **-0.032** | 2.2e-22 | 92,970 |
| SPY | ofi_norm | 5 min | **-0.022** | 3.2e-11 | 92,966 |
| QQQ | ofi_norm | 1 min | **-0.024** | 2.6e-13 | 94,977 |
| QQQ | ofi_norm | 5 min | **-0.014** | 3.1e-05 | 94,973 |

*(An earlier version of this table gave these as "<1e-22", "<1e-11", "<1e-13"
and "<1e-5". All four bounds were tighter than the truth, and the last one was
contradicted by this repository's own `results/summary_stats.csv`, which records
3.1e-05. The values above are recomputed from the committed ICs and sample
sizes.)*

**Three reasons not to read those p-values as evidence of alpha.**

1. **The signal and the forward return share `close[t]`.** `ofi_norm[t]` has
   `close[t]` in its numerator and `fwd_ret_1[t] = close[t+1]/close[t] − 1` has
   it in its denominator. Bid-ask bounce in `close[t]` induces a negative
   correlation between them mechanically, with no information involved. This is
   the leading explanation for the sign, not one candidate among three.
2. **Observations overlap.** `src/signal_analysis.py:24` computes
   `pct_change(h).shift(-h)`, so at h = 60 consecutive rows share 59 of 60
   minutes, while `scipy.stats.spearmanr` assumes independent pairs. The
   effective sample at h = 60 is closer to 1,550 than 92,911, which inflates
   the reported t by roughly √60 ≈ 7.7×.
3. **`ofi_norm` is the winner of 32 comparisons** (2 symbols × 4 signals ×
   4 horizons) and no correction was applied for that search. Seventeen of the
   32 rows in `results/summary_stats.csv` have p > 0.05; only the winner is
   shown here.

An IC of −0.032 is small even taken at face value. Taken with the three points
above, the defensible claim is that a mechanical microstructure effect is
visible in the data, not that a tradeable signal was found — which is also what
the strategy backtest below concludes.

### IC Decay Profile
- Strongest at 1-minute horizon (SPY IC = -0.032)
- Declines toward ~0 by 60 minutes. Monotone for SPY at the four saved horizons
  (−0.0319 → −0.0218 → −0.0185 → −0.0068); **not monotone for QQQ**, which goes
  −0.0135 at 5 min → −0.0140 at 15 min before falling to −0.0043 at 60 min. The
  earlier claim of monotone decay was true of one of the two symbols.
- ICIR: SPY −0.696, QQQ −0.532 (mean daily IC ÷ std of daily IC). Note this is
  an *information coefficient* ratio, not an information ratio in the
  return sense; it is printed by `src/signal_analysis.py:226` and is not written
  to any committed file.

### Intraday Pattern
- IC is most negative in the thin post-close extended-hours buckets
  (≈21:00–22:30 UTC): up to **−0.14**. Three caveats that matter more than the
  number: this is the largest of ~34 half-hour buckets scanned with no
  multiple-testing correction and no minimum sample beyond `n ≥ 50`
  (`src/signal_analysis.py:132-145`); the sample is mostly EST, where 22:00 UTC
  is **5:00 PM ET**, over an hour *after* the 16:00 close rather than "near
  market close" as an earlier version of this line said; and thin
  extended-hours books are exactly where the bid-ask bounce confound above is
  largest. Read it as consistent with the mechanical explanation, not against it.
- Weaker during core hours (14:30-18:00 UTC = 10:30 AM - 2:00 PM ET)
- These figures are printed to stdout and rendered into
  `results/figures/intraday_heatmap.png`; they are in no committed CSV.

### Quintile Analysis
Return spread from Q1 (low signal) to Q5 (high signal):
- **SPY 1-min**: Q1 = +0.11 bps, Q5 = -0.10 bps (spread: 0.21 bps) — monotone
- **SPY 5-min**: Q1 = +0.15 bps, Q5 = -0.19 bps (spread: 0.34 bps) — **not
  monotone**: Q3 = −0.06 sits below Q4 = −0.02. An earlier version headed both
  bullets with "monotonic"; it holds at 1 minute only.

### Strategy Performance (Honest Assessment)

**The naive threshold strategy is unprofitable after costs, and it is worse
than trading at random.**

| Symbol | Hold | Realized total return | Sharpe (annualized) | Random-entry Sharpe | Win Rate | Avg Return/Trade |
|--------|------|----------------------:|--------------------:|--------------------:|---------:|-----------------:|
| SPY | 1 min | −98.7% | −99.0 | **−58.9** | 24.6% | -1.10 bps |
| SPY | 5 min | −82.1% | −31.3 | **−12.9** | 36.3% | -1.20 bps |
| SPY | 60 min | −16.7% | −2.85 | **−1.07** | 46.2% | -1.18 bps |
| QQQ | 60 min | −11.7% | −1.37 | **−0.83** | 48.3% | -0.76 bps |

**The random-entry column is the one to read.** `src/backtest.py:115` runs a
Monte Carlo of randomly-timed trades at the same frequency and cost, and the
signal-driven strategy is worse than it at every hold period in
`results/backtest_results.csv` — eight rows out of eight. Whatever the negative
IC is picking up, routing trades on it does worse than routing them by coin
flip. That comparison was already in the committed results and was not stated
in this README; it should have been the headline of this section.

**On those Sharpe figures.** −99.0 is not a risk-adjusted return, it is a units
artifact. `annualized_return = avg_return × daily_trades × 252` scales a
per-trade mean by ~81,000 trades a year with no compounding and no capital
constraint, which is how the committed CSV ends up with `ann_return_pct` of
−884% and −895% — figures a long/short book cannot produce, since it cannot
lose more than everything. The realized column above is the honest one: −98.7%
over the ~122-day sample. `compute_metrics` now also reports
`annualized_return_compounded`, `per_trade_sharpe` and an
`annualization_implausible` flag so the artifact cannot be quoted alone.

**Why it fails as a standalone strategy:**
1. **Transaction costs dominate**: at 1bp round-trip, the ~0.2 bps quintile spread is eaten by friction
2. **The signal is mean-reverting, not momentum**: the naive strategy trades in the direction of the signal, but the relationship predicts reversal
3. **Extremely high frequency**: ~39K–40K trades over the six-month sample at 1-min hold — cost bleed is catastrophic
4. **The entry band is fitted in-sample**: `src/backtest.py:26` computes `mu` and `sigma` over the *entire* series and applies the ±1σ band to every bar, so January trades use a σ that includes March. A trailing-window σ is required for an out-of-sample read, and this backtest does not have one.

### Potential Value (Not as a standalone strategy)

Despite failing as a standalone alpha, OFI has clear value as:
1. **Execution timing signal**: Avoid buying when OFI is extremely positive (expect mean-reversion)
2. **Regime indicator**: High |OFI| = high short-term volatility/reversal risk
3. **Composite factor ingredient**: Combined with momentum/value signals at lower frequency
4. **Contrarian signal**: Flip the sign — short after high OFI, long after low OFI — but costs still dominate at minute frequency

## Methodology

### Data
- **Source**: Alpaca Markets API (free tier)
- **Symbols**: SPY, QQQ
- **Period**: Oct 2025 — Mar 2026 (122 trading days, ~188K minute bars)
- **Limitations**: IEX quotes/trades not accessible on free plan; OHLCV proxy used instead

### OFI Construction
Since Level 1 quote data was unavailable, we use the OHLCV proxy:

```
Volume_OFI = (close - open) / (high - low + ε) × volume
```

This approximates net buying/selling pressure within each bar. Variants computed:
- **Raw OFI**: As above
- **Normalized OFI**: Divided by bar volume (bounded [-1, 1])
- **Rolling OFI**: 5/10/20-minute rolling sum
- **Cumulative OFI**: Intraday cumulative (resets daily)

### Analysis Pipeline
1. **IC Analysis**: Spearman rank correlation between OFI[t] and return[t+h]
2. **IC Decay**: IC at horizons h = 1, 2, 3, 5, 10, 15, 30, 60 minutes
3. **Quintile Returns**: Sort by OFI, compute average forward return per quintile
4. **Intraday Pattern**: IC by 30-minute time-of-day buckets
5. **Strategy Backtest**: Threshold entry (±1σ), fixed hold period, 1bp round-trip cost
6. **Monte Carlo Baseline**: 500 simulations of random entry for comparison

## Project Structure

```
ofi_signal/
├── data/
│   ├── raw/               # Parquet files from Alpaca
│   └── processed/         # OFI signals
├── src/
│   ├── data_fetcher.py    # Alpaca API data collection
│   ├── ofi_calculator.py  # OFI signal construction
│   ├── signal_analysis.py # IC, quintile, intraday analysis
│   ├── backtest.py        # Strategy backtest + Monte Carlo
│   └── live_monitor.py    # 60s poll of Alpaca's minute-bar REST endpoint
├── results/
│   ├── figures/           # All charts
│   ├── summary_stats.csv  # IC/p-value for all signal×horizon combos
│   └── backtest_results.csv
└── README.md
```

## Reproduce

```bash
pip install -r requirements.txt
pip install pyarrow             # required by the parquet cache, missing from requirements.txt
# Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment or ../.env

python src/data_fetcher.py      # Fetch data (~3 min)
python src/ofi_calculator.py    # Compute signals
python src/signal_analysis.py   # IC analysis + charts
python src/backtest.py          # Strategy backtest
python src/live_monitor.py      # 60-second REST poll (Ctrl+C to stop)
```

**The committed results cannot be reproduced by running this.** Three reasons,
all worth stating rather than letting someone discover them:

- `data/raw/` and `data/processed/` are gitignored, so no input data ships.
- `src/data_fetcher.py:106` sets `end = datetime.now()` and `start = end - 180
  days`, with no date pinning. Running it today fetches a different six months
  than the **Oct 2025 – Mar 2026** window these results describe.
- `src/backtest.py:115` calls `np.random.choice` with no seed, so the
  `random_sharpe_mean` and `random_sharpe_p95` columns in
  `results/backtest_results.csv` are not reproducible even on identical data.

Pinning explicit dates, seeding the Monte Carlo, and committing the processed
signal panel would fix all three. Until then, treat `results/` as a record of
one run rather than as a regenerable artifact.

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

## Author

Built as part of quantitative research portfolio. See also:
- [kuant-core](https://github.com/zwmjj/kuant-core) — Research library: event-driven backtester, factor library, cost and risk toolkit
- [kuant-strategies](https://github.com/zwmjj/kuant-strategies) — Strategy implementations built on `kuant-core`
- [alt-data-research](https://github.com/zwmjj/alt-data-research) — Alternative data alpha research
