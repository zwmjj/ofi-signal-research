# Order Flow Imbalance (OFI) Signal Research

Empirical analysis of Order Flow Imbalance as a short-term price predictor for US equities, using minute-level data from Alpaca Markets.

## Key Findings

### Signal has statistically significant but negative predictive power

The OFI signal shows a **mean reversion** relationship with future returns — the opposite of the momentum effect typically documented in academic literature:

| Symbol | Signal | Horizon | IC | p-value | Interpretation |
|--------|--------|---------|----:|--------:|----------------|
| SPY | ofi_norm | 1 min | **-0.032** | <1e-22 | Strong mean-reversion |
| SPY | ofi_norm | 5 min | **-0.022** | <1e-11 | Persistent reversal |
| QQQ | ofi_norm | 1 min | **-0.024** | <1e-13 | Strong mean-reversion |
| QQQ | ofi_norm | 5 min | **-0.014** | <1e-5 | Moderate reversal |

**Why negative IC?** The OHLCV-based Volume_OFI proxy captures *realized* buying/selling pressure within a bar. Negative IC means bars with net buying pressure are followed by negative returns — classic microstructure mean-reversion. This is consistent with:
- Bid-ask bounce effects at minute frequency
- Inventory-driven rebalancing by market makers
- Temporary price impact reverting after large volume bars

### IC Decay Profile
- Strongest at 1-minute horizon (SPY IC = -0.032)
- Decays monotonically to ~0 by 60 minutes
- Information Ratio: SPY IR = -0.696 (daily), QQQ IR = -0.532

### Intraday Pattern
- Signal strongest near **market close** (22:00-22:30 UTC = 6:00-6:30 PM ET extended hours): IC up to **-0.14**
- Weaker during core hours (14:30-18:00 UTC = 10:30 AM - 2:00 PM ET)
- Pre-market (08:00-09:00 UTC) shows mixed signal

### Quintile Analysis
Monotonic return spread from Q1 (low OFI) to Q5 (high OFI):
- **SPY 1-min**: Q1 = +0.11 bps, Q5 = -0.10 bps (spread: 0.21 bps)
- **SPY 5-min**: Q1 = +0.15 bps, Q5 = -0.19 bps (spread: 0.34 bps)

### Strategy Performance (Honest Assessment)

**The naive threshold strategy is unprofitable after costs:**

| Symbol | Hold | Sharpe | Win Rate | Avg Return/Trade |
|--------|------|-------:|---------:|-----------------:|
| SPY | 1 min | -99.0 | 24.6% | -1.10 bps |
| SPY | 5 min | -31.3 | 36.3% | -1.20 bps |
| SPY | 60 min | -2.85 | 46.2% | -1.18 bps |
| QQQ | 60 min | -1.37 | 48.3% | -0.76 bps |

**Why it fails as a standalone strategy:**
1. **Transaction costs dominate**: At 1bp round-trip cost, the ~0.2 bps quintile spread is eaten by friction
2. **Signal is mean-reverting, not momentum**: The naive strategy trades in the direction of OFI, but the signal predicts reversal
3. **Extremely high frequency**: 40K+ trades over 5 months at 1-min hold — cost bleeds are catastrophic

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
│   └── live_monitor.py    # Real-time signal monitor
├── results/
│   ├── figures/           # All charts
│   ├── summary_stats.csv  # IC/p-value for all signal×horizon combos
│   └── backtest_results.csv
└── README.md
```

## Reproduce

```bash
pip install -r requirements.txt
# Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment or ../.env

python src/data_fetcher.py      # Fetch data (~3 min)
python src/ofi_calculator.py    # Compute OFI signals
python src/signal_analysis.py   # IC analysis + charts
python src/backtest.py          # Strategy backtest
python src/live_monitor.py      # Real-time monitor (Ctrl+C to stop)
```

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

## Author

Built as part of quantitative research portfolio. See also:
- [kuant-core](https://github.com/zwmjj/kuant-core) — Research library: event-driven backtester, factor library, cost and risk toolkit
- [kuant-strategies](https://github.com/zwmjj/kuant-strategies) — Strategy implementations built on `kuant-core`
- [alt-data-research](https://github.com/zwmjj/alt-data-research) — Alternative data alpha research
