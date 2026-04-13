"""Predictive power analysis for OFI signals — IC decay, quintile returns, intraday patterns."""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIG_DIR = RESULTS_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10


def compute_forward_returns(df: pd.DataFrame, horizons: list[int] = [1, 5, 15, 60]) -> pd.DataFrame:
    """Compute forward returns at multiple horizons per symbol."""
    results = []
    for sym, grp in df.groupby('symbol'):
        grp = grp.sort_values('timestamp').copy()
        for h in horizons:
            grp[f'fwd_ret_{h}'] = grp['close'].pct_change(h).shift(-h)
        results.append(grp)
    return pd.concat(results, ignore_index=True)


def compute_ic_series(signal: pd.Series, fwd_returns: pd.Series, window: int = 390) -> pd.Series:
    """Rolling Spearman IC between signal and forward returns.
    Default window = 390 (one trading day of minute bars).
    """
    ic_values = []
    ic_dates = []
    for i in range(window, len(signal)):
        s = signal.iloc[i - window:i]
        r = fwd_returns.iloc[i - window:i]
        valid = s.notna() & r.notna()
        if valid.sum() < 30:
            continue
        ic, _ = stats.spearmanr(s[valid], r[valid])
        ic_values.append(ic)
        ic_dates.append(signal.index[i] if hasattr(signal.index, '__getitem__') else i)
    return pd.Series(ic_values, index=ic_dates, name='IC')


def compute_ic_by_day(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.Series:
    """Compute daily Spearman IC (cross-sectional within each day's minute bars)."""
    df = df.copy()
    df['date'] = df['timestamp'].dt.date
    ic_list = []
    for date, grp in df.groupby('date'):
        s = grp[signal_col].dropna()
        r = grp[return_col].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 30:
            continue
        ic, _ = stats.spearmanr(s[common], r[common])
        ic_list.append({'date': date, 'ic': ic})
    if not ic_list:
        return pd.Series(dtype=float)
    result = pd.DataFrame(ic_list).set_index('date')['ic']
    return result


def compute_ic_decay(df: pd.DataFrame, signal_col: str,
                     horizons: list[int] = [1, 2, 3, 5, 10, 15, 30, 60]) -> pd.DataFrame:
    """IC at each forward horizon — measures signal decay."""
    rows = []
    for sym, grp in df.groupby('symbol'):
        grp = grp.sort_values('timestamp').copy()
        for h in horizons:
            fwd = grp['close'].pct_change(h).shift(-h)
            valid = grp[signal_col].notna() & fwd.notna()
            if valid.sum() < 100:
                continue
            ic, pval = stats.spearmanr(grp.loc[valid, signal_col], fwd[valid])
            rows.append({
                'symbol': sym, 'horizon': h, 'ic': ic, 'pval': pval,
                'abs_ic': abs(ic), 'n_obs': int(valid.sum()),
            })
    return pd.DataFrame(rows)


def quintile_analysis(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.DataFrame:
    """Sort by signal quintile, compute average forward return per group."""
    df = df.dropna(subset=[signal_col, return_col]).copy()
    df['quintile'] = pd.qcut(df[signal_col], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                             duplicates='drop')
    result = df.groupby('quintile')[return_col].agg(['mean', 'std', 'count']).reset_index()
    result.columns = ['quintile', 'avg_return', 'std_return', 'count']
    result['avg_return_bps'] = result['avg_return'] * 10000
    return result


def intraday_pattern(df: pd.DataFrame, signal_col: str, return_col: str) -> pd.DataFrame:
    """IC by time of day (30-min buckets)."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['time_bucket'] = df['hour'].astype(str).str.zfill(2) + ':' + \
                        (df['minute'] // 30 * 30).astype(str).str.zfill(2)
    rows = []
    for bucket, grp in df.groupby('time_bucket'):
        valid = grp[signal_col].notna() & grp[return_col].notna()
        if valid.sum() < 50:
            continue
        ic, _ = stats.spearmanr(grp.loc[valid, signal_col], grp.loc[valid, return_col])
        rows.append({'time': bucket, 'ic': ic, 'abs_ic': abs(ic), 'n_obs': int(valid.sum())})
    return pd.DataFrame(rows)


# ── Plotting ──

def plot_ic_decay(decay_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sym in decay_df['symbol'].unique():
        d = decay_df[decay_df['symbol'] == sym]
        ax.plot(d['horizon'], d['ic'], 'o-', label=sym, linewidth=2, markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Forward Horizon (minutes)')
    ax.set_ylabel('Spearman IC')
    ax.set_title('OFI Signal: IC Decay Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / 'ic_decay_curve.png')
    plt.close()
    print(f'  Saved ic_decay_curve.png')


def plot_quintile_returns(quint_df: pd.DataFrame, symbol: str, horizon: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d32f2f', '#ff7043', '#bdbdbd', '#66bb6a', '#2e7d32']
    bars = ax.bar(quint_df['quintile'], quint_df['avg_return_bps'], color=colors, edgecolor='white')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Average Forward Return (bps)')
    ax.set_title(f'OFI Quintile Returns — {symbol}, {horizon}-min horizon')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, quint_df['avg_return_bps']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(FIG_DIR / f'quintile_returns_{symbol}_{horizon}m.png')
    plt.close()
    print(f'  Saved quintile_returns_{symbol}_{horizon}m.png')


def plot_intraday_heatmap(intraday_df: pd.DataFrame):
    if len(intraday_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = intraday_df.set_index('time')[['ic']].T
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'IC'})
    ax.set_title('OFI Signal: IC by Time of Day')
    ax.set_ylabel('')
    fig.savefig(FIG_DIR / 'intraday_heatmap.png')
    plt.close()
    print(f'  Saved intraday_heatmap.png')


def plot_ic_timeseries(ic_series: pd.Series, symbol: str, signal_name: str):
    if len(ic_series) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ic_series)), ic_series.values, color=['#2e7d32' if v > 0 else '#d32f2f' for v in ic_series.values],
           alpha=0.7, width=1.0)
    ax.axhline(ic_series.mean(), color='blue', linestyle='--', label=f'Mean IC={ic_series.mean():.4f}')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Daily IC')
    ax.set_title(f'OFI Daily IC Time Series — {symbol} ({signal_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / f'ic_timeseries_{symbol}.png')
    plt.close()
    print(f'  Saved ic_timeseries_{symbol}.png')


def main():
    processed_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    ofi_path = processed_dir / 'ofi_signals.parquet'
    df = pd.read_parquet(ofi_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    print('=' * 60)
    print('OFI Signal Research — Predictive Power Analysis')
    print('=' * 60)

    # Compute forward returns
    print('\n[1] Computing forward returns...')
    horizons = [1, 5, 15, 60]
    df = compute_forward_returns(df, horizons)
    print(f'  Forward returns computed for horizons: {horizons}')

    # IC Decay
    print('\n[2] IC decay analysis...')
    signal_cols = ['ofi_norm', 'ofi_roll5', 'ofi_roll10', 'ofi_roll20']
    all_decay = []
    for sig in signal_cols:
        decay = compute_ic_decay(df, sig, horizons=[1, 2, 3, 5, 10, 15, 30, 60])
        decay['signal'] = sig
        all_decay.append(decay)
    decay_df = pd.concat(all_decay, ignore_index=True)
    print(decay_df.to_string(index=False))

    # Plot IC decay for best signal
    best_sig = decay_df.groupby('signal')['abs_ic'].mean().idxmax()
    print(f'\n  Best signal by avg |IC|: {best_sig}')
    plot_ic_decay(decay_df[decay_df['signal'] == best_sig])

    # Daily IC time series
    print('\n[3] Daily IC time series...')
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym].copy()
        daily_ic = compute_ic_by_day(sym_df, best_sig, 'fwd_ret_1')
        if len(daily_ic) > 0:
            print(f'  {sym}: mean IC={daily_ic.mean():.4f}, '
                  f'std={daily_ic.std():.4f}, IR={daily_ic.mean() / daily_ic.std():.3f}, '
                  f'%positive={( daily_ic > 0).mean():.1%}')
            plot_ic_timeseries(daily_ic, sym, best_sig)

    # Quintile returns
    print('\n[4] Quintile analysis...')
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        for h in [1, 5]:
            ret_col = f'fwd_ret_{h}'
            quint = quintile_analysis(sym_df, best_sig, ret_col)
            print(f'\n  {sym} — {h}-min forward:')
            print(quint.to_string(index=False))
            plot_quintile_returns(quint, sym, h)

    # Intraday pattern
    print('\n[5] Intraday pattern...')
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        intra = intraday_pattern(sym_df, best_sig, 'fwd_ret_1')
        if len(intra) > 0:
            print(f'\n  {sym}:')
            print(intra.to_string(index=False))
    # Combined intraday heatmap
    all_intra = intraday_pattern(df, best_sig, 'fwd_ret_1')
    plot_intraday_heatmap(all_intra)

    # Summary stats CSV
    print('\n[6] Writing summary_stats.csv...')
    summary_rows = []
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        for sig in signal_cols:
            for h in horizons:
                ret_col = f'fwd_ret_{h}'
                valid = sym_df[sig].notna() & sym_df[ret_col].notna()
                if valid.sum() < 100:
                    continue
                ic, pval = stats.spearmanr(sym_df.loc[valid, sig], sym_df.loc[valid, ret_col])
                summary_rows.append({
                    'symbol': sym, 'signal': sig, 'horizon_min': h,
                    'ic': round(ic, 6), 'pval': round(pval, 6),
                    'abs_ic': round(abs(ic), 6), 'n_obs': int(valid.sum()),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / 'summary_stats.csv', index=False)
    print(f'  Saved summary_stats.csv ({len(summary)} rows)')
    print(f'\n  Best overall: {summary.loc[summary["abs_ic"].idxmax()].to_dict()}')


if __name__ == '__main__':
    main()
