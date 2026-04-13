"""Simple threshold-based OFI strategy backtest with Monte Carlo baseline."""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIG_DIR = RESULTS_DIR / 'figures'


def run_backtest(df: pd.DataFrame, signal_col: str, price_col: str = 'close',
                 threshold_std: float = 1.0, hold_minutes: int = 5,
                 cost_bp: float = 1.0) -> dict:
    """Threshold-based OFI strategy.

    Entry: signal > +threshold_std*σ → long, signal < -threshold_std*σ → short
    Exit: hold for hold_minutes then flatten
    Cost: cost_bp bps per round-trip
    """
    df = df.sort_values('timestamp').copy()
    sig = df[signal_col]
    prices = df[price_col]

    mu = sig.mean()
    sigma = sig.std()
    upper = mu + threshold_std * sigma
    lower = mu - threshold_std * sigma

    trades = []
    i = 0
    while i < len(df) - hold_minutes:
        s = sig.iloc[i]
        if s > upper:
            direction = 1  # long
        elif s < lower:
            direction = -1  # short
        else:
            i += 1
            continue

        entry_price = prices.iloc[i]
        exit_price = prices.iloc[i + hold_minutes]
        raw_ret = direction * (exit_price / entry_price - 1)
        net_ret = raw_ret - cost_bp / 10000

        trades.append({
            'entry_time': df['timestamp'].iloc[i],
            'exit_time': df['timestamp'].iloc[i + hold_minutes],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'raw_return': raw_ret,
            'net_return': net_ret,
        })
        i += hold_minutes  # no overlapping trades

    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'total_return': 0}

    trades_df = pd.DataFrame(trades)
    returns = trades_df['net_return']

    # Compute metrics
    n_trades = len(trades_df)
    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean()
    daily_trades = max(1, n_trades / df['timestamp'].dt.date.nunique())
    annualized_return = avg_return * daily_trades * 252
    annualized_vol = returns.std() * np.sqrt(daily_trades * 252)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Drawdown
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret / running_max - 1)
    max_dd = drawdown.min()

    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1e-8
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    long_trades = (trades_df['direction'] == 1).sum()
    short_trades = (trades_df['direction'] == -1).sum()

    return {
        'n_trades': n_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'total_return': round(total_return * 100, 4),
        'annualized_return': round(annualized_return * 100, 2),
        'annualized_vol': round(annualized_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(max_dd * 100, 3),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'avg_return_bps': round(avg_return * 10000, 3),
        'trades_df': trades_df,
        'cum_returns': cum_ret,
    }


def run_random_baseline(df: pd.DataFrame, hold_minutes: int = 5,
                        cost_bp: float = 1.0, n_simulations: int = 500) -> dict:
    """Monte Carlo random entry baseline."""
    prices = df.sort_values('timestamp')['close'].values
    n = len(prices)
    sharpes = []

    for _ in range(n_simulations):
        # Random entries (same frequency as roughly 1-std threshold)
        n_entries = max(10, n // (hold_minutes * 5))
        entry_idxs = np.random.choice(range(n - hold_minutes), size=n_entries, replace=False)
        directions = np.random.choice([-1, 1], size=n_entries)

        returns = []
        for idx, d in zip(sorted(entry_idxs), directions):
            raw_ret = d * (prices[idx + hold_minutes] / prices[idx] - 1)
            net_ret = raw_ret - cost_bp / 10000
            returns.append(net_ret)

        returns = np.array(returns)
        if len(returns) > 1 and returns.std() > 0:
            daily_trades = max(1, n_entries / df['timestamp'].dt.date.nunique())
            ann_ret = returns.mean() * daily_trades * 252
            ann_vol = returns.std() * np.sqrt(daily_trades * 252)
            sharpes.append(ann_ret / ann_vol)

    return {
        'mean_sharpe': round(np.mean(sharpes), 3),
        'std_sharpe': round(np.std(sharpes), 3),
        'p5_sharpe': round(np.percentile(sharpes, 5), 3),
        'p95_sharpe': round(np.percentile(sharpes, 95), 3),
        'sharpes': sharpes,
    }


def run_buy_hold(df: pd.DataFrame) -> dict:
    """Buy-and-hold benchmark."""
    prices = df.sort_values('timestamp')['close']
    total_ret = prices.iloc[-1] / prices.iloc[0] - 1
    n_days = df['timestamp'].dt.date.nunique()
    ann_ret = total_ret * (252 / n_days) if n_days > 0 else 0
    daily_returns = prices.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252 * 390)  # minute returns
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + daily_returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    return {
        'total_return': round(total_ret * 100, 2),
        'annualized_return': round(ann_ret * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(max_dd * 100, 2),
    }


def plot_equity_curves(strategy_result: dict, bh_result: dict, random_result: dict,
                       symbol: str, hold_min: int):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Strategy
    cum = strategy_result['cum_returns']
    ax.plot(range(len(cum)), cum.values, label=f'OFI Strategy (SR={strategy_result["sharpe"]:.2f})',
            color='#1a3d6a', linewidth=2)

    # Random baseline band
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label=f'Random baseline (SR={random_result["mean_sharpe"]:.2f})')

    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(f'OFI Strategy Equity Curve — {symbol}, {hold_min}-min hold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / f'equity_curve_{symbol}_{hold_min}m.png')
    plt.close()
    print(f'  Saved equity_curve_{symbol}_{hold_min}m.png')


def plot_drawdown(strategy_result: dict, symbol: str, hold_min: int):
    cum = strategy_result['cum_returns']
    dd = cum / cum.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(dd)), dd.values, 0, color='#d32f2f', alpha=0.5)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Drawdown')
    ax.set_title(f'OFI Strategy Drawdown — {symbol}')
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / f'drawdown_{symbol}_{hold_min}m.png')
    plt.close()
    print(f'  Saved drawdown_{symbol}_{hold_min}m.png')


def main():
    processed_dir = Path(__file__).resolve().parent.parent / 'data' / 'processed'
    df = pd.read_parquet(processed_dir / 'ofi_signals.parquet')
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    print('=' * 60)
    print('OFI Signal Research — Strategy Backtest')
    print('=' * 60)

    # Find best signal from summary_stats
    try:
        stats_df = pd.read_csv(RESULTS_DIR / 'summary_stats.csv')
        best_row = stats_df.loc[stats_df['abs_ic'].idxmax()]
        best_sig = best_row['signal']
    except Exception:
        best_sig = 'ofi_norm'

    print(f'\nUsing signal: {best_sig}')

    hold_periods = [1, 5, 15, 60]
    all_results = []

    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym].copy()

        print(f'\n{"=" * 40}')
        print(f'{sym}')
        print(f'{"=" * 40}')

        # Buy-and-hold baseline
        bh = run_buy_hold(sym_df)
        print(f'\n  Buy & Hold: return={bh["total_return"]:.2f}%, '
              f'Sharpe={bh["sharpe"]:.3f}, MaxDD={bh["max_drawdown"]:.2f}%')

        # Random baseline
        print(f'\n  Running Monte Carlo random baseline...')

        for hold in hold_periods:
            print(f'\n  --- Hold {hold} min ---')

            # OFI strategy
            result = run_backtest(sym_df, best_sig, hold_minutes=hold)
            if result['n_trades'] == 0:
                print(f'    No trades generated')
                continue

            print(f'    Trades: {result["n_trades"]} (L:{result["long_trades"]} S:{result["short_trades"]})')
            print(f'    Total return: {result["total_return"]:.4f}%')
            print(f'    Ann. return: {result["annualized_return"]:.2f}%')
            print(f'    Sharpe: {result["sharpe"]:.3f}')
            print(f'    Max DD: {result["max_drawdown"]:.3f}%')
            print(f'    Win rate: {result["win_rate"]:.1f}%')
            print(f'    Profit factor: {result["profit_factor"]:.3f}')
            print(f'    Avg return/trade: {result["avg_return_bps"]:.3f} bps')

            # Random baseline for this hold period
            rand = run_random_baseline(sym_df, hold_minutes=hold)
            print(f'    Random SR: {rand["mean_sharpe"]:.3f} ± {rand["std_sharpe"]:.3f} '
                  f'[{rand["p5_sharpe"]:.3f}, {rand["p95_sharpe"]:.3f}]')

            # Plots for 5-min hold
            if hold == 5 and result['n_trades'] > 10:
                plot_equity_curves(result, bh, rand, sym, hold)
                plot_drawdown(result, sym, hold)

            all_results.append({
                'symbol': sym, 'hold_min': hold, 'signal': best_sig,
                'n_trades': result['n_trades'],
                'total_return_pct': result['total_return'],
                'ann_return_pct': result['annualized_return'],
                'sharpe': result['sharpe'],
                'max_dd_pct': result['max_drawdown'],
                'win_rate_pct': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'avg_bps': result['avg_return_bps'],
                'random_sharpe_mean': rand['mean_sharpe'],
                'random_sharpe_p95': rand['p95_sharpe'],
                'bh_sharpe': bh['sharpe'],
            })

    # Save backtest results
    bt_df = pd.DataFrame(all_results)
    bt_path = RESULTS_DIR / 'backtest_results.csv'
    bt_df.to_csv(bt_path, index=False)
    print(f'\nSaved backtest_results.csv')
    print(f'\n{bt_df.to_string(index=False)}')


if __name__ == '__main__':
    main()
