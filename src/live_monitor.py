"""OFI signal monitor -- polls Alpaca's minute-bar REST endpoint once a minute.

This is a poll of a historical-bars endpoint on a 60s loop, not a streaming
connection: there is no WebSocket here. On Alpaca's free tier the most recent
15 minutes of SIP data are not returnable, so the loop will error every tick
until a paid feed is configured.
"""
import os
import sys
import time
import signal
import csv
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

_env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)

API_KEY = os.environ.get('ALPACA_API_KEY', os.environ.get('APCA_API_KEY_ID', ''))
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', os.environ.get('APCA_API_SECRET_KEY', ''))

if not API_KEY or not SECRET_KEY:
    sys.exit('ERROR: Alpaca credentials not found')

data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
LIVE_CSV = DATA_DIR / 'live_signals.csv'

SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT']
LOOKBACK_BARS = 60  # minutes of history for rolling OFI
POLL_INTERVAL = 60  # seconds between updates

_running = True


def _shutdown(signum, frame):
    global _running
    print('\nShutting down...', flush=True)
    _running = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def fetch_recent_bars(symbols: list[str], n_bars: int = 60) -> pd.DataFrame:
    """Fetch the most recent N minute bars."""
    from datetime import timedelta
    end = datetime.utcnow()
    start = end - timedelta(minutes=n_bars * 2)  # extra buffer for gaps
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
    )
    bars = data_client.get_stock_bars(req)
    df = bars.df.reset_index()
    return df


def compute_live_ofi(bars: pd.DataFrame, symbol: str) -> dict:
    """Compute OFI variants from recent minute bars for one symbol."""
    sym_bars = bars[bars['symbol'] == symbol].sort_values('timestamp').tail(LOOKBACK_BARS).copy()
    if len(sym_bars) < 5:
        return None

    eps = 1e-8
    sym_bars['ofi_raw'] = (sym_bars['close'] - sym_bars['open']) / \
                           (sym_bars['high'] - sym_bars['low'] + eps) * sym_bars['volume']

    vol = sym_bars['volume'].replace(0, np.nan)
    sym_bars['ofi_norm'] = sym_bars['ofi_raw'] / vol

    sym_bars['ofi_roll10'] = sym_bars['ofi_raw'].rolling(10, min_periods=1).sum()

    latest = sym_bars.iloc[-1]
    ofi_norm = latest['ofi_norm'] if pd.notna(latest['ofi_norm']) else 0
    ofi_roll10 = sym_bars['ofi_roll10'].iloc[-1]

    # Signal interpretation
    ofi_std = sym_bars['ofi_norm'].std()
    if ofi_std > 0:
        z_score = ofi_norm / ofi_std
    else:
        z_score = 0

    if z_score > 1:
        direction = 'BUY'
        strength = min(z_score / 3, 1.0)
    elif z_score < -1:
        direction = 'SELL'
        strength = min(abs(z_score) / 3, 1.0)
    else:
        direction = 'NEUTRAL'
        strength = abs(z_score) / 3

    return {
        'timestamp': latest['timestamp'],
        'symbol': symbol,
        'close': latest['close'],
        'volume': latest['volume'],
        'ofi_raw': round(latest['ofi_raw'], 2),
        'ofi_norm': round(ofi_norm, 4),
        'ofi_roll10': round(ofi_roll10, 2),
        'z_score': round(z_score, 3),
        'direction': direction,
        'strength': round(strength, 3),
    }


def log_signal(signal_data: dict):
    """Append signal to CSV log."""
    file_exists = LIVE_CSV.exists()
    with open(LIVE_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=signal_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(signal_data)


def main():
    global _running
    print('=' * 60)
    print('OFI Live Signal Monitor')
    print(f'Symbols: {SYMBOLS}')
    print(f'Poll interval: {POLL_INTERVAL}s')
    print(f'Log file: {LIVE_CSV}')
    print('=' * 60)

    # Check market status
    try:
        clock = trading_client.get_clock()
        print(f'Market open: {clock.is_open}')
        if not clock.is_open:
            print(f'Next open: {clock.next_open}')
            print('Note: running anyway for testing, but signals may be stale.')
    except Exception as e:
        print(f'Clock check failed: {e}')

    tick = 0
    while _running:
        tick += 1
        now = datetime.utcnow().strftime('%H:%M:%S')
        print(f'\n[{now}] Tick #{tick}', flush=True)

        try:
            bars = fetch_recent_bars(SYMBOLS, LOOKBACK_BARS)
            print(f'  Fetched {len(bars)} bars')

            for sym in SYMBOLS:
                sig = compute_live_ofi(bars, sym)
                if sig is None:
                    print(f'  {sym}: insufficient data')
                    continue

                log_signal(sig)
                arrow = '↑' if sig['direction'] == 'BUY' else '↓' if sig['direction'] == 'SELL' else '→'
                print(f'  {sym}: ${sig["close"]:.2f}  OFI={sig["ofi_norm"]:+.4f}  '
                      f'z={sig["z_score"]:+.3f}  {arrow} {sig["direction"]} '
                      f'(strength={sig["strength"]:.3f})')

        except Exception as e:
            print(f'  ERROR: {e}', flush=True)

        # Wait for next tick
        for _ in range(POLL_INTERVAL):
            if not _running:
                break
            time.sleep(1)

    print(f'\nStopped. Signals logged to {LIVE_CSV}')


if __name__ == '__main__':
    main()
