"""Fetch historical bars and quotes from Alpaca API for OFI signal research."""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest, StockQuotesRequest, StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame

# Load credentials
_env_path = Path(__file__).resolve().parent.parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)

API_KEY = os.environ.get('ALPACA_API_KEY', os.environ.get('APCA_API_KEY_ID', ''))
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', os.environ.get('APCA_API_SECRET_KEY', ''))

if not API_KEY or not SECRET_KEY:
    sys.exit('ERROR: Alpaca credentials not found in env or .env file')

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_minute_bars(symbols: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch minute-level OHLCV bars. Batches by month to avoid timeouts."""
    all_frames = []
    cursor = start
    while cursor < end:
        month_end = min(cursor + timedelta(days=30), end)
        print(f'  Fetching bars {cursor.date()} → {month_end.date()} for {symbols}...', flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=cursor,
                end=month_end,
            )
            bars = client.get_stock_bars(req)
            df = bars.df.reset_index()
            all_frames.append(df)
            print(f'    got {len(df)} rows', flush=True)
        except Exception as e:
            print(f'    ERROR: {e}', flush=True)
        cursor = month_end
    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=['symbol', 'timestamp']).sort_values(['symbol', 'timestamp'])
    return result


def fetch_historical_quotes(symbols: list[str], start: datetime, end: datetime,
                            max_days: int = 5) -> pd.DataFrame:
    """Fetch historical Level 1 quotes (IEX). Limited to a few days due to volume."""
    quote_start = end - timedelta(days=max_days)
    if quote_start < start:
        quote_start = start
    print(f'  Fetching quotes {quote_start.date()} → {end.date()} for {symbols}...', flush=True)
    try:
        req = StockQuotesRequest(
            symbol_or_symbols=symbols,
            start=quote_start,
            end=end,
        )
        quotes = client.get_stock_quotes(req)
        df = quotes.df.reset_index()
        print(f'    got {len(df)} rows', flush=True)
        return df
    except Exception as e:
        print(f'    Quote fetch failed: {e}', flush=True)
        return pd.DataFrame()


def fetch_historical_trades(symbols: list[str], start: datetime, end: datetime,
                            max_days: int = 5) -> pd.DataFrame:
    """Fetch historical trades (IEX). Limited to a few days."""
    trade_start = end - timedelta(days=max_days)
    if trade_start < start:
        trade_start = start
    print(f'  Fetching trades {trade_start.date()} → {end.date()} for {symbols}...', flush=True)
    try:
        req = StockTradesRequest(
            symbol_or_symbols=symbols,
            start=trade_start,
            end=end,
        )
        trades = client.get_stock_trades(req)
        df = trades.df.reset_index()
        print(f'    got {len(df)} rows', flush=True)
        return df
    except Exception as e:
        print(f'    Trade fetch failed: {e}', flush=True)
        return pd.DataFrame()


def main():
    symbols = ['SPY', 'QQQ']
    end = datetime.now()
    start = end - timedelta(days=180)

    print('=' * 60)
    print('OFI Signal Research — Data Fetcher')
    print('=' * 60)

    # 1. Minute bars (6 months)
    print(f'\n[1] Minute bars: {start.date()} → {end.date()}')
    bars = fetch_minute_bars(symbols, start, end)
    if len(bars) > 0:
        bars_path = RAW_DIR / 'minute_bars.parquet'
        bars.to_parquet(bars_path, index=False)
        print(f'\nBars saved: {bars_path}')
        print(f'  Total rows: {len(bars):,}')
        print(f'  Date range: {bars["timestamp"].min()} → {bars["timestamp"].max()}')
        for sym in symbols:
            sym_bars = bars[bars['symbol'] == sym]
            print(f'  {sym}: {len(sym_bars):,} bars, '
                  f'{sym_bars["timestamp"].dt.date.nunique()} trading days')
        null_pct = bars.isnull().sum() / len(bars) * 100
        nulls = null_pct[null_pct > 0]
        if len(nulls) > 0:
            print(f'  Missing values: {dict(nulls.round(2))}')
        else:
            print(f'  Missing values: none')
    else:
        print('  ERROR: no bar data fetched')

    # 2. Historical quotes (last 5 days)
    print(f'\n[2] Historical quotes (last 5 trading days)')
    quotes = fetch_historical_quotes(symbols, start, end, max_days=5)
    has_quotes = len(quotes) > 0
    if has_quotes:
        quotes_path = RAW_DIR / 'quotes.parquet'
        quotes.to_parquet(quotes_path, index=False)
        print(f'  Quotes saved: {quotes_path}')
        print(f'  Total rows: {len(quotes):,}')
        print(f'  → Level 1 quotes AVAILABLE (IEX)')
    else:
        print(f'  → Level 1 quotes NOT available, will use OHLCV proxy')

    # 3. Historical trades (last 5 days)
    print(f'\n[3] Historical trades (last 5 trading days)')
    trades = fetch_historical_trades(symbols, start, end, max_days=5)
    if len(trades) > 0:
        trades_path = RAW_DIR / 'trades.parquet'
        trades.to_parquet(trades_path, index=False)
        print(f'  Trades saved: {trades_path}')
        print(f'  Total rows: {len(trades):,}')
    else:
        print(f'  → Trade data not available')

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Bars: {len(bars):,} rows')
    print(f'  Quotes: {len(quotes):,} rows ({"available" if has_quotes else "NOT available"})')
    print(f'  Trades: {len(trades):,} rows')
    print(f'  OFI approach: {"Quote-based (standard)" if has_quotes else "OHLCV proxy (Volume_OFI)"}')


if __name__ == '__main__':
    main()
