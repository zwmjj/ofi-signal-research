"""Compute Order Flow Imbalance (OFI) signal variants.

Standard OFI (Cont et al. 2014):
  OFI_t = Δbid_size * I(bid_price >= bid_price_{t-1})
        - Δask_size * I(ask_price <= ask_price_{t-1})

OHLCV Proxy (when quotes unavailable):
  Volume_OFI = (close - open) / (high - low + ε) * volume
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_quote_ofi(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Standard OFI from Level 1 quote changes (bid/ask price and size).

    Expects columns: symbol, timestamp, bid_price, bid_size, ask_price, ask_size.
    Returns per-symbol, per-timestamp OFI values.
    """
    results = []
    for sym, grp in quotes_df.groupby('symbol'):
        grp = grp.sort_values('timestamp').copy()

        # Resample to 1-minute snapshots (last quote per minute)
        grp = grp.set_index('timestamp')
        snap = grp.resample('1min').last().dropna(subset=['bid_price', 'ask_price'])

        bp = snap['bid_price']
        bs = snap['bid_size']
        ap = snap['ask_price']
        as_ = snap['ask_size']

        # OFI = Δbid_size * I(bid_up) - Δask_size * I(ask_down)
        delta_bs = bs.diff()
        delta_as = as_.diff()
        bid_up = (bp >= bp.shift(1)).astype(float)
        ask_dn = (ap <= ap.shift(1)).astype(float)

        ofi = delta_bs * bid_up - delta_as * ask_dn
        snap['ofi_raw'] = ofi
        snap['symbol'] = sym
        results.append(snap.reset_index())

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def compute_volume_ofi(bars_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV-based OFI proxy when tick data is unavailable.

    Volume_OFI = (close - open) / (high - low + ε) * volume
    Positive → net buying pressure, Negative → net selling pressure.
    """
    df = bars_df.copy()
    eps = 1e-8
    df['ofi_raw'] = (df['close'] - df['open']) / (df['high'] - df['low'] + eps) * df['volume']
    return df


def add_ofi_variants(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """Add normalized, rolling, and cumulative OFI variants.

    Expects 'ofi_raw' column already computed. Operates per-symbol.
    """
    results = []
    for sym, grp in df.groupby('symbol'):
        grp = grp.sort_values('timestamp').copy()

        # WARNING -- `ofi_norm` contains no volume information.
        #
        #   ofi_raw  = (close - open) / (high - low + eps) * volume
        #   ofi_norm = ofi_raw / volume
        #            = (close - open) / (high - low + eps)
        #
        # The volume term cancels exactly. `ofi_norm` is the candlestick
        # body-to-range ratio: a legitimate scale-free microstructure feature,
        # but NOT an order-flow imbalance, and not what the name says.
        #
        # This matters because `ofi_norm` is what signal_analysis and backtest
        # select as the best variant, so it is the signal behind every headline
        # number in the README. It is left as-is rather than silently corrected
        # because the committed results were produced by exactly this line, and
        # the input data is not committed, so they cannot be regenerated. See
        # `ofi_vol_scaled` below for the variant this was meant to be.
        vol = grp['volume'] if 'volume' in grp.columns else grp['ofi_raw'].abs()
        grp['ofi_norm'] = grp['ofi_raw'] / (vol.replace(0, np.nan))

        # The intended normalization: scale by TYPICAL volume rather than by
        # this bar's own volume, so the relative-volume information survives.
        # A bar with a small body on ten times normal volume should not score
        # the same as the identical body on normal volume, which is precisely
        # what `ofi_norm` does.
        #
        # NOT EVALUATED. This column is computed but does not appear in any
        # committed result; adding it to the signal list and re-running is the
        # obvious next step for this project.
        typical_vol = vol.rolling(390, min_periods=20).median()
        grp['ofi_vol_scaled'] = grp['ofi_raw'] / (typical_vol.replace(0, np.nan))

        # Rolling OFI
        for w in windows:
            grp[f'ofi_roll{w}'] = grp['ofi_raw'].rolling(w, min_periods=1).sum()

        # Cumulative OFI (reset each trading day)
        grp['date'] = grp['timestamp'].dt.date
        grp['ofi_cum'] = grp.groupby('date')['ofi_raw'].cumsum()
        grp.drop(columns=['date'], inplace=True)

        results.append(grp)

    return pd.concat(results, ignore_index=True)


def process_and_save(bars_path: str = None, quotes_path: str = None) -> pd.DataFrame:
    """Main entry: load raw data, compute OFI, save processed output."""
    raw_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'

    # Try quote-based OFI first
    qp = Path(quotes_path) if quotes_path else raw_dir / 'quotes.parquet'
    if qp.exists():
        print('Computing quote-based OFI (standard)...', flush=True)
        quotes = pd.read_parquet(qp)
        if len(quotes) > 100:
            ofi_df = compute_quote_ofi(quotes)
            if len(ofi_df) > 0:
                # Merge volume from bars if available
                bp = Path(bars_path) if bars_path else raw_dir / 'minute_bars.parquet'
                if bp.exists():
                    bars = pd.read_parquet(bp)
                    bars['timestamp'] = pd.to_datetime(bars['timestamp']).dt.tz_localize(None)
                    ofi_df['timestamp'] = pd.to_datetime(ofi_df['timestamp']).dt.tz_localize(None)
                    ofi_df = ofi_df.merge(
                        bars[['symbol', 'timestamp', 'volume', 'close', 'open', 'high', 'low', 'vwap']],
                        on=['symbol', 'timestamp'], how='left'
                    )
                ofi_df = add_ofi_variants(ofi_df)
                out_path = PROCESSED_DIR / 'ofi_signals.parquet'
                ofi_df.to_parquet(out_path, index=False)
                print(f'Saved quote-based OFI: {out_path} ({len(ofi_df):,} rows)')
                return ofi_df

    # Fallback: OHLCV proxy
    bp = Path(bars_path) if bars_path else raw_dir / 'minute_bars.parquet'
    if not bp.exists():
        raise FileNotFoundError(f'No bar data at {bp}')

    print('Computing OHLCV-based OFI proxy (Volume_OFI)...', flush=True)
    bars = pd.read_parquet(bp)
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    ofi_df = compute_volume_ofi(bars)
    ofi_df = add_ofi_variants(ofi_df)

    out_path = PROCESSED_DIR / 'ofi_signals.parquet'
    ofi_df.to_parquet(out_path, index=False)
    print(f'Saved OHLCV proxy OFI: {out_path} ({len(ofi_df):,} rows)')

    # Print summary
    for sym in ofi_df['symbol'].unique():
        s = ofi_df[ofi_df['symbol'] == sym]
        print(f'\n  {sym}:')
        for col in ['ofi_raw', 'ofi_norm', 'ofi_roll5', 'ofi_roll10', 'ofi_roll20', 'ofi_cum']:
            if col in s.columns:
                vals = s[col].dropna()
                print(f'    {col:12s}: mean={vals.mean():>12.2f}  std={vals.std():>12.2f}  '
                      f'min={vals.min():>12.2f}  max={vals.max():>12.2f}  NaN={s[col].isna().sum()}')

    return ofi_df


if __name__ == '__main__':
    process_and_save()
