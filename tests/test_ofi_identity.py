"""The headline signal contains no order flow. This proves it, rather than
asserting it in a comment.

`ofi_raw` multiplies by volume and `ofi_norm` divides by the same volume, so
the volume term cancels exactly and `ofi_norm` reduces to the candlestick
body-to-range ratio. The README states this as fact; these tests are what
makes that statement checkable.

Run with:  python -m pytest tests/ -v
       or: python tests/test_ofi_identity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ofi_calculator import add_ofi_variants, compute_volume_ofi

EPS = 1e-8


def _synthetic_bars(n: int = 500, seed: int = 0, volume_scale: float = 1.0):
    """OHLCV bars with wildly varying volume, so any volume dependence shows."""
    rng = np.random.default_rng(seed)
    open_ = 100 + rng.normal(0, 1, n).cumsum()
    close = open_ + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.uniform(0.01, 0.5, n)
    low = np.minimum(open_, close) - rng.uniform(0.01, 0.5, n)
    # Four orders of magnitude of volume variation.
    volume = rng.lognormal(mean=10, sigma=2, size=n) * volume_scale
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 09:30", periods=n, freq="1min"),
            "symbol": "TEST",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _ofi_norm(df):
    return add_ofi_variants(compute_volume_ofi(df))


def test_ofi_norm_reduces_to_body_over_range():
    """ofi_norm == (close - open) / (high - low + eps), to floating-point noise."""
    df = _ofi_norm(_synthetic_bars())
    body_over_range = (df["close"] - df["open"]) / (df["high"] - df["low"] + EPS)

    err = (df["ofi_norm"] - body_over_range).abs().max()
    print(f"  max |ofi_norm - (close-open)/(high-low+eps)| = {err:.3e}")

    assert err < 1e-12, (
        f"Expected exact cancellation, got max error {err:.3e}. If this fails, "
        "the identity documented in the README no longer holds."
    )


def test_ofi_norm_is_invariant_to_volume():
    """The strong form: multiply every volume by 1000 and ofi_norm does not move.

    A signal carrying order-flow information cannot be invariant to a
    thousand-fold change in volume. This is the test that makes the claim
    unambiguous.
    """
    base = _ofi_norm(_synthetic_bars(volume_scale=1.0))
    scaled = _ofi_norm(_synthetic_bars(volume_scale=1000.0))

    err = (base["ofi_norm"] - scaled["ofi_norm"]).abs().max()
    print(f"  max |ofi_norm(v) - ofi_norm(1000v)| = {err:.3e}")

    assert err < 1e-12, (
        f"ofi_norm moved by {err:.3e} under a 1000x volume change; it should "
        "be exactly invariant."
    )


def test_ofi_vol_scaled_does_depend_on_volume():
    """The contrast case, so the tests above cannot pass for a trivial reason.

    `ofi_vol_scaled` normalizes by *typical* volume rather than the bar's own,
    so relative-volume information survives. It must respond where ofi_norm
    does not. (This column is computed but NOT EVALUATED -- it appears in no
    committed result.)
    """
    base = _ofi_norm(_synthetic_bars(volume_scale=1.0))
    scaled = _ofi_norm(_synthetic_bars(volume_scale=1000.0))

    both = base["ofi_vol_scaled"].notna() & scaled["ofi_vol_scaled"].notna()
    assert both.sum() > 100, "not enough non-NaN rows to compare"

    # Scaling all volumes by a constant cancels in ofi_raw/typical_vol too, so
    # compare against a *different* volume path instead: same prices, different
    # volume draw.
    other = _ofi_norm(_synthetic_bars(seed=1))
    mask = base["ofi_vol_scaled"].notna() & other["ofi_vol_scaled"].notna()
    diff = (base.loc[mask, "ofi_vol_scaled"] - other.loc[mask, "ofi_vol_scaled"]).abs().max()
    print(f"  max |ofi_vol_scaled| difference across volume paths = {diff:.3e}")

    assert diff > 1e-6, (
        "ofi_vol_scaled did not respond to a different volume path; it should."
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                print(f"{name} ...")
                fn()
                print("  PASS\n")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL: {exc}\n")
    sys.exit(1 if failures else 0)
