"""
Mock OHLCV data generators for testing trendline breakout detection.

Three scenarios are provided:
  1. **descending_triangle** – flat support + descending resistance → breakout down
  2. **ascending_breakout** – rising support + flat resistance → breakout up
  3. **channel_with_breakout** – parallel descending channel → breakout up at the end
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    close: np.ndarray,
    noise_pct: float = 0.008,
    volume_base: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a realistic OHLCV DataFrame from a synthetic close series.
    Adds intra-bar noise for High/Low and random volume.
    """
    rng = np.random.default_rng(seed)
    n = len(close)
    noise = rng.normal(0, noise_pct, n) * close
    high = close + np.abs(noise) + rng.uniform(0.001, 0.005, n) * close
    low = close - np.abs(rng.normal(0, noise_pct, n) * close) - rng.uniform(0.001, 0.005, n) * close
    open_ = close + rng.normal(0, noise_pct * 0.5, n) * close
    volume = (volume_base * (1 + rng.normal(0, 0.3, n))).clip(100_000).astype(int)

    return pd.DataFrame({
        "Date": dates,
        "Open": np.round(open_, 2),
        "High": np.round(high, 2),
        "Low": np.round(low, 2),
        "Close": np.round(close, 2),
        "Volume": volume,
    })


def descending_triangle(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Flat support ~95, descending resistance from ~110 → ~97.
    Last ~20 bars break below support.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    support = 95.0
    resist_start, resist_end = 110.0, 97.0
    resist_slope = np.linspace(resist_start, resist_end, n_bars)

    close = np.empty(n_bars)
    close[0] = 103.0
    for i in range(1, n_bars):
        mid = (support + resist_slope[i]) / 2
        pull = 0.05 * (mid - close[i - 1])
        bounce_support = max(0, 0.3 * (support - close[i - 1])) if close[i - 1] < support + 2 else 0
        bounce_resist = min(0, 0.3 * (resist_slope[i] - close[i - 1])) if close[i - 1] > resist_slope[i] - 2 else 0
        close[i] = close[i - 1] + pull + bounce_support + bounce_resist + rng.normal(0, 0.5)

    # Force breakout down in the last 15 bars
    for i in range(n_bars - 15, n_bars):
        close[i] = support - (i - (n_bars - 15)) * 0.35 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)


def ascending_breakout(n_bars: int = 300, seed: int = 99) -> pd.DataFrame:
    """
    Rising support from ~90 → ~108, flat resistance ~110.
    Last ~15 bars break above resistance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    resist = 110.0
    support_line = np.linspace(90, 108, n_bars)

    close = np.empty(n_bars)
    close[0] = 100.0
    for i in range(1, n_bars):
        mid = (support_line[i] + resist) / 2
        pull = 0.05 * (mid - close[i - 1])
        bounce_s = max(0, 0.3 * (support_line[i] - close[i - 1])) if close[i - 1] < support_line[i] + 2 else 0
        bounce_r = min(0, 0.3 * (resist - close[i - 1])) if close[i - 1] > resist - 2 else 0
        close[i] = close[i - 1] + pull + bounce_s + bounce_r + rng.normal(0, 0.5)

    for i in range(n_bars - 15, n_bars):
        close[i] = resist + (i - (n_bars - 15)) * 0.4 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)


def channel_with_breakout(n_bars: int = 300, seed: int = 77) -> pd.DataFrame:
    """
    Descending parallel channel (width ~8), then breakout up in the last 20 bars.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    channel_top = np.linspace(120, 104, n_bars)
    channel_width = 8.0
    channel_bot = channel_top - channel_width

    close = np.empty(n_bars)
    close[0] = 116.0
    for i in range(1, n_bars):
        mid = (channel_top[i] + channel_bot[i]) / 2
        pull = 0.06 * (mid - close[i - 1])
        bounce_t = min(0, 0.35 * (channel_top[i] - close[i - 1])) if close[i - 1] > channel_top[i] - 1.5 else 0
        bounce_b = max(0, 0.35 * (channel_bot[i] - close[i - 1])) if close[i - 1] < channel_bot[i] + 1.5 else 0
        close[i] = close[i - 1] + pull + bounce_t + bounce_b + rng.normal(0, 0.4)

    for i in range(n_bars - 20, n_bars):
        close[i] = channel_top[n_bars - 20] + (i - (n_bars - 20)) * 0.45 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)
