"""
Mock OHLCV data generators for testing trendline breakout detection.

Scenarios provided:
  1. **descending_triangle** – flat support + descending resistance → breakout down
  2. **ascending_breakout** – rising support + flat resistance → breakout up
  3. **channel_with_breakout** – parallel descending channel → breakout up at the end
  4. **m_top** – double top at similar highs → neckline breakdown
  5. **w_bottom** – double bottom at similar lows → neckline breakout up
  6. **head_and_shoulders_top** – left shoulder / head / right shoulder → neckline breakdown
  7. **inverse_head_and_shoulders** – inverted H&S → neckline breakout up
  8. **rising_wedge** – converging upward trend → breakout down
  9. **falling_wedge** – converging downward trend → breakout up
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

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


def _phases_to_close(
    phases: list[tuple[float, float, int]],
    noise_std: float = 0.4,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a close series from a list of ``(start_price, end_price, n_bars)``
    phases.  Each phase is a noisy linear interpolation; phases are
    concatenated seamlessly.
    """
    rng = np.random.default_rng(seed)
    segments: list[np.ndarray] = []
    for start, end, n in phases:
        seg = np.linspace(start, end, n) + rng.normal(0, noise_std, n)
        segments.append(seg)
    return np.round(np.concatenate(segments), 2)


# ------------------------------------------------------------------
# Pattern 1: Descending triangle
# ------------------------------------------------------------------

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

    for i in range(n_bars - 15, n_bars):
        close[i] = support - (i - (n_bars - 15)) * 0.35 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 2: Ascending triangle (breakout up)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Pattern 3: Descending channel → breakout up
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Pattern 4: M-top (double top)
# ------------------------------------------------------------------

def m_top(n_bars: int = 300, seed: int = 55) -> pd.DataFrame:
    """
    Double top at ~110 with neckline ~100.

    Shape: rise → peak1 → pullback to neckline → peak2 → breakdown.
    The resistance line is nearly flat connecting the two peaks;
    the support/neckline gets broken at the end.
    """
    phases = [
        (90.0, 110.0, 70),
        (110.0, 109.5, 20),
        (109.0, 100.0, 40),
        (100.0, 100.5, 15),
        (101.0, 110.5, 45),
        (110.5, 109.0, 20),
        (108.5, 100.5, 35),
        (100.0, 97.0, 25),
        (96.5, 91.0, 30),
    ]
    close = _phases_to_close(phases, noise_std=0.45, seed=seed)
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 5: W-bottom (double bottom)
# ------------------------------------------------------------------

def w_bottom(n_bars: int = 300, seed: int = 60) -> pd.DataFrame:
    """
    Double bottom at ~90 with neckline ~100.

    Shape: decline → trough1 → bounce to neckline → trough2 → breakout up.
    """
    phases = [
        (110.0, 90.0, 65),
        (90.0, 90.5, 20),
        (91.0, 100.0, 40),
        (100.0, 99.5, 15),
        (99.0, 90.5, 40),
        (90.5, 91.0, 20),
        (91.5, 99.5, 35),
        (100.0, 101.5, 20),
        (102.0, 108.0, 30),
    ]
    close = _phases_to_close(phases, noise_std=0.45, seed=seed)
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 6: Head and Shoulders Top
# ------------------------------------------------------------------

def head_and_shoulders_top(n_bars: int = 350, seed: int = 33) -> pd.DataFrame:
    """
    Classic H&S top: left shoulder ~108, head ~115, right shoulder ~108,
    neckline ~100.  Breaks down through neckline at the end.
    """
    phases = [
        (92.0, 108.0, 45),
        (108.0, 107.0, 15),
        (106.5, 100.0, 30),
        (100.0, 100.5, 10),
        (101.0, 115.0, 40),
        (115.0, 114.0, 15),
        (113.5, 100.5, 35),
        (100.5, 100.0, 10),
        (100.5, 108.5, 35),
        (108.5, 107.5, 15),
        (107.0, 100.5, 30),
        (100.0, 98.0, 20),
        (97.5, 91.0, 30),
    ]
    close = _phases_to_close(phases, noise_std=0.5, seed=seed)
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 7: Inverse Head and Shoulders (H&S Bottom)
# ------------------------------------------------------------------

def inverse_head_and_shoulders(n_bars: int = 350, seed: int = 44) -> pd.DataFrame:
    """
    Inverse H&S: left shoulder ~92, head ~85, right shoulder ~92,
    neckline ~100.  Breaks up through neckline at the end.
    """
    phases = [
        (108.0, 92.0, 45),
        (92.0, 93.0, 15),
        (93.5, 100.0, 30),
        (100.0, 99.5, 10),
        (99.0, 85.0, 40),
        (85.0, 86.0, 15),
        (86.5, 99.5, 35),
        (99.5, 100.0, 10),
        (99.5, 92.0, 35),
        (92.0, 93.0, 15),
        (93.5, 99.5, 30),
        (100.0, 102.0, 20),
        (102.5, 110.0, 30),
    ]
    close = _phases_to_close(phases, noise_std=0.5, seed=seed)
    dates = pd.bdate_range("2024-01-01", periods=len(close))
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 8: Rising Wedge (bearish)
# ------------------------------------------------------------------

def rising_wedge(n_bars: int = 300, seed: int = 66) -> pd.DataFrame:
    """
    Both support and resistance rise, but support rises faster so the
    channel converges.  Eventually price breaks down through support.

    Resistance: ~100 → ~112,  Support: ~94 → ~110  (converging).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    resist = np.linspace(100, 112, n_bars)
    support = np.linspace(94, 110, n_bars)

    close = np.empty(n_bars)
    close[0] = 97.0
    for i in range(1, n_bars):
        mid = (resist[i] + support[i]) / 2
        pull = 0.06 * (mid - close[i - 1])
        bounce_r = min(0, 0.35 * (resist[i] - close[i - 1])) if close[i - 1] > resist[i] - 1.5 else 0
        bounce_s = max(0, 0.35 * (support[i] - close[i - 1])) if close[i - 1] < support[i] + 1.5 else 0
        close[i] = close[i - 1] + pull + bounce_r + bounce_s + rng.normal(0, 0.4)

    base = support[n_bars - 20]
    for i in range(n_bars - 20, n_bars):
        close[i] = base - (i - (n_bars - 20)) * 0.4 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)


# ------------------------------------------------------------------
# Pattern 9: Falling Wedge (bullish)
# ------------------------------------------------------------------

def falling_wedge(n_bars: int = 300, seed: int = 88) -> pd.DataFrame:
    """
    Both support and resistance fall, but resistance falls faster so the
    channel converges.  Eventually price breaks up through resistance.

    Resistance: ~110 → ~96,  Support: ~104 → ~94  (converging).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)

    resist = np.linspace(110, 96, n_bars)
    support = np.linspace(104, 94, n_bars)

    close = np.empty(n_bars)
    close[0] = 107.0
    for i in range(1, n_bars):
        mid = (resist[i] + support[i]) / 2
        pull = 0.06 * (mid - close[i - 1])
        bounce_r = min(0, 0.35 * (resist[i] - close[i - 1])) if close[i - 1] > resist[i] - 1.5 else 0
        bounce_s = max(0, 0.35 * (support[i] - close[i - 1])) if close[i - 1] < support[i] + 1.5 else 0
        close[i] = close[i - 1] + pull + bounce_r + bounce_s + rng.normal(0, 0.35)

    base = resist[n_bars - 20]
    for i in range(n_bars - 20, n_bars):
        close[i] = base + (i - (n_bars - 20)) * 0.4 + rng.normal(0, 0.2)

    close = np.round(close, 2)
    return _make_ohlcv(dates, close, seed=seed)
