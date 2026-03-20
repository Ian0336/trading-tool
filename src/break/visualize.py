"""
Visualization helpers for trendline breakout detection.

Produces a candlestick-style chart overlaid with:
  - Swing-high / swing-low pivots (triangles)
  - Best-fit resistance & support trendlines
  - Breakout event markers (arrows)
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    BreakoutEvent,
    Pivot,
    TrendlineInfo,
    atr,
    best_trendline_from_pivots,
    clean_pivots,
    pivots_fractal,
    scan_all_breakouts,
)


def _plot_candlesticks(ax: plt.Axes, df: pd.DataFrame, dates: np.ndarray) -> None:
    """Minimal OHLC bars drawn with matplotlib (no mplfinance dependency)."""
    up = df["Close"] >= df["Open"]
    down = ~up

    color_up = "#26a69a"
    color_down = "#ef5350"

    # Wicks
    ax.vlines(dates[up], df["Low"][up], df["High"][up], color=color_up, linewidth=0.6)
    ax.vlines(dates[down], df["Low"][down], df["High"][down], color=color_down, linewidth=0.6)

    # Bodies
    bar_width = 0.6
    ax.bar(
        dates[up], (df["Close"] - df["Open"])[up], bottom=df["Open"][up],
        width=bar_width, color=color_up, edgecolor=color_up, linewidth=0.3,
    )
    ax.bar(
        dates[down], (df["Open"] - df["Close"])[down], bottom=df["Close"][down],
        width=bar_width, color=color_down, edgecolor=color_down, linewidth=0.3,
    )


def plot_trendline_breakouts(
    df: pd.DataFrame,
    title: str = "Trendline Breakout Detection",
    lookback_bars: int = 200,
    pivot_left: int = 3,
    pivot_right: int = 3,
    tol_atr: float = 0.8,
    margin_atr: float = 0.15,
    max_violations: int = 0,
    min_span_bars: int = 20,
    figsize: tuple = (18, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    All-in-one chart: candlesticks + pivots + trendlines + breakout arrows.

    Parameters
    ----------
    df : OHLCV DataFrame with a ``Date`` column (or DatetimeIndex).
    save_path : If given, the figure is saved to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    df = df.reset_index(drop=True).copy()
    atr_s = atr(df, 14)

    all_pivots = clean_pivots(pivots_fractal(df, pivot_left, pivot_right))
    events = scan_all_breakouts(
        df,
        lookback_bars=lookback_bars,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        tol_atr=tol_atr,
        margin_atr=margin_atr,
        max_violations=max_violations,
        min_span_bars=min_span_bars,
    )

    # Use integer bar indices for x-axis, label with dates
    x = np.arange(len(df))
    has_dates = "Date" in df.columns

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=figsize, height_ratios=[4, 1],
        sharex=True, gridspec_kw={"hspace": 0.05},
    )

    # --- Candlesticks ---
    _plot_candlesticks(ax_price, df, x)

    # --- Pivots ---
    for p in all_pivots:
        marker = "v" if p.kind == "H" else "^"
        color = "#e91e63" if p.kind == "H" else "#2196f3"
        y_offset = p.price * 1.005 if p.kind == "H" else p.price * 0.995
        ax_price.scatter(p.bar, y_offset, marker=marker, color=color, s=40, zorder=5)

    # --- Best trendlines (fitted on last confirmed window) ---
    t = len(df) - 1
    t_fit = t - pivot_right
    start = max(0, t_fit - lookback_bars)

    high_pivs = [p for p in all_pivots if p.kind == "H" and start <= p.bar <= t_fit]
    low_pivs = [p for p in all_pivots if p.kind == "L" and start <= p.bar <= t_fit]

    mR, bR, infoR = best_trendline_from_pivots(
        high_pivs, "resistance", atr_s,
        tol_atr=tol_atr, max_violations=max_violations,
        min_span_bars=min_span_bars,
    )
    mS, bS, infoS = best_trendline_from_pivots(
        low_pivs, "support", atr_s,
        tol_atr=tol_atr, max_violations=max_violations,
        min_span_bars=min_span_bars,
    )

    extend_bars = 10
    if mR is not None:
        x_line = np.arange(start, min(t + extend_bars, len(df)))
        y_line = mR * x_line + bR
        ax_price.plot(
            x_line, y_line, "--", color="#e91e63", linewidth=1.4, alpha=0.85,
            label=f"Resistance (touches={infoR.touches}, score={infoR.score:.1f})",
        )

    if mS is not None:
        x_line = np.arange(start, min(t + extend_bars, len(df)))
        y_line = mS * x_line + bS
        ax_price.plot(
            x_line, y_line, "--", color="#2196f3", linewidth=1.4, alpha=0.85,
            label=f"Support (touches={infoS.touches}, score={infoS.score:.1f})",
        )

    # --- Breakout arrows ---
    for ev in events:
        if ev.direction == "up":
            ax_price.annotate(
                "", xy=(ev.bar, ev.close),
                xytext=(ev.bar, ev.close * 0.985),
                arrowprops=dict(arrowstyle="->", color="#00c853", lw=2),
            )
            ax_price.scatter(ev.bar, ev.close, marker="*", color="#00c853", s=120, zorder=6)
        else:
            ax_price.annotate(
                "", xy=(ev.bar, ev.close),
                xytext=(ev.bar, ev.close * 1.015),
                arrowprops=dict(arrowstyle="->", color="#ff1744", lw=2),
            )
            ax_price.scatter(ev.bar, ev.close, marker="*", color="#ff1744", s=120, zorder=6)

    ax_price.set_title(title, fontsize=14, fontweight="bold")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.25)
    ax_price.set_ylabel("Price")

    # --- Volume ---
    if "Volume" in df.columns:
        colors = np.where(df["Close"] >= df["Open"], "#26a69a", "#ef5350")
        ax_vol.bar(x, df["Volume"], color=colors, width=0.7, alpha=0.7)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.25)

    # X-axis date labels (show ~10 labels)
    if has_dates:
        n_labels = 10
        step = max(1, len(df) // n_labels)
        tick_positions = list(range(0, len(df), step))
        tick_labels = [df["Date"].iloc[i].strftime("%Y-%m-%d") for i in tick_positions]
        ax_vol.set_xticks(tick_positions)
        ax_vol.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax_vol.set_xlabel("Date")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
