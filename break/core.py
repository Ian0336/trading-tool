"""
Trendline Breakout Detection
=============================
Automatically identify support/resistance trendlines from swing pivots,
then detect breakouts using a cross-over rule with ATR-based tolerance.

Workflow:
  1. Compute ATR for volatility-adaptive thresholds.
  2. Detect swing-high / swing-low pivots (fractal method).
  3. Enumerate pivot pairs → candidate trendlines → score by touches/violations.
  4. Pick the best resistance line (from swing highs) and support line (from swing lows).
  5. On each bar, check whether price crosses the line beyond a tolerance margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Pivot:
    bar: int
    kind: str   # "H" or "L"
    price: float


@dataclass
class TrendlineInfo:
    m: float
    b: float
    touches: int
    violations: int
    span: int
    score: float


@dataclass
class BreakoutResult:
    bar: int
    breakout_up: bool
    breakout_down: bool
    resistance: Optional[TrendlineInfo]
    support: Optional[TrendlineInfo]


@dataclass
class BreakoutEvent:
    bar: int
    timestamp: object
    direction: str          # "up" or "down"
    close: float
    line_value: float
    line_m: float
    line_b: float
    atr_value: float
    margin: float


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range over *n* periods."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


# ---------------------------------------------------------------------------
# Swing-point (fractal) pivots
# ---------------------------------------------------------------------------
def pivots_fractal(
    df: pd.DataFrame, left: int = 3, right: int = 3
) -> List[Pivot]:
    """
    Return fractal pivots.

    A bar is a swing-high when its High is the max of the surrounding
    ``left + 1 + right`` bars.  Swing-low is the analogous min of Low.

    .. note:: The rightmost ``right`` bars cannot be confirmed yet, which
       prevents look-ahead bias when used with ``right >= 1``.
    """
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    pivots: List[Pivot] = []
    for i in range(left, len(df) - right):
        window_h = H[i - left : i + right + 1]
        window_l = L[i - left : i + right + 1]
        if H[i] == window_h.max():
            pivots.append(Pivot(i, "H", float(H[i])))
        if L[i] == window_l.min():
            pivots.append(Pivot(i, "L", float(L[i])))
    pivots.sort(key=lambda p: p.bar)
    return pivots


def clean_pivots(pivots: List[Pivot]) -> List[Pivot]:
    """
    Merge consecutive same-type pivots (keep the more extreme one)
    and enforce strict H-L alternation.
    """
    out: List[Pivot] = []
    for p in pivots:
        if not out:
            out.append(p)
            continue
        prev = out[-1]
        if p.kind == prev.kind:
            if (p.kind == "H" and p.price > prev.price) or (
                p.kind == "L" and p.price < prev.price
            ):
                out[-1] = p
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------------------
# Best trendline from a set of pivots
# ---------------------------------------------------------------------------
def best_trendline_from_pivots(
    piv_points: List[Pivot],
    kind: str,
    atr_s: pd.Series,
    tol_atr: float = 0.8,
    max_violations: int = 0,
    min_span_bars: int = 20,
) -> Tuple[Optional[float], Optional[float], Optional[TrendlineInfo]]:
    """
    Enumerate all pivot-pair candidate lines, score each one by
    *touches* (pivot within tolerance) vs *violations* (pivot clearly
    on the wrong side), and return the best ``(m, b, TrendlineInfo)``.

    Parameters
    ----------
    kind : ``"resistance"`` or ``"support"``
    tol_atr : How many ATRs a pivot may deviate and still count as a "touch".
    max_violations : Maximum allowed pivots clearly on the wrong side.
    min_span_bars : Minimum bar-distance between the two anchor pivots.
    """
    if len(piv_points) < 2:
        return None, None, None

    xs = np.array([p.bar for p in piv_points], dtype=float)
    ys = np.array([p.price for p in piv_points], dtype=float)

    i0, i1 = int(xs.min()), int(xs.max())
    avg_atr = float(np.nanmean(atr_s.iloc[i0 : i1 + 1]))
    if np.isnan(avg_atr) or avg_atr == 0:
        return None, None, None
    tol = tol_atr * avg_atr

    best: Optional[dict] = None
    n = len(piv_points)

    for i in range(n - 1):
        for j in range(i + 1, n):
            x1, y1 = xs[i], ys[i]
            x2, y2 = xs[j], ys[j]
            if x2 == x1:
                continue
            span = int(abs(x2 - x1))
            if span < min_span_bars:
                continue

            m = (y2 - y1) / (x2 - x1)
            b_val = y1 - m * x1
            y_line = m * xs + b_val

            # residual: positive = "correct side", negative = violation
            if kind == "resistance":
                resid = y_line - ys
            else:
                resid = ys - y_line

            violations = int(np.sum(resid < -tol))
            if violations > max_violations:
                continue

            touches = int(np.sum(np.abs(resid) <= tol))
            mean_abs = float(np.mean(np.abs(resid)))
            score = touches * 10 - violations * 50 + span / 50 - mean_abs / tol

            if best is None or score > best["score"]:
                best = dict(
                    m=float(m),
                    b=float(b_val),
                    touches=touches,
                    violations=violations,
                    span=span,
                    score=float(score),
                )

    if best is None:
        return None, None, None

    info = TrendlineInfo(**best)
    return best["m"], best["b"], info


# ---------------------------------------------------------------------------
# Single-bar breakout check
# ---------------------------------------------------------------------------
def is_breakout(
    df: pd.DataFrame,
    atr_s: pd.Series,
    m: float,
    b: float,
    kind: str,
    t: int,
    margin_atr: float = 0.15,
    use_close: bool = True,
) -> bool:
    """
    Cross-over rule: price crosses the trendline (with ATR margin)
    between bar ``t-1`` and bar ``t``.
    """
    if t <= 0 or t >= len(df):
        return False

    a = float(atr_s.iloc[t])
    if np.isnan(a) or a == 0:
        return False
    margin = margin_atr * a

    if kind == "resistance":
        col = "Close" if use_close else "High"
        p_t = float(df[col].iloc[t])
        p_prev = float(df[col].iloc[t - 1])
        line_t = m * t + b
        line_prev = m * (t - 1) + b
        return (p_t > line_t + margin) and (p_prev <= line_prev + margin)
    else:
        col = "Close" if use_close else "Low"
        p_t = float(df[col].iloc[t])
        p_prev = float(df[col].iloc[t - 1])
        line_t = m * t + b
        line_prev = m * (t - 1) + b
        return (p_t < line_t - margin) and (p_prev >= line_prev - margin)


# ---------------------------------------------------------------------------
# Detect breakout on the latest bar
# ---------------------------------------------------------------------------
def detect_latest_breakout(
    df: pd.DataFrame,
    lookback_bars: int = 200,
    pivot_left: int = 3,
    pivot_right: int = 3,
    tol_atr: float = 0.8,
    margin_atr: float = 0.15,
    max_violations: int = 0,
    min_span_bars: int = 20,
) -> Optional[BreakoutResult]:
    """
    On the **latest confirmed bar** find the best resistance & support
    trendlines and check for a breakout.
    """
    df = df.reset_index(drop=True).copy()
    atr_s = atr(df, 14)

    all_pivots = clean_pivots(pivots_fractal(df, pivot_left, pivot_right))

    t = len(df) - 1
    t_fit = t - pivot_right
    if t_fit <= 0:
        return None

    start = max(0, t_fit - lookback_bars)

    # --- resistance (swing highs) ---
    high_pivs = [p for p in all_pivots if p.kind == "H" and start <= p.bar <= t_fit]
    mR, bR, infoR = best_trendline_from_pivots(
        high_pivs, "resistance", atr_s,
        tol_atr=tol_atr, max_violations=max_violations,
        min_span_bars=min_span_bars,
    )
    breakout_up = False
    if mR is not None:
        breakout_up = is_breakout(
            df, atr_s, mR, bR, "resistance", t, margin_atr=margin_atr
        )

    # --- support (swing lows) ---
    low_pivs = [p for p in all_pivots if p.kind == "L" and start <= p.bar <= t_fit]
    mS, bS, infoS = best_trendline_from_pivots(
        low_pivs, "support", atr_s,
        tol_atr=tol_atr, max_violations=max_violations,
        min_span_bars=min_span_bars,
    )
    breakout_down = False
    if mS is not None:
        breakout_down = is_breakout(
            df, atr_s, mS, bS, "support", t, margin_atr=margin_atr
        )

    return BreakoutResult(
        bar=t,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        resistance=infoR if mR is not None else None,
        support=infoS if mS is not None else None,
    )


# ---------------------------------------------------------------------------
# Scan entire history (no look-ahead) → list of BreakoutEvent
# ---------------------------------------------------------------------------
def scan_all_breakouts(
    df: pd.DataFrame,
    lookback_bars: int = 200,
    pivot_left: int = 3,
    pivot_right: int = 3,
    tol_atr: float = 0.8,
    margin_atr: float = 0.15,
    max_violations: int = 0,
    min_span_bars: int = 20,
    min_pivots_for_line: int = 3,
) -> List[BreakoutEvent]:
    """
    Walk forward through every bar (after a warm-up window) and record
    each breakout event.  No future data is used: pivots are only
    confirmed after ``pivot_right`` bars have passed.

    Returns a list of :class:`BreakoutEvent` sorted by bar index.
    """
    df = df.reset_index(drop=True).copy()
    atr_s = atr(df, 14)

    all_pivots = clean_pivots(pivots_fractal(df, pivot_left, pivot_right))

    events: List[BreakoutEvent] = []

    warmup = max(lookback_bars // 2, min_span_bars + pivot_left + pivot_right + 14)

    for t in range(warmup, len(df)):
        t_fit = t - pivot_right
        if t_fit <= 0:
            continue

        start = max(0, t_fit - lookback_bars)
        a_val = float(atr_s.iloc[t])
        if np.isnan(a_val) or a_val == 0:
            continue

        # resistance
        high_pivs = [
            p for p in all_pivots if p.kind == "H" and start <= p.bar <= t_fit
        ]
        if len(high_pivs) >= min_pivots_for_line:
            mR, bR, infoR = best_trendline_from_pivots(
                high_pivs, "resistance", atr_s,
                tol_atr=tol_atr, max_violations=max_violations,
                min_span_bars=min_span_bars,
            )
            if mR is not None and is_breakout(
                df, atr_s, mR, bR, "resistance", t, margin_atr=margin_atr
            ):
                line_val = mR * t + bR
                ts = df["Date"].iloc[t] if "Date" in df.columns else t
                events.append(
                    BreakoutEvent(
                        bar=t,
                        timestamp=ts,
                        direction="up",
                        close=float(df["Close"].iloc[t]),
                        line_value=line_val,
                        line_m=mR,
                        line_b=bR,
                        atr_value=a_val,
                        margin=margin_atr * a_val,
                    )
                )

        # support
        low_pivs = [
            p for p in all_pivots if p.kind == "L" and start <= p.bar <= t_fit
        ]
        if len(low_pivs) >= min_pivots_for_line:
            mS, bS, infoS = best_trendline_from_pivots(
                low_pivs, "support", atr_s,
                tol_atr=tol_atr, max_violations=max_violations,
                min_span_bars=min_span_bars,
            )
            if mS is not None and is_breakout(
                df, atr_s, mS, bS, "support", t, margin_atr=margin_atr
            ):
                line_val = mS * t + bS
                ts = df["Date"].iloc[t] if "Date" in df.columns else t
                events.append(
                    BreakoutEvent(
                        bar=t,
                        timestamp=ts,
                        direction="down",
                        close=float(df["Close"].iloc[t]),
                        line_value=line_val,
                        line_m=mS,
                        line_b=bS,
                        atr_value=a_val,
                        margin=margin_atr * a_val,
                    )
                )

    events.sort(key=lambda e: e.bar)
    return events
