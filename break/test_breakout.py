"""
Integration test & demo for the trendline breakout detection module.

Runs all mock scenarios, prints breakout events, and saves charts to PNG.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .core import detect_latest_breakout, scan_all_breakouts
from .mock_data import (
    ascending_breakout,
    channel_with_breakout,
    descending_triangle,
    falling_wedge,
    head_and_shoulders_top,
    inverse_head_and_shoulders,
    m_top,
    rising_wedge,
    w_bottom,
)
from .visualize import plot_trendline_breakouts


def _divider(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def run_scenario(
    name: str,
    df: pd.DataFrame,
    save_dir: Path,
    **kwargs,
) -> None:
    _divider(name)

    result = detect_latest_breakout(df, **kwargs)
    if result is None:
        print("  [No result — not enough data]")
    else:
        print(f"  Latest bar index   : {result.bar}")
        print(f"  Breakout UP        : {result.breakout_up}")
        print(f"  Breakout DOWN      : {result.breakout_down}")
        if result.resistance:
            r = result.resistance
            print(f"  Resistance line    : y = {r.m:.4f}·x + {r.b:.2f}  "
                  f"(touches={r.touches}, violations={r.violations}, score={r.score:.1f})")
        if result.support:
            s = result.support
            print(f"  Support line       : y = {s.m:.4f}·x + {s.b:.2f}  "
                  f"(touches={s.touches}, violations={s.violations}, score={s.score:.1f})")

    events = scan_all_breakouts(df, **kwargs)
    print(f"\n  Total breakout events across history: {len(events)}")
    if events:
        rows = [asdict(e) for e in events]
        ev_df = pd.DataFrame(rows)
        print(ev_df.to_string(index=False, max_cols=10))

    fig = plot_trendline_breakouts(df, title=name, **kwargs)
    out_path = save_dir / f"{name.lower().replace(' ', '_')}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved → {out_path}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    save_dir = Path(__file__).resolve().parent / "output"
    save_dir.mkdir(exist_ok=True)

    common_params = dict(
        lookback_bars=200,
        pivot_left=3,
        pivot_right=3,
        tol_atr=0.8,
        margin_atr=0.15,
        max_violations=0,
        min_span_bars=20,
    )

    scenarios = [
        ("Descending Triangle Breakdown", descending_triangle(300)),
        ("Ascending Breakout Up", ascending_breakout(300)),
        ("Channel Breakout Up", channel_with_breakout(300)),
        ("M Top Double Top", m_top()),
        ("W Bottom Double Bottom", w_bottom()),
        ("Head and Shoulders Top", head_and_shoulders_top()),
        ("Inverse Head and Shoulders", inverse_head_and_shoulders()),
        ("Rising Wedge Bearish", rising_wedge()),
        ("Falling Wedge Bullish", falling_wedge()),
    ]

    for name, df in scenarios:
        run_scenario(name, df, save_dir, **common_params)

    print(f"\n{'=' * 60}")
    print(f"  All done — charts saved in {save_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
