"""
Quick analysis script: load cached CSV → breakout scan → chart.

Usage:
    uv run python run_analysis.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

from collect_data.fetch import load_csv, save_csv, fetch_klines, to_break_df

core = importlib.import_module("break.core")
viz = importlib.import_module("break.visualize")

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    ("BTCUSDT Spot 15m", "BTCUSDT_spot_15m_7d.csv"),
    ("ETHUSDT Spot 15m", "ETHUSDT_spot_15m_7d.csv"),
    ("BTCUSDT Futures 15m", "BTCUSDT_futures_15m_7d.csv"),
    ("ETHUSDT Futures 15m", "ETHUSDT_futures_15m_7d.csv"),
]


def analyze(title: str, csv_name: str) -> None:
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    df = load_csv(csv_path)
    print(f"\n{'=' * 60}")
    print(f"  {title}  ({len(df)} bars)")
    print(f"  {df['Date'].iloc[0]}  →  {df['Date'].iloc[-1]}")
    print(f"{'=' * 60}")

    # Latest bar check
    result = core.detect_latest_breakout(df)
    if result:
        print(f"  Latest bar breakout UP   : {result.breakout_up}")
        print(f"  Latest bar breakout DOWN : {result.breakout_down}")
        if result.resistance:
            r = result.resistance
            print(f"  Resistance: touches={r.touches}, score={r.score:.1f}")
        if result.support:
            s = result.support
            print(f"  Support:    touches={s.touches}, score={s.score:.1f}")

    # Full scan
    events = core.scan_all_breakouts(df)
    print(f"\n  Historical breakout events: {len(events)}")
    for e in events:
        print(f"    {e.timestamp}  {e.direction:>4}  close={e.close:.2f}  "
              f"line={e.line_value:.2f}  atr={e.atr_value:.2f}")

    # Chart
    fig = viz.plot_trendline_breakouts(df, title=title)
    out = OUTPUT_DIR / f"{csv_name.replace('.csv', '')}_chart.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\n  Chart → {out}")

    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    for title, csv_name in FILES:
        analyze(title, csv_name)

    print(f"\n{'=' * 60}")
    print(f"  Done — all charts in {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
