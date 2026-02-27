"""
End-to-end pipeline: fetch data → breakout scan → chart.

Usage:
    uv run python run_analysis.py               # fetch fresh data, then analyse
    uv run python run_analysis.py 20260225_14    # skip fetch, analyse existing run dir
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from collect_data import collect
from collect_data.fetch import load_csv

core = importlib.import_module("break.core")
viz = importlib.import_module("break.visualize")

DATA_ROOT = Path("data")


def _find_run_dir(name: str) -> Path:
    d = DATA_ROOT / name
    if not d.is_dir():
        raise FileNotFoundError(f"Run directory not found: {d}")
    return d


def analyze(title: str, csv_path: Path, chart_dir: Path) -> None:
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    df = load_csv(csv_path)
    print(f"\n{'=' * 60}")
    print(f"  {title}  ({len(df)} bars)")
    print(f"  {df['Date'].iloc[0]}  →  {df['Date'].iloc[-1]}")
    print(f"{'=' * 60}")

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

    events = core.scan_all_breakouts(df)
    print(f"\n  Historical breakout events: {len(events)}")
    for e in events:
        print(f"    {e.timestamp}  {e.direction:>4}  close={e.close:.2f}  "
              f"line={e.line_value:.2f}  atr={e.atr_value:.2f}")

    fig = viz.plot_trendline_breakouts(df, title=title)
    out = chart_dir / f"{csv_path.stem}_chart.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\n  Chart → {out}")

    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    # If a run directory name is given, skip fetching and use it directly.
    # Otherwise, fetch fresh data first.
    if len(sys.argv) > 1:
        run_dir = _find_run_dir(sys.argv[1])
    else:
        run_dir = collect()

    chart_dir = run_dir / "charts"
    chart_dir.mkdir(exist_ok=True)

    print(f"  Analysing run directory: {run_dir}")

    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        print(f"  No CSV files found in {run_dir}")
        return

    for csv_path in csvs:
        title = csv_path.stem.replace("_", " ").title()
        analyze(title, csv_path, chart_dir)

    print(f"\n{'=' * 60}")
    print(f"  Done — all charts in {chart_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
