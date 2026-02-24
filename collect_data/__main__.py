"""
Demo: fetch last 7 days of BTC / ETH 15m klines (Spot & Futures)
and show how to pipe the data into break/core.py.

Run:
    uv run python -m collect_data
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from .fetch import fetch_klines, save_csv, to_break_df

# Output directory for cached CSVs
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

INTERVAL = "15m"
LAST_N_DAYS = 7

JOBS = [
    # (symbol, market)
    ("BTCUSDT", "spot"),
    ("ETHUSDT", "spot"),
    ("BTCUSDT", "futures"),
    ("ETHUSDT", "futures"),
]


def _divider(text: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {text}")
    print(f"{'─' * 62}")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol, market in JOBS:
        _divider(f"{symbol}  {INTERVAL}  [{market.upper()}]  last {LAST_N_DAYS} days")

        # 1) Fetch raw klines
        raw_df = fetch_klines(
            symbol=symbol,
            interval=INTERVAL,
            last_n_days=LAST_N_DAYS,
            market=market,
        )

        # 2) Save raw CSV
        csv_path = DATA_DIR / f"{symbol}_{market}_{INTERVAL}_7d.csv"
        save_csv(raw_df, csv_path)

        # 3) Convert to break-compatible format
        break_df = to_break_df(raw_df)

        # 4) Preview
        print(f"\n  Rows         : {len(break_df)}")
        print(f"  First bar    : {break_df['Date'].iloc[0]}")
        print(f"  Last bar     : {break_df['Date'].iloc[-1]}")
        print(f"\n  head(3):\n")
        print(break_df.head(3).to_string(index=False))
        print(f"\n  tail(1):\n")
        print(break_df.tail(1).to_string(index=False))

        # 5) Quick sanity — pipe into break/core.py
        _run_breakout_check(symbol, market, break_df)

    print(f"\n{'─' * 62}")
    print(f"  All done — CSVs saved in {DATA_DIR}")
    print(f"{'─' * 62}\n")


def _run_breakout_check(symbol: str, market: str, break_df) -> None:
    """Run detect_latest_breakout and print the result."""
    try:
        import importlib
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        # 'break' is a Python keyword so we must use importlib to import the package
        break_core = importlib.import_module("break.core")
        detect_latest_breakout = break_core.detect_latest_breakout

        result = detect_latest_breakout(
            break_df,
            lookback_bars=200,
            pivot_left=3,
            pivot_right=3,
            tol_atr=0.8,
            margin_atr=0.15,
            max_violations=0,
        )
        if result is None:
            print("  Breakout check : not enough data")
            return

        print(f"\n  Breakout check on latest bar (bar={result.bar}):")
        print(f"    Breakout UP        : {result.breakout_up}")
        print(f"    Breakout DOWN      : {result.breakout_down}")
        if result.resistance:
            r = result.resistance
            print(f"    Resistance line    : y = {r.m:.6f}·x + {r.b:.2f}  "
                  f"(touches={r.touches}, score={r.score:.1f})")
        if result.support:
            s = result.support
            print(f"    Support line       : y = {s.m:.6f}·x + {s.b:.2f}  "
                  f"(touches={s.touches}, score={s.score:.1f})")
    except Exception as exc:
        print(f"  Breakout check skipped: {exc}")


if __name__ == "__main__":
    main()
