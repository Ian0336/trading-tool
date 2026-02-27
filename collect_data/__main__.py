"""
Fetch klines from Binance and save to a timestamped directory.

Run standalone:
    uv run python -m collect_data

Or import the ``collect()`` function from other scripts.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from .fetch import fetch_klines, make_run_dir, save_csv, to_break_df

INTERVAL = "15m"
LAST_N_DAYS = 7

DEFAULT_JOBS: List[Tuple[str, str]] = [
    ("BTCUSDT", "futures"),
    ("ETHUSDT", "futures"),
]


def _divider(text: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {text}")
    print(f"{'─' * 62}")


def collect(
    jobs: List[Tuple[str, str]] | None = None,
    interval: str = INTERVAL,
    last_n_days: int = LAST_N_DAYS,
    data_root: str | Path = "data",
) -> Path:
    """
    Fetch klines for every (symbol, market) pair and save CSVs into a
    timestamped subdirectory under *data_root*.

    Returns
    -------
    Path to the run directory, e.g. ``data/20260225_14/``.
    """
    if jobs is None:
        jobs = DEFAULT_JOBS

    run_dir = make_run_dir(data_root)

    for symbol, market in jobs:
        _divider(f"{symbol}  {interval}  [{market.upper()}]  last {last_n_days} days")

        raw_df = fetch_klines(
            symbol=symbol,
            interval=interval,
            last_n_days=last_n_days,
            market=market,
        )

        csv_path = run_dir / f"{symbol}_{market}_{interval}_{last_n_days}d.csv"
        save_csv(raw_df, csv_path)

        break_df = to_break_df(raw_df)

        print(f"\n  Rows         : {len(break_df)}")
        print(f"  First bar    : {break_df['Date'].iloc[0]}")
        print(f"  Last bar     : {break_df['Date'].iloc[-1]}")
        print(f"\n  head(3):\n")
        print(break_df.head(3).to_string(index=False))
        print(f"\n  tail(1):\n")
        print(break_df.tail(1).to_string(index=False))

    print(f"\n{'─' * 62}")
    print(f"  All done — CSVs saved in {run_dir}")
    print(f"{'─' * 62}\n")

    return run_dir


def main() -> None:
    collect()


if __name__ == "__main__":
    main()
