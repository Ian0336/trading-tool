"""
High-level fetch API
=====================
fetch_klines() → raw DataFrame with all Binance kline columns
to_break_df()  → renamed/typed DataFrame ready for break/core.py

Binance kline columns (index):
  0  open_time       ms timestamp
  1  open
  2  high
  3  low
  4  close
  5  volume
  6  close_time      ms timestamp
  7  quote_asset_volume
  8  number_of_trades
  9  taker_buy_base_asset_volume
  10 taker_buy_quote_asset_volume
  11 ignore
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from .client import _ms, iter_kline_pages

logger = logging.getLogger(__name__)

MarketType = Literal["spot", "futures"]

# Column names for the raw Binance response
_RAW_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "_ignore",
]


def fetch_klines(
    symbol: str,
    interval: str = "15m",
    start: int | float | str | None = None,
    end: int | float | str | None = None,
    market: MarketType = "spot",
    last_n_days: int | None = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines for a single symbol over a time range.

    Parameters
    ----------
    symbol : e.g. ``"BTCUSDT"``
    interval : Binance interval string, e.g. ``"15m"``, ``"1h"``, ``"1d"``
    start : start time — ms timestamp, Unix seconds, or ISO8601 string.
            If omitted, ``last_n_days`` must be provided.
    end : end time — same formats as ``start``.
          Defaults to now (UTC).
    market : ``"spot"`` or ``"futures"``
    last_n_days : convenience shortcut — fetch this many days ending now.
                  Overrides ``start``/``end`` when provided.

    Returns
    -------
    pd.DataFrame with columns:
        open_time, open, high, low, close, volume, close_time,
        quote_volume, trades, taker_buy_base, taker_buy_quote
    open_time / close_time are UTC-aware pd.Timestamp.
    All price / volume columns are float64.
    """
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    if last_n_days is not None:
        end_ms = now_ms
        start_ms = end_ms - last_n_days * 86_400_000
    else:
        if start is None:
            raise ValueError("Provide either 'start' or 'last_n_days'.")
        start_ms = _ms(start)
        end_ms = _ms(end) if end is not None else now_ms

    if start_ms >= end_ms:
        raise ValueError(f"start_ms ({start_ms}) must be < end_ms ({end_ms})")

    logger.info(
        "Fetching %s %s [%s → %s] from %s …",
        symbol, interval,
        pd.Timestamp(start_ms, unit="ms", tz="UTC"),
        pd.Timestamp(end_ms, unit="ms", tz="UTC"),
        market,
    )

    all_rows: list[list] = []
    for page in iter_kline_pages(symbol, interval, start_ms, end_ms, market):
        all_rows.extend(page)

    if not all_rows:
        logger.warning("No data returned for %s %s.", symbol, interval)
        return pd.DataFrame(columns=_RAW_COLS[:-1])

    df = pd.DataFrame(all_rows, columns=_RAW_COLS)
    df = df.drop(columns=["_ignore"])

    # Remove any duplicate bars (Binance can return overlapping boundary bars)
    df = df.drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    # Types
    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)

    # Timestamps → UTC datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Drop bars beyond end_ms (last page may contain one extra bar)
    df = df[df["open_time"] <= pd.Timestamp(end_ms, unit="ms", tz="UTC")]
    df = df.reset_index(drop=True)

    logger.info("Fetched %d bars for %s.", len(df), symbol)
    return df


def to_break_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a ``fetch_klines()`` DataFrame into the format expected by
    ``break/core.py``:

        Date, Open, High, Low, Close, Volume

    - ``Date`` is tz-naive UTC (pandas works best with tz-naive for indexing).
    - All OHLCV columns are float64.
    """
    out = pd.DataFrame({
        "Date": df["open_time"].dt.tz_localize(None),
        "Open": df["open"],
        "High": df["high"],
        "Low": df["low"],
        "Close": df["close"],
        "Volume": df["volume"],
    })
    return out.reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save a klines DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)


def load_csv(path: str | Path, to_break: bool = True) -> pd.DataFrame:
    """
    Load a previously saved klines CSV.

    Parameters
    ----------
    to_break : if True, runs ``to_break_df()`` before returning.
    """
    path = Path(path)
    df = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    if to_break:
        return to_break_df(df)
    return df
