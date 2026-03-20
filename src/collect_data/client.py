"""
Binance public REST client
===========================
Handles:
  - Spot   → https://api.binance.com/api/v3/klines
  - USD-M Futures → https://fapi.binance.com/fapi/v1/klines

Features:
  - Automatic pagination: splits any time-range into ≤1000-bar chunks
  - Exponential back-off retry on transient errors (5xx, timeout)
  - Binance-specific back-off on 429 (rate limit) and 418 (IP ban)
  - No API key required (public market data endpoints)
"""

from __future__ import annotations

import logging
import time
from typing import Generator, List

import httpx

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ constants

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

SPOT_KLINE_PATH = "/api/v3/klines"
FUTURES_KLINE_PATH = "/fapi/v1/klines"

PAGE_LIMIT = 1000          # max rows per Binance request
DEFAULT_TIMEOUT = 10.0     # seconds
MAX_RETRIES = 6
RETRY_BASE_DELAY = 1.0     # seconds, doubles each attempt


# ------------------------------------------------------------------ helpers

def _ms(ts: int | float | str) -> int:
    """
    Convert various timestamp formats to milliseconds (int).

    Accepts:
      - int / float already in ms  (> 1e10)
      - int / float in seconds     (<= 1e10)
      - ISO 8601 string, e.g. "2024-01-01" or "2024-01-01T00:00:00Z"
    """
    if isinstance(ts, str):
        import pandas as pd
        return int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)
    ts = int(ts)
    if ts <= 10_000_000_000:   # seconds → ms
        ts *= 1000
    return ts


def _interval_ms(interval: str) -> int:
    """Return the duration of one bar in milliseconds."""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "3d": 259_200_000,
        "1w": 604_800_000,
    }
    key = interval.lower()
    if key not in mapping:
        raise ValueError(f"Unknown interval '{interval}'. Supported: {list(mapping)}")
    return mapping[key]


# ------------------------------------------------------------------ core fetcher

def _request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict,
) -> list:
    """
    GET ``url`` with ``params``, retry with exponential back-off.
    Handles 429 / 418 Binance rate-limit responses specially.
    """
    delay = RETRY_BASE_DELAY
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.get(url, params=params, timeout=DEFAULT_TIMEOUT)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay * 2))
                logger.warning("Rate limited (429). Sleeping %ds …", retry_after)
                time.sleep(retry_after)
                delay = retry_after * 2
                continue

            if resp.status_code == 418:
                retry_after = int(resp.headers.get("Retry-After", 60))
                logger.warning("IP banned (418). Sleeping %ds …", retry_after)
                time.sleep(retry_after)
                delay = retry_after * 2
                continue

            resp.raise_for_status()
            return resp.json()

        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.1fs …",
                attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 120)

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                last_exc = exc
                logger.warning(
                    "Server error %d on attempt %d. Retrying in %.1fs …",
                    exc.response.status_code, attempt, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 120)
            else:
                raise

    raise RuntimeError(
        f"Failed to fetch {url} after {MAX_RETRIES} attempts"
    ) from last_exc


# ------------------------------------------------------------------ pagination

def iter_kline_pages(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    market: str = "spot",
) -> Generator[List[list], None, None]:
    """
    Yield pages of raw Binance kline rows covering [start_ms, end_ms].
    Each page is a list of lists (Binance kline format).

    Parameters
    ----------
    market : ``"spot"`` or ``"futures"``
    """
    market = market.lower()
    if market == "spot":
        base = SPOT_BASE
        path = SPOT_KLINE_PATH
    elif market in ("futures", "usdm", "usd-m"):
        base = FUTURES_BASE
        path = FUTURES_KLINE_PATH
    else:
        raise ValueError(f"Unknown market '{market}'. Use 'spot' or 'futures'.")

    url = base + path
    bar_ms = _interval_ms(interval)

    current_start = start_ms

    with httpx.Client() as client:
        while current_start < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": PAGE_LIMIT,
            }

            rows: List[list] = _request_with_retry(client, url, params)

            if not rows:
                break

            yield rows

            last_open_ms = int(rows[-1][0])
            current_start = last_open_ms + bar_ms

            # Binance returned fewer rows than limit → we've reached the end
            if len(rows) < PAGE_LIMIT:
                break

            logger.debug(
                "Fetched %d bars up to %s, continuing …",
                len(rows),
                _ms_to_str(last_open_ms),
            )


def _ms_to_str(ms: int) -> str:
    import pandas as pd
    return str(pd.Timestamp(ms, unit="ms", tz="UTC"))
