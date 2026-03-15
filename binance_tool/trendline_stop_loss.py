#!/usr/bin/env python3
"""
Trendline Stop-Loss Monitor
============================
1. Fetches all open USDⓈ-M Futures positions.
2. Lets you pick the position to protect.
3. Defines the trend line the same way TradingView does:
     • Two anchor points, each given as (price, N bars ago).
     • Slope = Δprice / Δbars  (price per bar), then converted to price/ms
       so the line can be evaluated at any future moment.
4. Polls mark price every 60 s; fires a market close order when price crosses
   the extrapolated line.

Key design rules
----------------
- LONG position  → stop fires when mark price falls *to or below* the line.
- SHORT position → stop fires when mark price rises *to or above* the line.
- "N bars ago" anchors to the OPEN of that bar (same convention as TradingView
  bar-index coordinates).  The default interval is 15 m.
- All time display uses UTC+8 (Asia/Taipei).

Usage
-----
    uv run python binance_tool/trendline_stop_loss.py
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import sys
import time
from datetime import timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx
import pandas as pd

# ── Network retry constants ─────────────────────────────────────────────────────
_RETRY_INITIAL_WAIT_S = 5     # first back-off after a network failure
_RETRY_MAX_WAIT_S     = 60   # cap for exponential back-off
_NETWORK_ERRORS       = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.get_keys import get_secret

DISPLAY_TZ = "Asia/Taipei"   # UTC+8

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Binance Futures constants ──────────────────────────────────────────────────
BASE_URL = "https://fapi.binance.com"

_INTERVAL_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
}


# ── Credentials ────────────────────────────────────────────────────────────────

def _load_credentials() -> tuple[str, str]:
    """Load API key and secret from macOS Keychain."""
    api_key = get_secret("Binance_API_Key", "trading-tool")
    api_secret = get_secret("Binance_API_Secret", "trading-tool")

    missing: list[str] = []
    if not api_key:
        missing.append("API key  (service='Binance_API_Key',    account='trading-tool')")
    if not api_secret:
        missing.append("API secret (service='Binance_API_Secret', account='trading-tool')")

    if missing:
        print("[ERROR] Could not load from macOS Keychain:")
        for m in missing:
            print(f"  • {m}")
        print("\nTo add them, run:")
        print("  security add-generic-password -s Binance_API_Key    -a trading-tool -w <KEY>")
        print("  security add-generic-password -s Binance_API_Secret -a trading-tool -w <SECRET>")
        sys.exit(1)

    return api_key, api_secret


# ── Signed HTTP helpers ────────────────────────────────────────────────────────

def _build_client(api_key: str) -> httpx.Client:
    return httpx.Client(headers={"X-MBX-APIKEY": api_key}, timeout=10)


def _sign(params: dict, api_secret: str) -> dict:
    """Append HMAC-SHA256 signature in-place and return the dict."""
    query = urlencode(params, doseq=True)
    sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def _request_with_retry(fn, *args, **kwargs):
    """
    Call ``fn(*args, **kwargs)`` and retry with exponential back-off whenever
    a transient network error is raised.  Gives up only on KeyboardInterrupt.
    """
    wait = _RETRY_INITIAL_WAIT_S
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except _NETWORK_ERRORS as exc:
            attempt += 1
            log.warning(
                "Network error (attempt %d): %s — retrying in %ds …",
                attempt, exc, wait,
            )
            time.sleep(wait)
            wait = min(wait + 5, _RETRY_MAX_WAIT_S)
        except httpx.HTTPStatusError as exc:
            # Non-2xx from Binance: surface immediately, don't retry
            raise


def _signed_get(client: httpx.Client, api_secret: str,
                path: str, extra: dict | None = None) -> list | dict:
    def _do():
        params: dict = {
            "timestamp":  int(time.time() * 1000),
            "recvWindow": 5000,
            **(extra or {}),
        }
        _sign(params, api_secret)
        resp = client.get(f"{BASE_URL}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    return _request_with_retry(_do)


def _signed_post(client: httpx.Client, api_secret: str,
                 path: str, data: dict | None = None) -> dict:
    def _do():
        params: dict = {
            "timestamp":  int(time.time() * 1000),
            "recvWindow": 5000,
            **(data or {}),
        }
        _sign(params, api_secret)
        resp = client.post(f"{BASE_URL}{path}", data=params)
        resp.raise_for_status()
        return resp.json()

    return _request_with_retry(_do)


# ── Position helpers ────────────────────────────────────────────────────────────

def fetch_open_positions(client: httpx.Client, secret: str) -> list[dict]:
    """Return all positions with a non-zero positionAmt."""
    all_pos = _signed_get(client, secret, "/fapi/v3/positionRisk")
    assert isinstance(all_pos, list)
    return [p for p in all_pos if float(p["positionAmt"]) != 0.0]


def is_hedge_mode(client: httpx.Client, secret: str) -> bool:
    """Return True when account is in Hedge Mode (dual side)."""
    data = _signed_get(client, secret, "/fapi/v1/positionSide/dual")
    assert isinstance(data, dict)
    return bool(data.get("dualSidePosition", False))


# ── Bar-index helpers ──────────────────────────────────────────────────────────

def current_bar_open_ms(interval: str) -> int:
    """
    Return the open-time (ms) of the bar that is currently forming,
    snapped to the bar boundary.

    This matches TradingView's bar-index 0 = current (open) bar.
    """
    bar_ms = _INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    return (now_ms // bar_ms) * bar_ms


def bars_ago_to_ms(bars_ago: int, interval: str) -> int:
    """
    Convert "N bars ago" to the open-time (ms) of that bar.

    bars_ago = 0  →  current (still-forming) bar open
    bars_ago = 1  →  last fully closed bar open
    bars_ago = N  →  N bars back from the current bar open
    """
    bar_ms = _INTERVAL_MS[interval]
    return current_bar_open_ms(interval) - bars_ago * bar_ms


def ms_to_display(ms: int) -> str:
    """Format a millisecond timestamp as a human-readable UTC+8 string."""
    return (
        pd.Timestamp(ms, unit="ms", tz="UTC")
        .tz_convert(DISPLAY_TZ)
        .strftime("%Y-%m-%d %H:%M")
    )


# ── Trend-line math ─────────────────────────────────────────────────────────────

def line_price_at(t1_ms: int, p1: float, t2_ms: int, p2: float, now_ms: int) -> float:
    """
    Extrapolate the straight line through (t1_ms, p1) and (t2_ms, p2) to now_ms.
    Works for both interpolation and extrapolation.
    """
    if t2_ms == t1_ms:
        return p1
    slope = (p2 - p1) / (t2_ms - t1_ms)   # price per millisecond
    return p1 + slope * (now_ms - t1_ms)


# ── Close-order helper ─────────────────────────────────────────────────────────

def close_position(
    client: httpx.Client,
    secret: str,
    pos: dict,
    hedge: bool,
) -> dict:
    """Place a MARKET order that fully closes the given position."""
    symbol = pos["symbol"]
    amt    = float(pos["positionAmt"])
    side   = "SELL" if amt > 0 else "BUY"
    qty    = abs(amt)

    order: dict = {
        "symbol":   symbol,
        "side":     side,
        "type":     "MARKET",
        "quantity": qty,
    }

    if hedge:
        # In Hedge Mode positionSide is mandatory and reduceOnly is forbidden
        order["positionSide"] = pos.get("positionSide", "BOTH")
    else:
        order["reduceOnly"] = "true"

    return _signed_post(client, secret, "/fapi/v1/order", order)


# ── Interactive prompts ─────────────────────────────────────────────────────────

def prompt_select_position(positions: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("  Open Positions")
    print("=" * 60)
    for i, p in enumerate(positions, start=1):
        amt   = float(p["positionAmt"])
        side  = "LONG " if amt > 0 else "SHORT"
        entry = float(p["entryPrice"])
        mark  = float(p["markPrice"])
        pnl   = float(p["unRealizedProfit"])
        print(
            f"  [{i}] {p['symbol']:<12s}  {side}"
            f"  qty={amt:>+14g}"
            f"  entry={entry:>12.4f}"
            f"  mark={mark:>12.4f}"
            f"  uPnL={pnl:>+10.4f}"
        )
    print()

    while True:
        raw = input("Select position number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(positions):
                return positions[idx - 1]
        print(f"  Please enter a number between 1 and {len(positions)}.")


def prompt_interval() -> str:
    """Ask for candle interval once; default is 15m."""
    raw = input("\nCandle interval [15m]: ").strip() or "15m"
    if raw not in _INTERVAL_MS:
        print(f"  Unknown interval '{raw}', falling back to 15m.")
        return "15m"
    return raw


def _ask_price(prompt_text: str) -> float:
    while True:
        raw = input(prompt_text).strip()
        try:
            return float(raw)
        except ValueError:
            print("  Enter a valid number.")


def _ask_positive_int(prompt_text: str) -> int:
    while True:
        raw = input(prompt_text).strip()
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Enter a positive integer.")


def prompt_trendline_points(interval: str) -> tuple[tuple[int, float], tuple[int, float]]:
    """
    Collect two anchor points, each specified independently as
    (price, bars ago from current bar).

    Returns ((t1_ms, p1), (t2_ms, p2)).  The two points may be given in any
    order; the function always returns them sorted oldest-first (t1 < t2).
    """
    bar_ms = _INTERVAL_MS[interval]

    def ask_point(label: str) -> tuple[int, float]:
        print(f"\n--- Anchor point {label} ---")
        price    = _ask_price("  Price: ")
        bars_ago = _ask_positive_int("  Bars ago from current bar: ")
        ts_ms    = bars_ago_to_ms(bars_ago, interval)
        print(f"  → {ms_to_display(ts_ms)} ~ {ms_to_display(ts_ms + bar_ms)}  (UTC+8)")
        return ts_ms, price

    a = ask_point("1")
    b = ask_point("2")

    if a[0] == b[0]:
        print("[ERROR] Both points land on the same bar — cannot define a line.")
        sys.exit(1)

    # Ensure chronological order regardless of input sequence
    t1_ms, p1 = min(a, b, key=lambda x: x[0])
    t2_ms, p2 = max(a, b, key=lambda x: x[0])
    return (t1_ms, p1), (t2_ms, p2)


# ── Monitor loop ───────────────────────────────────────────────────────────────

def _refresh_mark_price(
    client: httpx.Client,
    secret: str,
    symbol: str,
    pos_side: str,
    hedge: bool,
) -> float | None:
    """
    Refresh mark price for the monitored position.
    Returns None if the position has been closed externally.
    """
    data = _signed_get(client, secret, "/fapi/v3/positionRisk", {"symbol": symbol})
    assert isinstance(data, list)

    for cp in data:
        if cp["symbol"] != symbol:
            continue
        if hedge and cp.get("positionSide") != pos_side:
            continue
        if float(cp["positionAmt"]) == 0.0:
            return None         # position already closed
        return float(cp["markPrice"])

    return None     # no matching row → position closed


def run_monitor(
    client: httpx.Client,
    secret: str,
    pos: dict,
    hedge: bool,
    t1_ms: int,
    p1: float,
    t2_ms: int,
    p2: float,
    poll_interval_s: int = 60,
) -> None:
    symbol   = pos["symbol"]
    amt      = float(pos["positionAmt"])
    is_long  = amt > 0
    pos_side = pos.get("positionSide", "BOTH")

    direction = "≤ (below or at)" if is_long else "≥ (above or at)"
    print(f"\n{'=' * 60}")
    print(f"  Monitoring  {symbol}  ({'LONG' if is_long else 'SHORT'})")
    print(f"  Trend line anchors (UTC+8):")
    print(f"    Point 1: {ms_to_display(t1_ms)}  →  {p1:.6g}")
    print(f"    Point 2: {ms_to_display(t2_ms)}  →  {p2:.6g}")
    print(f"  Stop fires when mark price {direction} line value")
    print(f"  Poll every {poll_interval_s}s  |  Ctrl-C to abort")
    print(f"{'=' * 60}\n")

    try:
        while True:
            now_ms  = int(time.time() * 1000)
            line_px = line_price_at(t1_ms, p1, t2_ms, p2, now_ms)

            mark_px = _refresh_mark_price(client, secret, symbol, pos_side, hedge)
            if mark_px is None:
                log.warning("Position %s no longer found — may already be closed.", symbol)
                break

            triggered = (is_long and mark_px <= line_px) or (not is_long and mark_px >= line_px)

            status = "*** TRIGGER ***" if triggered else ""
            log.info(
                "%s  mark=%.4f  line=%.4f  %s",
                symbol, mark_px, line_px, status,
            )

            if triggered:
                log.warning("Stop-loss line hit! Placing market close order …")
                result = close_position(client, secret, pos, hedge)
                log.info("Order result: orderId=%s  status=%s",
                         result.get("orderId"), result.get("status"))
                print("\nPosition closed. Exiting.")
                break

            time.sleep(poll_interval_s)

    except KeyboardInterrupt:
        print("\nMonitoring aborted by user.")
    except Exception as exc:
        # Unexpected non-network error — log and re-raise so caller knows
        log.error("Unexpected error in monitor loop: %s", exc, exc_info=True)
        raise


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    api_key, api_secret = _load_credentials()

    with _build_client(api_key) as client:
        log.info("Fetching open positions …")
        positions = fetch_open_positions(client, api_secret)

        if not positions:
            print("No open positions found.")
            return

        hedge = is_hedge_mode(client, api_secret)
        log.info("Account mode: %s", "Hedge" if hedge else "One-way")

        selected = prompt_select_position(positions)
        symbol   = selected["symbol"]

        print(f"\nSelected: {symbol}  "
              f"({'LONG' if float(selected['positionAmt']) > 0 else 'SHORT'})")
        print("Define the trend line: two anchor points, left→right (older → newer).\n")

        interval = prompt_interval()
        (t1_ms, p1), (t2_ms, p2) = prompt_trendline_points(interval)

        input("\nPress Enter to start monitoring (Ctrl-C to abort) … ")

        run_monitor(client, api_secret, selected, hedge, t1_ms, p1, t2_ms, p2)


if __name__ == "__main__":
    main()
