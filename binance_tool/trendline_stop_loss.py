#!/usr/bin/env python3
"""
Trendline Stop-Loss Monitor
============================
1. Fetches all open USDⓈ-M Futures positions.
2. Lets you pick the position to protect.
3. Asks for two candle close-time + interval; fetches the close prices from
   Binance history to define the trend line.
4. Polls mark price every 60 s; fires a market close order when price crosses
   the extrapolated line.

Key design rules
----------------
- LONG position  → stop fires when mark price falls *to or below* the line.
- SHORT position → stop fires when mark price rises *to or above* the line.
- The line is extrapolated beyond the two anchor points using a constant slope
  (i.e. it extends infinitely in both directions at the same rate).

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

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from collect_data.client import iter_kline_pages
from utils.get_keys import get_secret

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


def _signed_get(client: httpx.Client, api_secret: str,
                path: str, extra: dict | None = None) -> list | dict:
    params: dict = {
        "timestamp":  int(time.time() * 1000),
        "recvWindow": 5000,
        **(extra or {}),
    }
    _sign(params, api_secret)
    resp = client.get(f"{BASE_URL}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


def _signed_post(client: httpx.Client, api_secret: str,
                 path: str, data: dict | None = None) -> dict:
    params: dict = {
        "timestamp":  int(time.time() * 1000),
        "recvWindow": 5000,
        **(data or {}),
    }
    _sign(params, api_secret)
    resp = client.post(f"{BASE_URL}{path}", data=params)
    resp.raise_for_status()
    return resp.json()


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


# ── Kline / close-price helpers ────────────────────────────────────────────────

def fetch_close_price(symbol: str, close_time_str: str, interval: str) -> float:
    """
    Return the close price of the candle whose close_time equals close_time_str.

    Parameters
    ----------
    close_time_str : "YYYY-MM-DD HH:MM"  (interpreted as UTC)
    interval       : Binance interval, e.g. "15m"
    """
    bar_ms = _INTERVAL_MS.get(interval.lower())
    if bar_ms is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. "
            f"Supported: {list(_INTERVAL_MS)}"
        )

    close_ts = pd.Timestamp(close_time_str, tz="UTC")
    close_ms = int(close_ts.value // 1_000_000)

    # The candle that closes at close_ms opens one interval earlier
    start_ms = close_ms - bar_ms
    end_ms   = close_ms + 1_000   # 1-second buffer to ensure we capture this bar

    rows: list[list] = []
    for page in iter_kline_pages(symbol, interval, start_ms, end_ms, market="futures"):
        rows.extend(page)

    if not rows:
        raise ValueError(
            f"No klines returned for {symbol} {interval} around {close_time_str}. "
            "Check that the timestamp and symbol are correct."
        )

    # Binance kline column 4 = close price
    return float(rows[-1][4])


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


def prompt_candle_point(label: str, symbol: str) -> tuple[int, float]:
    """
    Ask for a candle close-time and interval, auto-fetch the close price.
    Returns (timestamp_ms, close_price).
    """
    print(f"\n--- Anchor point {label} ---")
    print("  Timestamp: YYYY-MM-DD HH:MM  (UTC, e.g.  2026-02-22 14:15)")

    ts_str   = input("  Close time (UTC): ").strip()
    interval = input("  Candle interval  [15m]: ").strip() or "15m"

    print(f"  Fetching {symbol} {interval} candle at {ts_str} …")
    price = fetch_close_price(symbol, ts_str, interval)
    print(f"  → Close price: {price:.6g}")

    override = input("  Use this price? [Y/n]: ").strip().lower()
    if override == "n":
        price = float(input("  Enter price manually: ").strip())

    ts_ms = int(pd.Timestamp(ts_str, tz="UTC").value // 1_000_000)
    return ts_ms, price


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
    print(f"  Trend line anchors:")
    print(f"    Point 1: {pd.Timestamp(t1_ms, unit='ms', tz='UTC')}  →  {p1:.6g}")
    print(f"    Point 2: {pd.Timestamp(t2_ms, unit='ms', tz='UTC')}  →  {p2:.6g}")
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
        print("Define the trend line using two candle close prices.\n")

        t1_ms, p1 = prompt_candle_point("1", symbol)
        t2_ms, p2 = prompt_candle_point("2", symbol)

        if t1_ms == t2_ms:
            print("[ERROR] Both anchor points share the same timestamp — cannot define a line.")
            return

        input("\nPress Enter to start monitoring (Ctrl-C to abort) … ")

        run_monitor(client, api_secret, selected, hedge, t1_ms, p1, t2_ms, p2)


if __name__ == "__main__":
    main()
