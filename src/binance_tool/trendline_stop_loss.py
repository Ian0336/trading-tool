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
4. Places a STOP_MARKET algo order on the exchange at the current trendline
   price, then refreshes it every 15 minutes so the order always tracks the
   moving trendline.  Any existing STOP_MARKET algo order for the symbol
   (e.g. one placed by order_tool.py) is replaced on each refresh cycle.

Key design rules
----------------
- LONG position  → stop fires when mark price falls *to or below* the line.
- SHORT position → stop fires when mark price rises *to or above* the line.
- "N bars ago" anchors to the OPEN of that bar (same convention as TradingView
  bar-index coordinates).  The default interval is 15 m.
- All time display uses UTC+8 (Asia/Taipei).

Usage
-----
    uv run python -m binance_tool.trendline_stop_loss
"""

from __future__ import annotations

import logging
import sys
import time

import httpx
import pandas as pd

from binance_tool.shared import (
    LIVE_BASE,
    build_client,
    decimal_places,
    fetch_open_positions,
    is_hedge_mode,
    load_credentials,
    public_get,
    round_price,
    signed_delete,
    signed_get,
    signed_post,
)

DISPLAY_TZ = "Asia/Taipei"   # UTC+8

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_URL = LIVE_BASE

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


# ── Exchange-info helper ──────────────────────────────────────────────────────

def fetch_price_tick(symbol: str) -> str:
    """Return the PRICE_FILTER tickSize string for *symbol*."""
    data = public_get(BASE_URL, "/fapi/v1/exchangeInfo")
    assert isinstance(data, dict)
    for sym in data.get("symbols", []):
        if sym["symbol"] != symbol:
            continue
        for f in sym.get("filters", []):
            if f.get("filterType") == "PRICE_FILTER":
                return f["tickSize"]
    raise ValueError(f"PRICE_FILTER not found for {symbol}")


# ── Algo-order helpers ──────────────────────────────────────────────────────────

def fetch_open_algo_stop_orders(
    client: httpx.Client, secret: str, symbol: str
) -> list[dict]:
    """
    Return all open STOP_MARKET algo orders for *symbol*.
    Includes orders placed by order_tool.py or this script.
    """
    data = signed_get(client, secret, BASE_URL,
                      "/fapi/v1/openAlgoOrders",
                      {"symbol": symbol}, retry=True)
    orders: list[dict] = []
    if isinstance(data, dict):
        orders = data.get("orders", [])
    elif isinstance(data, list):
        orders = data
    return [o for o in orders if o.get("type") in {"STOP_MARKET", "STOP"}]


def cancel_algo_order(
    client: httpx.Client, secret: str, algo_id: int
) -> None:
    """Cancel an algo order by *algoId*, ignoring "already terminal" errors."""
    try:
        signed_delete(client, secret, BASE_URL,
                      "/fapi/v1/algoOrder",
                      {"algoId": algo_id}, retry=True)
        log.info("Cancelled algo order algoId=%s", algo_id)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text if exc.response is not None else ""
        if "2011" in body or "2034" in body:
            log.info("Algo order algoId=%s already terminal (ignored).", algo_id)
        else:
            log.warning("Failed to cancel algoId=%s: %s", algo_id, exc)


def place_stop_algo_order(
    client: httpx.Client,
    secret: str,
    pos: dict,
    hedge: bool,
    trigger_price: float,
    tick_size: str,
) -> dict:
    """
    Place a STOP_MARKET algo order (closePosition=true) that protects *pos*.

    - LONG  → SELL stop triggers when mark price ≤ trigger_price
    - SHORT → BUY  stop triggers when mark price ≥ trigger_price
    """
    amt     = float(pos["positionAmt"])
    symbol  = pos["symbol"]
    side    = "SELL" if amt > 0 else "BUY"
    dp      = decimal_places(tick_size)
    price_s = f"{round_price(trigger_price, tick_size):.{dp}f}"

    order: dict = {
        "algoType":      "CONDITIONAL",
        "symbol":        symbol,
        "side":          side,
        "type":          "STOP_MARKET",
        "triggerPrice":  price_s,
        "workingType":   "MARK_PRICE",
        "closePosition": "true",
    }

    if hedge:
        order["positionSide"] = pos.get("positionSide", "BOTH")

    result = signed_post(client, secret, BASE_URL,
                         "/fapi/v1/algoOrder", order, retry=True)
    assert isinstance(result, dict)
    log.info(
        "Placed STOP_MARKET algo  algoId=%s  triggerPrice=%s  side=%s",
        result.get("algoId"), price_s, side,
    )
    return result


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
    data = signed_get(client, secret, BASE_URL,
                      "/fapi/v3/positionRisk",
                      {"symbol": symbol}, retry=True)
    assert isinstance(data, list)

    for cp in data:
        if cp["symbol"] != symbol:
            continue
        if hedge and cp.get("positionSide") != pos_side:
            continue
        if float(cp["positionAmt"]) == 0.0:
            return None
        return float(cp["markPrice"])

    return None


def _refresh_algo_stop(
    client: httpx.Client,
    secret: str,
    pos: dict,
    hedge: bool,
    tick_size: str,
    t1_ms: int,
    p1: float,
    t2_ms: int,
    p2: float,
    current_algo_id: int | None,
) -> int | None:
    """
    Cancel any existing STOP_MARKET algo orders for the symbol (including ones
    placed by order_tool.py), then place a fresh order at the current trendline
    price.  Returns the new algoId, or None if placement fails.
    """
    symbol  = pos["symbol"]
    now_ms  = int(time.time() * 1000)
    line_px = line_price_at(t1_ms, p1, t2_ms, p2, now_ms)

    existing = fetch_open_algo_stop_orders(client, secret, symbol)
    for o in existing:
        cancel_algo_order(client, secret, int(o["algoId"]))

    try:
        result = place_stop_algo_order(client, secret, pos, hedge,
                                       line_px, tick_size)
        return int(result["algoId"])
    except httpx.HTTPStatusError as exc:
        log.error("Failed to place algo stop order: %s", exc)
        return None


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
    update_interval_s: int = 900,   # 15 minutes
) -> None:
    """
    Monitor a position protected by a trendline stop-loss.

    Strategy
    --------
    1. On entry (and every *update_interval_s* seconds), cancel any open
       STOP_MARKET algo orders for the symbol and place a new one at the
       current trendline price.  This keeps the exchange-side stop in sync
       with the moving trendline.
    2. Poll every *poll_interval_s* seconds to check whether the position
       still exists.  When it disappears (closed by the algo stop or
       manually), the loop exits and any remaining algo order is cleaned up.
    """
    symbol   = pos["symbol"]
    amt      = float(pos["positionAmt"])
    is_long  = amt > 0
    pos_side = pos.get("positionSide", "BOTH")

    print(f"\n{'=' * 60}")
    print(f"  Monitoring  {symbol}  ({'LONG' if is_long else 'SHORT'})")
    print(f"  Trend line anchors (UTC+8):")
    print(f"    Point 1: {ms_to_display(t1_ms)}  →  {p1:.6g}")
    print(f"    Point 2: {ms_to_display(t2_ms)}  →  {p2:.6g}")
    print(f"  Stop order refreshed every {update_interval_s // 60} min  "
          f"|  position polled every {poll_interval_s}s  |  Ctrl-C to abort")
    print(f"{'=' * 60}\n")

    log.info("Fetching price tick size for %s …", symbol)
    tick_size = fetch_price_tick(symbol)

    current_algo_id: int | None = None
    last_update_t = 0.0   # force an immediate refresh on the first iteration

    try:
        while True:
            now = time.monotonic()

            # ── Refresh algo stop every update_interval_s ───────────────────
            if now - last_update_t >= update_interval_s:
                now_ms  = int(time.time() * 1000)
                line_px = line_price_at(t1_ms, p1, t2_ms, p2, now_ms)
                log.info(
                    "Refreshing STOP_MARKET algo  trendline=%.4f  (every %dm)",
                    line_px, update_interval_s // 60,
                )
                current_algo_id = _refresh_algo_stop(
                    client, secret, pos, hedge, tick_size,
                    t1_ms, p1, t2_ms, p2, current_algo_id,
                )
                last_update_t = now

            # ── Check whether position is still open ────────────────────────
            mark_px = _refresh_mark_price(client, secret, symbol, pos_side, hedge)
            if mark_px is None:
                log.info("Position %s no longer open — stop fired or closed manually.", symbol)
                print("\nPosition closed. Exiting.")
                break

            now_ms  = int(time.time() * 1000)
            line_px = line_price_at(t1_ms, p1, t2_ms, p2, now_ms)
            next_refresh_s = max(0, update_interval_s - (time.monotonic() - last_update_t))
            log.info(
                "%s  mark=%.4f  line=%.4f  algoId=%s  next_refresh_in=%ds",
                symbol, mark_px, line_px,
                current_algo_id if current_algo_id else "—",
                int(next_refresh_s),
            )

            time.sleep(poll_interval_s)

    except KeyboardInterrupt:
        print("\nMonitoring aborted by user.")
        if current_algo_id is not None:
            log.info("Cancelling algo stop order algoId=%s …", current_algo_id)
            cancel_algo_order(client, secret, current_algo_id)
    except Exception as exc:
        log.error("Unexpected error in monitor loop: %s", exc, exc_info=True)
        raise


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    api_key, api_secret = load_credentials()

    with build_client(api_key) as client:
        log.info("Fetching open positions …")
        positions = fetch_open_positions(client, api_secret, BASE_URL, retry=True)

        if not positions:
            print("No open positions found.")
            return

        hedge = is_hedge_mode(client, api_secret, BASE_URL, retry=True)
        log.info("Account mode: %s", "Hedge" if hedge else "One-way")

        selected = prompt_select_position(positions)
        symbol   = selected["symbol"]

        print(f"\nSelected: {symbol}  "
              f"({'LONG' if float(selected['positionAmt']) > 0 else 'SHORT'})")
        print("Define the trend line: two anchor points, left→right (older → newer).\n")

        interval = prompt_interval()
        (t1_ms, p1), (t2_ms, p2) = prompt_trendline_points(interval)

        input("\nPress Enter to start monitoring (Ctrl-C to abort) … ")

        run_monitor(
            client, api_secret, selected, hedge,
            t1_ms, p1, t2_ms, p2,
            poll_interval_s=60,
            update_interval_s=900,
        )


if __name__ == "__main__":
    main()
