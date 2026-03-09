#!/usr/bin/env python3
"""
Futures Order Tool
==================
Interactive script for placing a full trade setup on USDⓈ-M Futures.

Flow
----
1. Fetches top-N symbols ranked by 24 h quote volume → you pick one.
2. You choose direction: LONG or SHORT.
3. You enter the limit entry price.
4. You enter the stop-loss price.
5. You enter your maximum acceptable loss (USDT).
   → Script back-calculates the required quantity.
6. You enter the take-profit price.
7. Confirms the full summary (qty, notional, R:R ratio).
8. Places three orders atomically:
     • LIMIT           – entry order
     • STOP_MARKET     – stop-loss  (reduceOnly / positionSide in Hedge Mode)
     • TAKE_PROFIT_MARKET – take-profit (same)

Price/quantity precision
------------------------
All prices are rounded to the symbol's tickSize.
All quantities are rounded down to the symbol's stepSize.
A minimum-notional check is performed before submission.

Usage
-----
    uv run python binance_tool/order_tool.py
    uv run python binance_tool/order_tool.py --top 30   # show top-30 by volume
    uv run python binance_tool/order_tool.py --testnet  # use demo-fapi.binance.com
"""

from __future__ import annotations

import argparse
import hmac
import hashlib
import logging
import math
import sys
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlencode

import httpx

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.get_keys import get_secret

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LIVE_BASE    = "https://fapi.binance.com"
TESTNET_BASE = "https://demo-fapi.binance.com"


# ── Credentials ────────────────────────────────────────────────────────────────

def _load_credentials() -> tuple[str, str]:
    """Load API key and secret from macOS Keychain."""
    api_key    = get_secret("Binance_API_Key",    "trading-tool")
    api_secret = get_secret("Binance_API_Secret", "trading-tool")

    missing: list[str] = []
    if not api_key:
        missing.append("service='Binance_API_Key'    account='trading-tool'")
    if not api_secret:
        missing.append("service='Binance_API_Secret' account='trading-tool'")
    print(api_key, api_secret)
    if missing:
        print("[ERROR] Cannot load credentials from macOS Keychain:")
        for m in missing:
            print(f"  • {m}")
        print("\nTo store them, run:")
        print("  security add-generic-password -s Binance_API_Key    -a trading-tool -w <KEY>")
        print("  security add-generic-password -s Binance_API_Secret -a trading-tool -w <SECRET>")
        sys.exit(1)

    return api_key, api_secret


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _build_client(api_key: str) -> httpx.Client:
    return httpx.Client(headers={"X-MBX-APIKEY": api_key}, timeout=10)


def _sign(params: dict, secret: str) -> dict:
    """Append HMAC-SHA256 signature in-place and return the dict."""
    query = urlencode(params, doseq=True)
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def _public_get(base_url: str, path: str, params: dict | None = None) -> list | dict:
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{base_url}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()


def _signed_get(client: httpx.Client, secret: str, base_url: str,
                path: str, extra: dict | None = None) -> list | dict:
    params: dict = {
        "timestamp":  int(time.time() * 1000),
        "recvWindow": 5000,
        **(extra or {}),
    }
    _sign(params, secret)
    resp = client.get(f"{base_url}{path}", params=params)
    resp.raise_for_status()
    return resp.json()


def _signed_post(client: httpx.Client, secret: str, base_url: str,
                 path: str, data: dict) -> dict:
    params: dict = {
        "timestamp":  int(time.time() * 1000),
        "recvWindow": 5000,
        **data,
    }
    _sign(params, secret)
    resp = client.post(f"{base_url}{path}", data=params)
    resp.raise_for_status()
    return resp.json()


# ── Exchange info / symbol rules ────────────────────────────────────────────────

def fetch_symbol_rules(base_url: str, symbol: str) -> dict:
    """
    Return precision rules for *symbol* from /fapi/v1/exchangeInfo.

    Keys returned
    -------------
    tick_size      : str  e.g. "0.10"
    step_size      : str  e.g. "0.001"
    min_notional   : float
    min_qty        : float
    max_qty        : float
    """
    data = _public_get(base_url, "/fapi/v1/exchangeInfo")
    assert isinstance(data, dict)

    for sym in data.get("symbols", []):
        if sym["symbol"] != symbol:
            continue

        rules: dict = {}
        for f in sym.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                rules["tick_size"] = f["tickSize"]
            elif ft == "LOT_SIZE":
                rules["step_size"] = f["stepSize"]
                rules["min_qty"]   = float(f["minQty"])
                rules["max_qty"]   = float(f["maxQty"])
            elif ft == "MIN_NOTIONAL":
                rules["min_notional"] = float(f.get("notional", f.get("minNotional", 0)))

        if "tick_size" not in rules:
            raise ValueError(f"PRICE_FILTER not found for {symbol}")
        if "step_size" not in rules:
            raise ValueError(f"LOT_SIZE not found for {symbol}")
        rules.setdefault("min_notional", 0.0)
        rules.setdefault("min_qty", 0.0)
        rules.setdefault("max_qty", float("inf"))
        return rules

    raise ValueError(f"Symbol '{symbol}' not found in exchangeInfo.")


# ── Volume ranking ─────────────────────────────────────────────────────────────

def fetch_top_volume(base_url: str, n: int = 20) -> list[dict]:
    """
    Return the top-n USDⓈ-M symbols ranked by 24 h quote volume (descending).
    Each element has: symbol, lastPrice, quoteVolume, priceChangePercent.
    """
    tickers = _public_get(base_url, "/fapi/v1/ticker/24hr")
    assert isinstance(tickers, list)

    usdt = [t for t in tickers if t["symbol"].endswith("USDT")]
    usdt.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return usdt[:n]


# ── Precision helpers ──────────────────────────────────────────────────────────

def _decimal_places(tick: str) -> int:
    """Return the number of decimal places implied by a tick/step string."""
    d = Decimal(tick)
    return max(0, -d.as_tuple().exponent)


def round_price(price: float, tick_size: str) -> float:
    """Round price to the nearest tick_size using banker-safe Decimal math."""
    tick = Decimal(tick_size)
    result = (Decimal(str(price)) / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick
    return float(result)


def floor_qty(qty: float, step_size: str) -> float:
    """Floor quantity to the nearest step_size (always round down)."""
    step = Decimal(step_size)
    result = (Decimal(str(qty)) / step).quantize(Decimal("1"), rounding=ROUND_DOWN) * step
    return float(result)


# ── Account mode ────────────────────────────────────────────────────────────────

def is_hedge_mode(client: httpx.Client, secret: str, base_url: str) -> bool:
    data = _signed_get(client, secret, base_url, "/fapi/v1/positionSide/dual")
    assert isinstance(data, dict)
    return bool(data.get("dualSidePosition", False))


# ── Interactive prompts ─────────────────────────────────────────────────────────

def prompt_symbol(top: list[dict]) -> str:
    """Display volume-ranked table and return the chosen symbol."""
    print("\n" + "=" * 72)
    print(f"  {'#':>3}  {'Symbol':<14}  {'Last Price':>14}  "
          f"{'24h Vol (USDT M)':>16}  {'24h Chg':>8}")
    print("  " + "-" * 68)

    for i, t in enumerate(top, start=1):
        vol_m   = float(t["quoteVolume"]) / 1_000_000
        chg     = float(t["priceChangePercent"])
        chg_str = f"{chg:+.2f}%"
        print(
            f"  {i:>3}  {t['symbol']:<14}  {float(t['lastPrice']):>14g}  "
            f"{vol_m:>15.1f}M  {chg_str:>8}"
        )

    print()
    while True:
        raw = input("Select symbol number: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(top):
                return top[idx - 1]["symbol"]
        print(f"  Enter a number between 1 and {len(top)}.")


def prompt_direction() -> str:
    """Return 'LONG' or 'SHORT'."""
    print("\nDirection:")
    print("  [1] LONG  (limit buy)")
    print("  [2] SHORT (limit sell)")
    while True:
        raw = input("Select [1/2]: ").strip()
        if raw == "1":
            return "LONG"
        if raw == "2":
            return "SHORT"
        print("  Please enter 1 or 2.")


def _ask_price(prompt_text: str, tick_size: str) -> float:
    """Read a price from stdin, round to tick_size, and return it."""
    dp = _decimal_places(tick_size)
    while True:
        raw = input(prompt_text).strip()
        try:
            price = round_price(float(raw), tick_size)
            print(f"  → Rounded to tick: {price:.{dp}f}")
            return price
        except (ValueError, ArithmeticError):
            print("  Invalid number, try again.")


def _ask_positive_float(prompt_text: str) -> float:
    while True:
        raw = input(prompt_text).strip()
        try:
            val = float(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Must be a positive number.")


def prompt_trade_params(
    symbol: str,
    direction: str,
    rules: dict,
    last_price: float,
) -> dict:
    """
    Interactively collect entry, sl, max_loss, tp.
    Returns a dict with keys:
        entry, sl, tp, qty, notional, risk_reward
    """
    is_long  = direction == "LONG"
    tick     = rules["tick_size"]
    step     = rules["step_size"]
    dp_price = _decimal_places(tick)
    dp_qty   = _decimal_places(step)

    print(f"\n  Current mark price ≈ {last_price:g}")
    print(f"  Tick size: {tick}  |  Step size: {step}  |  "
          f"Min notional: {rules['min_notional']} USDT")

    # ── Entry price ────────────────────────────────────────────────────────────
    direction_word = "buy" if is_long else "sell"
    entry = _ask_price(f"\nLimit {direction_word} price: ", tick)

    # ── Stop-loss price ────────────────────────────────────────────────────────
    while True:
        sl = _ask_price("Stop-loss price:  ", tick)
        if is_long and sl < entry:
            break
        if not is_long and sl > entry:
            break
        side_hint = "below entry" if is_long else "above entry"
        print(f"  Stop-loss must be {side_hint} for a {direction}.")

    # ── Max loss → quantity ────────────────────────────────────────────────────
    print("\nExpected max loss is calculated as:")
    print("  qty = max_loss / |entry − stop_loss|")

    max_loss = _ask_positive_float("Maximum acceptable loss (USDT): ")
    risk_per_unit = abs(entry - sl)
    raw_qty = max_loss / risk_per_unit
    qty = floor_qty(raw_qty, step)

    if qty < rules["min_qty"]:
        print(
            f"\n[WARN] Calculated qty {qty:.{dp_qty}f} is below minimum "
            f"{rules['min_qty']}. Adjusting to minimum."
        )
        qty = rules["min_qty"]

    if qty > rules["max_qty"]:
        print(
            f"\n[WARN] Calculated qty {qty:.{dp_qty}f} exceeds maximum "
            f"{rules['max_qty']}. Clamping to maximum."
        )
        qty = rules["max_qty"]

    notional = qty * entry
    actual_risk = qty * risk_per_unit

    if notional < rules["min_notional"]:
        print(
            f"\n[ERROR] Notional {notional:.2f} USDT is below the exchange "
            f"minimum of {rules['min_notional']} USDT. "
            "Increase max_loss or choose a larger price gap."
        )
        sys.exit(1)

    print(f"\n  Quantity  : {qty:.{dp_qty}f} {symbol.replace('USDT', '')}")
    print(f"  Notional  : {notional:.2f} USDT")
    print(f"  Actual risk: {actual_risk:.2f} USDT  "
          f"(max_loss={max_loss:.2f}, floored to step)")

    # ── Take-profit price ──────────────────────────────────────────────────────
    while True:
        tp = _ask_price("Take-profit price: ", tick)
        if is_long and tp > entry:
            break
        if not is_long and tp < entry:
            break
        side_hint = "above entry" if is_long else "below entry"
        print(f"  Take-profit must be {side_hint} for a {direction}.")

    # ── Risk : Reward ──────────────────────────────────────────────────────────
    reward = qty * abs(tp - entry)
    rr = reward / actual_risk if actual_risk > 0 else math.inf

    return {
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "qty":         qty,
        "notional":    notional,
        "actual_risk": actual_risk,
        "reward":      reward,
        "risk_reward": rr,
        "dp_price":    dp_price,
        "dp_qty":      dp_qty,
    }


def _confirm(symbol: str, direction: str, params: dict) -> bool:
    """Print full order summary and ask for confirmation."""
    dp_p = params["dp_price"]
    dp_q = params["dp_qty"]

    print("\n" + "=" * 56)
    print("  ORDER SUMMARY")
    print("=" * 56)
    print(f"  Symbol     : {symbol}")
    print(f"  Direction  : {direction}")
    print(f"  Entry      : {params['entry']:.{dp_p}f}  (LIMIT)")
    print(f"  Stop-loss  : {params['sl']:.{dp_p}f}  (STOP_MARKET)")
    print(f"  Take-profit: {params['tp']:.{dp_p}f}  (TAKE_PROFIT_MARKET)")
    print(f"  Quantity   : {params['qty']:.{dp_q}f}")
    print(f"  Notional   : {params['notional']:.2f} USDT")
    print(f"  Risk       : {params['actual_risk']:.2f} USDT")
    print(f"  Reward     : {params['reward']:.2f} USDT")
    print(f"  R:R ratio  : 1 : {params['risk_reward']:.2f}")
    print("=" * 56)

    ans = input("\nConfirm and place orders? [y/N]: ").strip().lower()
    return ans == "y"


# ── Order placement ────────────────────────────────────────────────────────────

def _build_base_order(symbol: str, qty: float, hedge: bool,
                      pos_side: str, dp_qty: int) -> dict:
    """Common fields shared by SL and TP orders."""
    base = {
        "symbol":      symbol,
        "quantity":    f"{qty:.{dp_qty}f}",
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    if hedge:
        base["positionSide"] = pos_side
    else:
        base["reduceOnly"] = "true"
    return base


def place_orders(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    direction: str,
    params: dict,
    hedge: bool,
) -> None:
    """Place the three orders: LIMIT entry, STOP_MARKET, TAKE_PROFIT_MARKET."""
    is_long  = direction == "LONG"
    pos_side = direction                            # "LONG" or "SHORT"
    entry_side = "BUY"  if is_long else "SELL"
    close_side = "SELL" if is_long else "BUY"

    dp_p = params["dp_price"]
    dp_q = params["dp_qty"]
    qty_str = f"{params['qty']:.{dp_q}f}"

    # ── 1. Limit entry ─────────────────────────────────────────────────────────
    entry_order: dict = {
        "symbol":        symbol,
        "side":          entry_side,
        "type":          "LIMIT",
        "price":         f"{params['entry']:.{dp_p}f}",
        "quantity":      qty_str,
        "timeInForce":   "GTC",
    }
    if hedge:
        entry_order["positionSide"] = pos_side

    # ── 2. Stop-loss ───────────────────────────────────────────────────────────
    sl_order = _build_base_order(symbol, params["qty"], hedge, pos_side, dp_q)
    sl_order.update({
        "side":       close_side,
        "type":       "STOP_MARKET",
        "stopPrice":  f"{params['sl']:.{dp_p}f}",
    })

    # ── 3. Take-profit ─────────────────────────────────────────────────────────
    tp_order = _build_base_order(symbol, params["qty"], hedge, pos_side, dp_q)
    tp_order.update({
        "side":       close_side,
        "type":       "TAKE_PROFIT_MARKET",
        "stopPrice":  f"{params['tp']:.{dp_p}f}",
    })

    print()
    for label, order in [("Entry (LIMIT)", entry_order),
                          ("Stop-loss (STOP_MARKET)", sl_order),
                          ("Take-profit (TAKE_PROFIT_MARKET)", tp_order)]:
        log.info("Placing %s …", label)
        result = _signed_post(client, secret, base_url, "/fapi/v1/order", order)
        order_id = result.get("orderId", "?")
        status   = result.get("status", "?")
        log.info("  ✓ orderId=%s  status=%s", order_id, status)

    print("\nAll three orders placed successfully.")


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Futures order tool: volume picker → risk sizing → entry/SL/TP"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top-volume symbols to display (default: 20)",
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Use Binance USDⓈ-M Futures testnet (demo-fapi.binance.com)",
    )
    return parser.parse_args()


def main() -> None:
    args     = _parse_args()
    base_url = TESTNET_BASE if args.testnet else LIVE_BASE

    if args.testnet:
        log.info("*** TESTNET MODE — no real orders will be placed ***")

    api_key, api_secret = _load_credentials()

    # ── 1. Volume ranking ──────────────────────────────────────────────────────
    log.info("Fetching top-%d symbols by 24 h volume …", args.top)
    top = fetch_top_volume(base_url, args.top)
    symbol = prompt_symbol(top)

    # ── 2. Direction ───────────────────────────────────────────────────────────
    direction = prompt_direction()
    log.info("Selected: %s  %s", symbol, direction)

    # ── 3. Symbol rules (precision) ────────────────────────────────────────────
    log.info("Fetching exchange rules for %s …", symbol)
    rules = fetch_symbol_rules(base_url, symbol)

    # Last price for reference display
    last_price = float(next(
        t["lastPrice"] for t in top if t["symbol"] == symbol
    ))

    # ── 4. Prices and risk input ───────────────────────────────────────────────
    print(f"\n=== {symbol}  {direction} ===")
    params = prompt_trade_params(symbol, direction, rules, last_price)

    # ── 5. Confirm ─────────────────────────────────────────────────────────────
    if not _confirm(symbol, direction, params):
        print("Cancelled.")
        return

    # ── 6. Account mode ────────────────────────────────────────────────────────
    with _build_client(api_key) as client:
        hedge = is_hedge_mode(client, api_secret, base_url)
        log.info("Account mode: %s", "Hedge" if hedge else "One-way")

        # ── 7. Place orders ────────────────────────────────────────────────────
        place_orders(
            client, api_secret, base_url,
            symbol, direction, params, hedge,
        )


if __name__ == "__main__":
    main()
