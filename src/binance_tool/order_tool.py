#!/usr/bin/env python3
"""
Futures Order Tool
==================
Interactive script for placing a full trade setup on USDⓈ-M Futures.

Safer staged flow
-----------------
1. Fetches top-N symbols ranked by 24 h quote volume → you pick one.
2. You choose direction: LONG or SHORT.
3. You enter the limit entry price.
4. You enter the stop-loss price.
5. You enter your maximum acceptable loss (quote asset).
   → Script back-calculates the required quantity.
6. You enter the take-profit price.
7. Confirms the full summary (qty, notional, R:R ratio).
8. Places the LIMIT entry order.
9. Waits for the entry to fill (up to 20 seconds).
   - If not filled, cancels the remaining entry.
   - If partially filled, proceeds with protection for the filled position.
10. Places protective exits as Algo orders:
     • STOP_MARKET          – stop-loss
     • TAKE_PROFIT_MARKET   – take-profit

Important
---------
This is NOT atomic. Binance entry orders and Algo exit orders are separate API
calls. This script reduces risk by:
- refusing to run if the symbol already has an open position
- canceling an unfilled/partially-filled entry after a timeout
- placing exit protection only after there is actual filled exposure
- attempting an emergency MARKET close if exit order placement fails

Usage
-----
    uv run python -m binance_tool.order_tool
    uv run python -m binance_tool.order_tool --top 30
    uv run python -m binance_tool.order_tool --testnet
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time

import httpx

from binance_tool.shared import (
    LIVE_BASE,
    TESTNET_BASE,
    build_client,
    decimal_places,
    floor_qty,
    is_hedge_mode,
    load_credentials,
    public_get,
    round_price,
    signed_delete,
    signed_get,
    signed_post,
    split_symbol,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Exchange info / symbol rules ───────────────────────────────────────────────

def fetch_symbol_rules(base_url: str, symbol: str) -> dict:
    """
    Return precision rules for *symbol* from /fapi/v1/exchangeInfo.

    Keys returned
    -------------
    tick_size      : str
    step_size      : str
    min_notional   : float
    min_qty        : float
    max_qty        : float
    order_types    : set[str]
    """
    data = public_get(base_url, "/fapi/v1/exchangeInfo")
    assert isinstance(data, dict)

    for sym in data.get("symbols", []):
        if sym["symbol"] != symbol:
            continue

        rules: dict = {
            "order_types": set(sym.get("orderTypes", [])),
        }
        for f in sym.get("filters", []):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                rules["tick_size"] = f["tickSize"]
            elif ft == "LOT_SIZE":
                rules["step_size"] = f["stepSize"]
                rules["min_qty"] = float(f["minQty"])
                rules["max_qty"] = float(f["maxQty"])
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
    Return the top-n USDC-margined USDⓈ-M symbols ranked by 24 h quote volume.
    Each element has: symbol, lastPrice, quoteVolume, priceChangePercent.
    """
    tickers = public_get(base_url, "/fapi/v1/ticker/24hr")
    assert isinstance(tickers, list)

    usdc = [t for t in tickers if t["symbol"].endswith("USDC")]
    usdc.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
    return usdc[:n]


# ── Account / position helpers ─────────────────────────────────────────────────

def fetch_available_balance(
    client: httpx.Client,
    secret: str,
    base_url: str,
    asset: str = "USDT",
) -> float:
    """
    Return the availableBalance for the given *asset* from GET /fapi/v3/account.
    """
    account = signed_get(client, secret, base_url, "/fapi/v3/account")
    assert isinstance(account, dict)
    for asset_info in account.get("assets", []):
        if asset_info.get("asset") == asset:
            return float(asset_info["availableBalance"])
    return 0.0


def fetch_max_leverage(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
) -> int:
    """
    Return the maximum allowed leverage for *symbol* at the first bracket.
    """
    data = signed_get(
        client,
        secret,
        base_url,
        "/fapi/v1/leverageBracket",
        {"symbol": symbol},
    )

    if isinstance(data, dict):
        data = [data]

    for entry in data:
        if entry.get("symbol") == symbol:
            brackets = entry.get("brackets", [])
            if brackets:
                return int(brackets[0]["initialLeverage"])

    return 125


def set_leverage(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    leverage: int,
) -> None:
    """Call POST /fapi/v1/leverage to set leverage for *symbol*."""
    result = signed_post(
        client,
        secret,
        base_url,
        "/fapi/v1/leverage",
        {"symbol": symbol, "leverage": leverage},
    )
    log.info(
        "Leverage set → %dx  (maxNotionalValue=%s)",
        result.get("leverage"),
        result.get("maxNotionalValue", "?"),
    )


def fetch_position_qty(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    hedge: bool,
    position_side: str,
) -> float:
    """
    Return the absolute open position quantity for *symbol*.

    In One-way mode, reads the BOTH position.
    In Hedge mode, reads the requested LONG/SHORT side.
    """
    data = signed_get(
        client,
        secret,
        base_url,
        "/fapi/v3/positionRisk",
        {"symbol": symbol},
    )
    if isinstance(data, dict):
        data = [data]

    target_side = position_side if hedge else "BOTH"

    for row in data:
        if row.get("symbol") != symbol:
            continue
        if row.get("positionSide") != target_side:
            continue
        try:
            return abs(float(row.get("positionAmt", "0")))
        except Exception:
            return 0.0

    return 0.0


def ensure_no_existing_position(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
) -> None:
    """
    Abort if *symbol* already has any non-zero position.
    This tool uses closePosition=true for protection, so mixing with an existing
    position is dangerous.
    """
    data = signed_get(
        client,
        secret,
        base_url,
        "/fapi/v3/positionRisk",
        {"symbol": symbol},
    )
    if isinstance(data, dict):
        data = [data]

    non_zero = []
    for row in data:
        if row.get("symbol") != symbol:
            continue
        try:
            amt = float(row.get("positionAmt", "0"))
        except Exception:
            amt = 0.0
        if abs(amt) > 0:
            non_zero.append(
                (
                    row.get("positionSide", "BOTH"),
                    amt,
                    row.get("entryPrice", "?"),
                )
            )

    if non_zero:
        print(f"\n[SAFETY BLOCK] {symbol} already has an open position:")
        for pos_side, amt, entry_price in non_zero:
            print(f"  • side={pos_side:<5} qty={amt:g} entry={entry_price}")
        print(
            "\nThis script uses closePosition=true for SL/TP algo orders.\n"
            "To avoid accidentally closing or mixing with an existing position, it refuses to continue."
        )
        sys.exit(1)


def prompt_leverage(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    notional: float,
) -> int:
    """
    Show margin balance, compute minimum leverage needed, let the user confirm.
    Returns the chosen leverage.
    """
    _, asset = split_symbol(symbol)
    if not asset:
        asset = "USDT"

    balance = fetch_available_balance(client, secret, base_url, asset)
    max_lev = fetch_max_leverage(client, secret, base_url, symbol)

    min_lev = math.ceil(notional / balance) if balance > 0 else max_lev
    min_lev = max(1, min(min_lev, max_lev))

    print(f"\n{'─' * 50}")
    print(f"  Futures {asset} available : {balance:.2f} {asset}")
    print(f"  Order notional           : {notional:.2f} {asset}")
    print(f"  Max leverage (exchange)  : {max_lev}x")
    print(f"  Minimum leverage needed  : {min_lev}x")
    print(f"{'─' * 50}")

    while True:
        raw = input(f"Leverage to use [{min_lev}x]: ").strip()
        if raw == "":
            chosen = min_lev
        else:
            try:
                chosen = int(raw)
            except ValueError:
                print("  Enter a whole number.")
                continue

        if chosen < min_lev:
            print(
                f"  {chosen}x is too low — margin would be "
                f"{notional / chosen:.2f} {asset} but only {balance:.2f} available. "
                f"Minimum is {min_lev}x."
            )
            continue
        if chosen > max_lev:
            print(f"  {chosen}x exceeds the exchange maximum of {max_lev}x.")
            continue

        margin_required = notional / chosen
        print(
            f"  Initial margin required: {margin_required:.2f} {asset}  "
            f"(balance={balance:.2f}, leverage={chosen}x)"
        )
        confirm = input("  Confirm? [Y/n]: ").strip().lower()
        if confirm != "n":
            return chosen


# ── Interactive prompts ────────────────────────────────────────────────────────

def prompt_symbol(top: list[dict]) -> str:
    """Display volume-ranked table and return the chosen symbol."""
    print("\n" + "=" * 72)
    print(
        f"  {'#':>3}  {'Symbol':<14}  {'Last Price':>14}  "
        f"{'24h Vol (Quote M)':>16}  {'24h Chg':>8}"
    )
    print("  " + "-" * 68)

    for i, t in enumerate(top, start=1):
        vol_m = float(t["quoteVolume"]) / 1_000_000
        chg = float(t["priceChangePercent"])
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
    dp = decimal_places(tick_size)
    while True:
        raw = input(prompt_text).strip()
        try:
            price = round_price(float(raw), tick_size)
            print(f"  → Rounded to tick: {price:.{dp}f}")
            return price
        except (ValueError, ArithmeticError):
            print("  Invalid number, try again.")


def _ask_positive_float(prompt_text: str, max_value: float = float('inf')) -> float:
    while True:
        raw = input(prompt_text).strip()
        try:
            val = float(raw)
            if val > 0:
                if val > max_value:
                    print(f"  Maximum value is {max_value}.")
                    continue
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
    base_asset, quote_asset = split_symbol(symbol)
    if not quote_asset:
        quote_asset = "USDT"

    is_long = direction == "LONG"
    tick = rules["tick_size"]
    step = rules["step_size"]
    dp_price = decimal_places(tick)
    dp_qty = decimal_places(step)

    print(f"\n  Current mark price ≈ {last_price:g}")
    print(
        f"  Tick size: {tick}  |  Step size: {step}  |  "
        f"Min notional: {rules['min_notional']} {quote_asset}"
    )

    # Entry
    direction_word = "buy" if is_long else "sell"
    entry = _ask_price(f"\nLimit {direction_word} price: ", tick)

    # Stop-loss
    while True:
        sl = _ask_price("Stop-loss price:  ", tick)
        if is_long and sl < entry:
            break
        if not is_long and sl > entry:
            break
        side_hint = "below entry" if is_long else "above entry"
        print(f"  Stop-loss must be {side_hint} for a {direction}.")

    # Max loss → quantity
    print("\nExpected max loss is calculated as:")
    print("  qty = max_loss / |entry − stop_loss|")

    max_loss = _ask_positive_float(f"Maximum acceptable loss ({quote_asset}, max:20): ", max_value=20)
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
            f"\n[ERROR] Notional {notional:.2f} {quote_asset} is below the exchange "
            f"minimum of {rules['min_notional']} {quote_asset}. "
            "Increase max_loss or choose a larger price gap."
        )
        sys.exit(1)

    print(f"\n  Quantity   : {qty:.{dp_qty}f} {base_asset}")
    print(f"  Notional   : {notional:.2f} {quote_asset}")
    print(
        f"  Actual risk: {actual_risk:.2f} {quote_asset}  "
        f"(max_loss={max_loss:.2f}, floored to step)"
    )

    # Take-profit
    while True:
        tp = _ask_price("Take-profit price: ", tick)
        if is_long and tp > entry:
            break
        if not is_long and tp < entry:
            break
        side_hint = "above entry" if is_long else "below entry"
        print(f"  Take-profit must be {side_hint} for a {direction}.")

    reward = qty * abs(tp - entry)
    rr = reward / actual_risk if actual_risk > 0 else math.inf

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "qty": qty,
        "notional": notional,
        "actual_risk": actual_risk,
        "reward": reward,
        "risk_reward": rr,
        "dp_price": dp_price,
        "dp_qty": dp_qty,
        "quote_asset": quote_asset,
        "base_asset": base_asset,
    }


def _confirm(symbol: str, direction: str, params: dict) -> bool:
    """Print full order summary and ask for confirmation."""
    dp_p = params["dp_price"]
    dp_q = params["dp_qty"]
    quote_asset = params["quote_asset"]

    print("\n" + "=" * 56)
    print("  ORDER SUMMARY")
    print("=" * 56)
    print(f"  Symbol     : {symbol}")
    print(f"  Direction  : {direction}")
    print(f"  Entry      : {params['entry']:.{dp_p}f}  (LIMIT)")
    print(f"  Stop-loss  : {params['sl']:.{dp_p}f}  (STOP_MARKET ALGO)")
    print(f"  Take-profit: {params['tp']:.{dp_p}f}  (TAKE_PROFIT_MARKET ALGO)")
    print(f"  Quantity   : {params['qty']:.{dp_q}f}")
    print(f"  Notional   : {params['notional']:.2f} {quote_asset}")
    print(f"  Risk       : {params['actual_risk']:.2f} {quote_asset}")
    print(f"  Reward     : {params['reward']:.2f} {quote_asset}")
    print(f"  R:R ratio  : 1 : {params['risk_reward']:.2f}")
    print("=" * 56)

    ans = input("\nConfirm and place orders? [y/N]: ").strip().lower()
    return ans == "y"


# ── Order helpers ──────────────────────────────────────────────────────────────

def _query_order(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    order_id: int,
) -> dict:
    data = signed_get(
        client,
        secret,
        base_url,
        "/fapi/v1/order",
        {"symbol": symbol, "orderId": order_id},
    )
    assert isinstance(data, dict)
    return data


def _cancel_order(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    order_id: int,
) -> dict:
    data = signed_delete(
        client,
        secret,
        base_url,
        "/fapi/v1/order",
        {"symbol": symbol, "orderId": order_id},
    )
    assert isinstance(data, dict)
    return data


def cancel_all_open_orders(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
) -> None:
    """
    Cancel every open regular order for *symbol* in one call.
    Silently ignores "no open orders" responses.
    """
    try:
        signed_delete(client, secret, base_url,
                      "/fapi/v1/allOpenOrders", {"symbol": symbol})
        log.info("All open regular orders for %s cancelled.", symbol)
    except httpx.HTTPStatusError as exc:
        if "code=-2011" in str(exc) or "code=-1013" in str(exc):
            log.info("No open regular orders to cancel for %s.", symbol)
        else:
            log.warning("cancel_all_open_orders failed: %s", exc)


def _cancel_algo_order(
    client: httpx.Client,
    secret: str,
    base_url: str,
    algo_id: int,
) -> dict:
    data = signed_delete(
        client,
        secret,
        base_url,
        "/fapi/v1/algoOrder",
        {"algoId": algo_id},
    )
    assert isinstance(data, dict)
    return data


def _wait_for_entry_exposure(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    order_id: int,
    timeout_s: float = 200.0,
    poll_s: float = 0.5,
) -> dict:
    """
    Wait for an entry order to either:
      - fill completely
      - partially fill
      - remain unfilled until timeout (then cancel)
      - get canceled/rejected/exired

    Returns the latest order object after any needed cancellation.
    """
    deadline = time.monotonic() + timeout_s
    last_status = None
    latest: dict | None = None

    while time.monotonic() < deadline:
        order = _query_order(client, secret, base_url, symbol, order_id)
        latest = order

        status = order.get("status")
        executed = order.get("executedQty", "0")

        if status != last_status:
            log.info("Entry status=%s  executedQty=%s", status, executed)
            last_status = status

        if status in {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
            return order

        time.sleep(poll_s)

    assert latest is not None

    executed_qty = float(latest.get("executedQty", "0"))
    status = latest.get("status", "?")

    log.warning(
        "Entry did not reach a terminal state within %.1fs (status=%s, executedQty=%s).",
        timeout_s,
        status,
        latest.get("executedQty", "0"),
    )

    try:
        _cancel_order(client, secret, base_url, symbol, order_id)
        log.info("Remaining entry quantity canceled.")
    except Exception as exc:
        log.warning("Cancel of remaining entry failed (may already be terminal): %s", exc)

    try:
        latest = _query_order(client, secret, base_url, symbol, order_id)
    except Exception:
        pass

    if executed_qty > 0:
        log.info("Entry had partial fill before cancellation.")
    return latest


def _place_exit_algo(
    client: httpx.Client,
    secret: str,
    base_url: str,
    *,
    symbol: str,
    side: str,
    hedge: bool,
    position_side: str,
    order_type: str,   # STOP_MARKET or TAKE_PROFIT_MARKET
    trigger_price: float,
    dp_price: int,
) -> dict:
    """
    Place a close-all protective algo order.
    """
    order: dict = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "triggerPrice": f"{trigger_price:.{dp_price}f}",
        "workingType": "MARK_PRICE",
        "closePosition": "true",
    }

    if hedge:
        order["positionSide"] = position_side

    result = signed_post(client, secret, base_url, "/fapi/v1/algoOrder", order)
    assert isinstance(result, dict)
    return result


# ── Order placement ────────────────────────────────────────────────────────────

def place_orders(
    client: httpx.Client,
    secret: str,
    base_url: str,
    symbol: str,
    direction: str,
    params: dict,
    hedge: bool,
) -> None:
    """
    Safer staged order flow:
      1) place LIMIT entry
      2) wait for fill / partial fill / timeout
      3) cancel any unfilled remainder
      4) place SL + TP as Algo orders protecting the actual position
      5) if Algo placement fails, try an emergency MARKET reduce-only close

    Ctrl-C at any stage triggers a cleanup sequence:
      - cancel the pending entry order (if not yet filled)
      - cancel any partially placed algo orders
    """
    is_long = direction == "LONG"
    pos_side = direction
    entry_side = "BUY" if is_long else "SELL"
    close_side = "SELL" if is_long else "BUY"

    dp_p = params["dp_price"]
    dp_q = params["dp_qty"]

    entry_id: int | None = None
    sl_algo_id: int | None = None
    tp_algo_id: int | None = None

    def _cleanup(reason: str) -> None:
        """Cancel every order we sent so far and exit."""
        print(f"\n[{reason}] Cancelling open orders for {symbol} …")

        # Cancel unfilled entry (regular order)
        cancel_all_open_orders(client, secret, base_url, symbol)

        # Cancel any algo orders we managed to place
        for algo_id in filter(None, [sl_algo_id, tp_algo_id]):
            try:
                _cancel_algo_order(client, secret, base_url, algo_id)
                log.info("Cancelled algo order algoId=%s", algo_id)
            except Exception as exc:
                log.warning("Failed to cancel algoId=%s: %s", algo_id, exc)

        print("Cleanup complete. Exiting.")

    try:
        # ── 1. Limit entry ─────────────────────────────────────────────────────
        entry_order: dict = {
            "symbol": symbol,
            "side": entry_side,
            "type": "LIMIT",
            "price": f"{params['entry']:.{dp_p}f}",
            "quantity": f"{params['qty']:.{dp_q}f}",
            "timeInForce": "GTC",
        }
        if hedge:
            entry_order["positionSide"] = pos_side

        log.info("Placing Entry (LIMIT) …")
        entry_result = signed_post(client, secret, base_url, "/fapi/v1/order", entry_order)
        entry_id = int(entry_result["orderId"])
        log.info("  ✓ entry orderId=%s  status=%s", entry_id, entry_result.get("status"))
        print("  (Press Ctrl-C at any time to cancel all orders and abort.)")

        # ── 2. Wait for exposure / cancel remainder ─────────────────────────────
        try:
            final_entry = _wait_for_entry_exposure(
                client, secret, base_url, symbol, entry_id,
                timeout_s=2000.0, poll_s=0.5,
            )
        except KeyboardInterrupt:
            _cleanup("Ctrl-C during fill wait")
            sys.exit(0)
        except Exception as exc:
            print(f"\n[ERROR] Could not manage entry order state safely: {exc}")
            return

        final_status = final_entry.get("status", "?")
        executed_qty = float(final_entry.get("executedQty", "0"))

        if executed_qty <= 0:
            print("\nNo position was opened. Entry did not fill.")
            print(f"Final entry status: {final_status}")
            return

        if final_status == "FILLED":
            print("\nEntry filled completely.")
        else:
            print("\nEntry filled partially; remaining quantity was canceled.")
            print(f"Filled quantity: {executed_qty:.{dp_q}f}")

        # Double-check actual open position on exchange
        current_pos_qty = fetch_position_qty(
            client, secret, base_url, symbol, hedge, pos_side,
        )

        if current_pos_qty <= 0:
            print("\n[WARN] Entry shows executed quantity, but no open position is visible now.")
            print("Protection orders were not placed.")
            return

        log.info("Open position confirmed on exchange: qty=%s", current_pos_qty)

        # ── 3. Place SL / TP as Algo orders ────────────────────────────────────
        log.info("Placing Stop-loss (STOP_MARKET algo) …")
        sl_result = _place_exit_algo(
            client, secret, base_url,
            symbol=symbol, side=close_side, hedge=hedge,
            position_side=pos_side, order_type="STOP_MARKET",
            trigger_price=params["sl"], dp_price=dp_p,
        )
        sl_algo_id = int(sl_result["algoId"])
        log.info("  ✓ stop-loss algoId=%s  algoStatus=%s",
                 sl_algo_id, sl_result.get("algoStatus", "?"))

        log.info("Placing Take-profit (TAKE_PROFIT_MARKET algo) …")
        tp_result = _place_exit_algo(
            client, secret, base_url,
            symbol=symbol, side=close_side, hedge=hedge,
            position_side=pos_side, order_type="TAKE_PROFIT_MARKET",
            trigger_price=params["tp"], dp_price=dp_p,
        )
        tp_algo_id = int(tp_result["algoId"])
        log.info("  ✓ take-profit algoId=%s  algoStatus=%s",
                 tp_algo_id, tp_result.get("algoStatus", "?"))

    except KeyboardInterrupt:
        _cleanup("Ctrl-C")
        sys.exit(0)

    except Exception as exc:
        log.error("Failed to place protective algo orders: %s", exc)

        # Roll back any algo we managed to place
        for algo_id in filter(None, [sl_algo_id, tp_algo_id]):
            try:
                _cancel_algo_order(client, secret, base_url, algo_id)
            except Exception:
                pass

        print("\n[CRITICAL] Protective exit order placement failed after entry exposure existed.")
        print("Attempting emergency market close …")

        latest_pos_qty = fetch_position_qty(
            client, secret, base_url, symbol, hedge, pos_side,
        )
        if latest_pos_qty <= 0:
            print("[SAFE EXIT] No open position detected anymore.")
            return

        emergency_order: dict = {
            "symbol": symbol,
            "side": close_side,
            "type": "MARKET",
            "quantity": f"{latest_pos_qty:.{dp_q}f}",
        }
        if hedge:
            emergency_order["positionSide"] = pos_side
        else:
            emergency_order["reduceOnly"] = "true"

        try:
            emergency = signed_post(client, secret, base_url, "/fapi/v1/order", emergency_order)
            print(f"[SAFE EXIT] Emergency close sent. orderId={emergency.get('orderId')}")
        except Exception as emergency_exc:
            print(f"[DANGER] Emergency close also failed: {emergency_exc}")
        return

    print("\nProtective orders placed successfully.")
    print(f"  • Stop-loss algoId   : {sl_algo_id}")
    print(f"  • Take-profit algoId : {tp_algo_id}")


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Futures order tool: volume picker → risk sizing → entry + protected exits"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top-volume symbols to display (default: 20)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use Binance USDⓈ-M Futures testnet (demo-fapi.binance.com)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_url = TESTNET_BASE if args.testnet else LIVE_BASE

    if args.testnet:
        log.info("*** TESTNET MODE — no real orders will be placed ***")

    api_key, api_secret = load_credentials()

    # 1. Volume ranking
    log.info("Fetching top-%d symbols by 24 h volume …", args.top)
    top = fetch_top_volume(base_url, args.top)
    symbol = prompt_symbol(top)

    # 2. Direction
    direction = prompt_direction()
    log.info("Selected: %s  %s", symbol, direction)

    # 3. Symbol rules
    log.info("Fetching exchange rules for %s …", symbol)
    rules = fetch_symbol_rules(base_url, symbol)

    last_price = float(next(t["lastPrice"] for t in top if t["symbol"] == symbol))

    # 4. Prices and risk input
    print(f"\n=== {symbol}  {direction} ===")
    params = prompt_trade_params(symbol, direction, rules, last_price)
    params["tick_size"] = rules["tick_size"]
    params["order_types"] = rules["order_types"]

    # 5. Confirm
    if not _confirm(symbol, direction, params):
        print("Cancelled.")
        return

    with build_client(api_key) as client:
        # 6. Account mode
        hedge = is_hedge_mode(client, api_secret, base_url)
        log.info("Account mode: %s", "Hedge" if hedge else "One-way")

        # 7. Safety: refuse to mix with an existing position
        ensure_no_existing_position(client, api_secret, base_url, symbol)

        # 8. Check balance and set leverage
        leverage = prompt_leverage(
            client,
            api_secret,
            base_url,
            symbol,
            params["notional"],
        )
        set_leverage(client, api_secret, base_url, symbol, leverage)

        # 9. Place protected order flow
        place_orders(
            client,
            api_secret,
            base_url,
            symbol,
            direction,
            params,
            hedge,
        )


if __name__ == "__main__":
    main()