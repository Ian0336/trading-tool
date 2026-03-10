#!/usr/bin/env python3
from __future__ import annotations

import hmac
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import httpx

# 讓你沿用現有的 get_secret()
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.get_keys import get_secret  # noqa: E402


LIVE_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://demo-fapi.binance.com"


def load_credentials() -> tuple[str, str]:
    api_key = get_secret("Binance_API_Key", "trading-tool")
    api_secret = get_secret("Binance_API_Secret", "trading-tool")

    if not api_key or not api_secret:
        print("[ERROR] 無法從 macOS Keychain 讀取 Binance API Key / Secret")
        print("請確認你之前 order_tool.py 用的是同一組 key。")
        sys.exit(1)

    return api_key, api_secret


def sign_params(params: dict, secret: str) -> dict:
    query = urlencode(params, doseq=True)
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def raise_for_status(resp: httpx.Response) -> None:
    if resp.is_error:
        try:
            body = resp.json()
            code = body.get("code", "?")
            msg = body.get("msg", resp.text)
        except Exception:
            code = "?"
            msg = resp.text
        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}  Binance code={code}: {msg}",
            request=resp.request,
            response=resp,
        )


def signed_get(client: httpx.Client, secret: str, base_url: str, path: str, extra: dict | None = None):
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000,
        **(extra or {}),
    }
    sign_params(params, secret)
    resp = client.get(f"{base_url}{path}", params=params)
    raise_for_status(resp)
    return resp.json()


def split_symbol(symbol: str) -> tuple[str, str]:
    for quote in ("USDC", "USDT", "BUSD"):
        if symbol.endswith(quote):
            return symbol[:-len(quote)], quote
    return symbol, ""


def fetch_positions(client: httpx.Client, secret: str, base_url: str) -> list[dict]:
    data = signed_get(client, secret, base_url, "/fapi/v3/positionRisk")
    if isinstance(data, dict):
        data = [data]

    positions: list[dict] = []
    for row in data:
        try:
            amt = float(row.get("positionAmt", "0"))
        except Exception:
            amt = 0.0

        if abs(amt) > 0:
            positions.append(row)

    return positions


def fetch_open_orders(client: httpx.Client, secret: str, base_url: str, symbol: str) -> list[dict]:
    data = signed_get(client, secret, base_url, "/fapi/v1/openOrders", {"symbol": symbol})
    return data if isinstance(data, list) else [data]


def fetch_open_algo_orders(client: httpx.Client, secret: str, base_url: str, symbol: str) -> list[dict]:
    data = signed_get(client, secret, base_url, "/fapi/v1/openAlgoOrders", {"symbol": symbol})
    return data if isinstance(data, list) else [data]


def fetch_adl_quantile(client: httpx.Client, secret: str, base_url: str, symbol: str) -> list[dict]:
    data = signed_get(client, secret, base_url, "/fapi/v1/adlQuantile", {"symbol": symbol})
    return data if isinstance(data, list) else [data]


def fmt_float(value: str | float | int, digits: int = 4) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return str(value)


def choose_position(positions: list[dict]) -> dict:
    print("\n目前有倉位的部位：")
    print("=" * 96)
    print(f"{'#':>3}  {'Symbol':<12} {'Side':<6} {'Qty':>12} {'Entry':>12} {'Mark':>12} {'PnL':>14} {'Liq':>12}")
    print("-" * 96)

    for i, p in enumerate(positions, start=1):
        symbol = p.get("symbol", "?")
        side = p.get("positionSide", "BOTH")
        qty = float(p.get("positionAmt", "0"))
        entry = p.get("entryPrice", "0")
        mark = p.get("markPrice", "0")
        pnl = p.get("unRealizedProfit", "0")
        liq = p.get("liquidationPrice", "0")

        print(
            f"{i:>3}  {symbol:<12} {side:<6} "
            f"{fmt_float(qty, 3):>12} {fmt_float(entry, 2):>12} "
            f"{fmt_float(mark, 2):>12} {fmt_float(pnl, 4):>14} {fmt_float(liq, 2):>12}"
        )

    print()

    while True:
        raw = input(f"選擇要查看的倉位 [1-{len(positions)}]: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(positions):
                return positions[idx - 1]
        print("請輸入有效編號。")


def print_position_detail(pos: dict, open_orders: list[dict], algo_orders: list[dict], adl_rows: list[dict]) -> None:
    symbol = pos.get("symbol", "?")
    base_asset, quote_asset = split_symbol(symbol)
    if not quote_asset:
        quote_asset = pos.get("marginAsset", "QUOTE")

    print("\n" + "=" * 88)
    print(f"倉位詳細資訊：{symbol}")
    print("=" * 88)
    print(f"base asset          : {base_asset}")
    print(f"quote asset         : {quote_asset}")
    print(f"position side       : {pos.get('positionSide')}")
    print(f"position qty        : {pos.get('positionAmt')}")
    print(f"entry price         : {pos.get('entryPrice')}")
    print(f"break-even price    : {pos.get('breakEvenPrice')}")
    print(f"mark price          : {pos.get('markPrice')}")
    print(f"unrealized pnl      : {pos.get('unRealizedProfit')} {quote_asset}")
    print(f"liquidation price   : {pos.get('liquidationPrice')}")
    print(f"notional            : {pos.get('notional')} {quote_asset}")
    print(f"leverage            : {pos.get('leverage')}")
    print(f"margin type         : {pos.get('marginType')}")
    print(f"margin asset        : {pos.get('marginAsset')}")
    print(f"isolated margin     : {pos.get('isolatedMargin')}")
    print(f"isolated wallet     : {pos.get('isolatedWallet')}")
    print(f"initial margin      : {pos.get('initialMargin')}")
    print(f"maint margin        : {pos.get('maintMargin')}")
    print(f"position init margin: {pos.get('positionInitialMargin')}")
    print(f"open order margin   : {pos.get('openOrderInitialMargin')}")
    print(f"adl                 : {pos.get('adl')}")
    print(f"bid notional        : {pos.get('bidNotional')}")
    print(f"ask notional        : {pos.get('askNotional')}")
    print(f"update time         : {pos.get('updateTime')}")

    print("\nADL quantile:")
    if adl_rows:
        print(json.dumps(adl_rows, indent=2, ensure_ascii=False))
    else:
        print("  (無資料)")

    print("\n一般 Open Orders (/fapi/v1/openOrders):")
    if not open_orders:
        print("  無")
    else:
        for i, o in enumerate(open_orders, start=1):
            print(f"\n  [{i}] orderId={o.get('orderId')}  status={o.get('status')}  type={o.get('type')}")
            print(f"      side={o.get('side')}  positionSide={o.get('positionSide')}  reduceOnly={o.get('reduceOnly')}")
            print(f"      origQty={o.get('origQty')}  executedQty={o.get('executedQty')}")
            print(f"      price={o.get('price')}  stopPrice={o.get('stopPrice')}")
            print(f"      timeInForce={o.get('timeInForce')}  workingType={o.get('workingType')}")

    print("\nAlgo Open Orders (/fapi/v1/openAlgoOrders):")
    if not algo_orders:
        print("  無")
    else:
        for i, a in enumerate(algo_orders, start=1):
            print(f"\n  [{i}] algoId={a.get('algoId')}  algoStatus={a.get('algoStatus')}  orderType={a.get('orderType')}")
            print(f"      side={a.get('side')}  positionSide={a.get('positionSide')}  closePosition={a.get('closePosition')}")
            print(f"      triggerPrice={a.get('triggerPrice')}  workingType={a.get('workingType')}")
            print(f"      quantity={a.get('quantity')}  price={a.get('price')}")
            print(f"      reduceOnly={a.get('reduceOnly')}  priceProtect={a.get('priceProtect')}")
            print(f"      actualOrderId={a.get('actualOrderId')}  actualPrice={a.get('actualPrice')}")
            print(f"      createTime={a.get('createTime')}  updateTime={a.get('updateTime')}")

    print("\n完整 positionRisk 原始 JSON：")
    print(json.dumps(pos, indent=2, ensure_ascii=False))


def main() -> None:
    use_testnet = "--testnet" in sys.argv
    base_url = TESTNET_BASE if use_testnet else LIVE_BASE

    api_key, api_secret = load_credentials()

    with httpx.Client(headers={"X-MBX-APIKEY": api_key}, timeout=10) as client:
        positions = fetch_positions(client, api_secret, base_url)

        if not positions:
            print("目前沒有非 0 倉位。")
            return

        selected = choose_position(positions)
        symbol = selected["symbol"]

        open_orders = fetch_open_orders(client, api_secret, base_url, symbol)
        algo_orders = fetch_open_algo_orders(client, api_secret, base_url, symbol)
        adl_rows = fetch_adl_quantile(client, api_secret, base_url, symbol)

        print_position_detail(selected, open_orders, algo_orders, adl_rows)


if __name__ == "__main__":
    main()