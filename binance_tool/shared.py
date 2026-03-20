"""
Shared Binance Futures helpers
==============================
Common plumbing shared by order_tool, trendline_stop_loss, and
inspect_position: credentials, HTTP signing, precision rounding,
account queries, and retry logic.
"""

from __future__ import annotations

import hmac
import hashlib
import logging
import sys
import time
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import urlencode

import httpx

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.get_keys import get_secret  # noqa: E402

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LIVE_BASE = "https://fapi.binance.com"
TESTNET_BASE = "https://demo-fapi.binance.com"

# ── Network retry ─────────────────────────────────────────────────────────────
_RETRY_INITIAL_WAIT_S = 5
_RETRY_MAX_WAIT_S = 60
NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


def request_with_retry(fn):
    """
    Call *fn()* and retry with linear back-off on transient network errors.
    Gives up only on ``KeyboardInterrupt``.
    """
    wait = _RETRY_INITIAL_WAIT_S
    attempt = 0
    while True:
        try:
            return fn()
        except NETWORK_ERRORS as exc:
            attempt += 1
            log.warning(
                "Network error (attempt %d): %s — retrying in %ds …",
                attempt, exc, wait,
            )
            time.sleep(wait)
            wait = min(wait + 5, _RETRY_MAX_WAIT_S)
        except httpx.HTTPStatusError:
            raise


# ── Credentials ────────────────────────────────────────────────────────────────

def load_credentials() -> tuple[str, str]:
    """Load API key and secret from macOS Keychain."""
    api_key = get_secret("Binance_API_Key", "trading-tool")
    api_secret = get_secret("Binance_API_Secret", "trading-tool")

    missing: list[str] = []
    if not api_key:
        missing.append("service='Binance_API_Key'    account='trading-tool'")
    if not api_secret:
        missing.append("service='Binance_API_Secret' account='trading-tool'")

    if missing:
        print("[ERROR] Cannot load credentials from macOS Keychain:")
        for m in missing:
            print(f"  • {m}")
        print("\nTo store them, run:")
        print("  security add-generic-password -s Binance_API_Key    -a trading-tool -w <KEY>")
        print("  security add-generic-password -s Binance_API_Secret -a trading-tool -w <SECRET>")
        sys.exit(1)

    return api_key, api_secret


# ── HTTP plumbing ──────────────────────────────────────────────────────────────

def build_client(api_key: str) -> httpx.Client:
    """Return an ``httpx.Client`` with the API-key header pre-set."""
    return httpx.Client(headers={"X-MBX-APIKEY": api_key}, timeout=10)


def sign_params(params: dict, secret: str) -> dict:
    """Append HMAC-SHA256 ``signature`` to *params* in-place."""
    query = urlencode(params, doseq=True)
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def raise_for_status(resp: httpx.Response) -> None:
    """
    Like ``resp.raise_for_status()`` but includes the Binance JSON error
    body so the actual error code/message is visible.
    """
    if resp.is_error:
        try:
            body = resp.json()
            code = body.get("code", "?")
            msg = body.get("msg", resp.text)
        except Exception:
            code = "?"
            msg = resp.text

        if resp.status_code == 401:
            print(
                f"\n[401 Unauthorized]  Binance code={code}: {msg}\n"
                "  Possible causes:\n"
                "  • API key not enabled for USDⓈ-M Futures trading\n"
                "  • Wrong key/secret stored in Keychain\n"
                "  • Key is IP-restricted and this machine is not whitelisted\n"
                "  Check: Binance → API Management → Edit Restrictions"
            )
            sys.exit(1)

        raise httpx.HTTPStatusError(
            f"HTTP {resp.status_code}  Binance code={code}: {msg}",
            request=resp.request,
            response=resp,
        )


def public_get(
    base_url: str,
    path: str,
    params: dict | None = None,
) -> list | dict:
    """Unsigned GET for public endpoints (e.g. ``/fapi/v1/exchangeInfo``)."""
    with httpx.Client(timeout=10) as c:
        resp = c.get(f"{base_url}{path}", params=params or {})
        raise_for_status(resp)
        return resp.json()


def signed_get(
    client: httpx.Client,
    secret: str,
    base_url: str,
    path: str,
    extra: dict | None = None,
    *,
    retry: bool = False,
) -> list | dict:
    """Signed GET.  Set *retry=True* for automatic network-error retry."""
    def _do():
        params: dict = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
            **(extra or {}),
        }
        sign_params(params, secret)
        resp = client.get(f"{base_url}{path}", params=params)
        raise_for_status(resp)
        return resp.json()

    return request_with_retry(_do) if retry else _do()


def signed_post(
    client: httpx.Client,
    secret: str,
    base_url: str,
    path: str,
    data: dict,
    *,
    retry: bool = False,
) -> dict:
    """Signed POST.  Set *retry=True* for automatic network-error retry."""
    def _do():
        params: dict = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
            **data,
        }
        sign_params(params, secret)
        resp = client.post(f"{base_url}{path}", data=params)
        raise_for_status(resp)
        return resp.json()

    return request_with_retry(_do) if retry else _do()


def signed_delete(
    client: httpx.Client,
    secret: str,
    base_url: str,
    path: str,
    extra: dict | None = None,
    *,
    retry: bool = False,
) -> dict | list:
    """Signed DELETE.  Set *retry=True* for automatic network-error retry."""
    def _do():
        params: dict = {
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
            **(extra or {}),
        }
        sign_params(params, secret)
        resp = client.delete(f"{base_url}{path}", params=params)
        raise_for_status(resp)
        return resp.json()

    return request_with_retry(_do) if retry else _do()


# ── Precision helpers ──────────────────────────────────────────────────────────

def decimal_places(tick: str) -> int:
    """Return the number of decimal places implied by a tick/step string."""
    d = Decimal(tick)
    return max(0, -d.as_tuple().exponent)


def round_price(price: float, tick_size: str) -> float:
    """Round *price* to the nearest *tick_size*."""
    tick = Decimal(tick_size)
    result = (Decimal(str(price)) / tick).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP,
    ) * tick
    return float(result)


def floor_qty(qty: float, step_size: str) -> float:
    """Floor *qty* to the nearest *step_size* (always round down)."""
    step = Decimal(step_size)
    result = (Decimal(str(qty)) / step).quantize(
        Decimal("1"), rounding=ROUND_DOWN,
    ) * step
    return float(result)


# ── Symbol helpers ─────────────────────────────────────────────────────────────

def split_symbol(symbol: str) -> tuple[str, str]:
    """Return ``(base_asset, quote_asset)`` for common USDⓈ-M symbols."""
    for quote in ("USDC", "USDT", "BUSD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)], quote
    return symbol, ""


# ── Account / position helpers ─────────────────────────────────────────────────

def is_hedge_mode(
    client: httpx.Client,
    secret: str,
    base_url: str,
    *,
    retry: bool = False,
) -> bool:
    """Return ``True`` when account is in Hedge Mode (dual side position)."""
    data = signed_get(client, secret, base_url,
                      "/fapi/v1/positionSide/dual", retry=retry)
    assert isinstance(data, dict)
    return bool(data.get("dualSidePosition", False))


def fetch_open_positions(
    client: httpx.Client,
    secret: str,
    base_url: str,
    *,
    retry: bool = False,
) -> list[dict]:
    """Return every position with a non-zero ``positionAmt``."""
    all_pos = signed_get(client, secret, base_url,
                         "/fapi/v3/positionRisk", retry=retry)
    if isinstance(all_pos, dict):
        all_pos = [all_pos]
    return [p for p in all_pos if float(p.get("positionAmt", "0")) != 0.0]
