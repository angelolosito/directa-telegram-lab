from __future__ import annotations

from typing import Any

import pandas as pd


def base_currency(config: dict) -> str:
    return str(config.get("project", {}).get("currency", "EUR") or "EUR").upper()


def instrument_currency(instrument: dict[str, Any], config: dict) -> str:
    return str(instrument.get("currency") or base_currency(config)).upper()


def configured_currency_pairs(watchlist: list[dict[str, Any]], config: dict) -> list[dict[str, str]]:
    currency_cfg = config.get("currency", {})
    symbol_by_currency = currency_cfg.get("fx_to_base_symbols") or {}
    base = base_currency(config)
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()

    for instrument in watchlist:
        currency = instrument_currency(instrument, config)
        if currency == base or currency in seen:
            continue
        symbol = symbol_by_currency.get(currency)
        if not symbol:
            continue
        pairs.append({"currency": currency, "symbol": symbol, "name": f"{currency}/{base}"})
        seen.add(currency)

    return pairs


def _latest_close(df: pd.DataFrame) -> float | None:
    if df.empty or "Close" not in df:
        return None
    clean = df.dropna(subset=["Close"])
    if clean.empty:
        return None
    latest = float(clean.iloc[-1]["Close"])
    return latest if latest > 0 else None


def latest_fx_rates(
    fx_data: dict[str, pd.DataFrame],
    pairs: list[dict[str, str]],
    config: dict,
) -> tuple[dict[str, float], list[str]]:
    base = base_currency(config)
    rates = {base: 1.0}
    errors: list[str] = []

    for pair in pairs:
        currency = pair["currency"]
        symbol = pair["symbol"]
        rate = _latest_close(fx_data.get(symbol, pd.DataFrame()))
        if rate is None:
            errors.append(f"{symbol}: cambio {currency}/{base} non disponibile.")
            continue
        rates[currency] = rate

    return rates, errors


def enrich_watchlist_with_fx(
    watchlist: list[dict[str, Any]],
    fx_rates: dict[str, float],
    config: dict,
) -> tuple[list[dict[str, Any]], list[str]]:
    base = base_currency(config)
    enriched: list[dict[str, Any]] = []
    errors: list[str] = []

    for instrument in watchlist:
        currency = instrument_currency(instrument, config)
        rate = fx_rates.get(currency)
        if rate is None:
            errors.append(
                f"{instrument.get('symbol', 'n/d')}: cambio {currency}/{base} mancante, strumento saltato."
            )
            continue
        item = dict(instrument)
        item["currency"] = currency
        item["base_currency"] = base
        item["fx_to_base"] = rate
        enriched.append(item)

    return enriched, errors


def fx_rate_from_meta(meta: dict | None) -> float:
    if not meta:
        return 1.0
    try:
        rate = float(meta.get("fx_to_base", 1.0) or 1.0)
    except Exception:
        return 1.0
    return rate if rate > 0 else 1.0


def price_to_base(price: float, meta: dict | None) -> float:
    return price * fx_rate_from_meta(meta)
