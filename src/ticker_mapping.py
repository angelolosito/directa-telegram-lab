from __future__ import annotations

from typing import Any


def fallback_symbols(config: dict, symbol: str, instrument: dict[str, Any] | None = None) -> list[str]:
    symbols: list[str] = []
    source = instrument or {}
    for key in ("yahoo_fallbacks", "fallback_symbols", "provider_fallbacks"):
        value = source.get(key)
        if isinstance(value, str):
            symbols.append(value)
        elif isinstance(value, list):
            symbols.extend(str(item) for item in value if item)

    configured = (config.get("data", {}).get("ticker_fallbacks") or {}).get(symbol, [])
    if isinstance(configured, str):
        symbols.append(configured)
    elif isinstance(configured, list):
        symbols.extend(str(item) for item in configured if item)

    unique: list[str] = []
    for item in symbols:
        if item and item != symbol and item not in unique:
            unique.append(item)
    return unique
