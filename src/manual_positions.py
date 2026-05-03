from __future__ import annotations

import csv
from pathlib import Path

from .strategy import Signal


MANUAL_POSITION_FIELDNAMES = [
    "symbol",
    "name",
    "instrument_type",
    "side",
    "qty",
    "avg_fill_price",
    "take_profit",
    "stop_loss",
    "last_price",
    "entry_date",
    "currency",
    "base_currency",
    "tradingview_symbol",
    "trade_value_display",
    "market_value_display",
    "unrealized_pnl_display",
    "unrealized_pnl_pct_display",
    "source",
]


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: {field}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Riga {row_number}: valore numerico non valido per {field}: {value}") from exc


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _reward_risk(entry: float, stop: float, target: float) -> float | None:
    risk = entry - stop
    reward = target - entry
    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 2)


def load_manual_positions(path: Path) -> list[Signal]:
    if not path.exists():
        raise FileNotFoundError(f"File posizioni manuali non trovato: {path}")

    signals: list[Signal] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: symbol")

            side = (row.get("side") or "Long").strip().lower()
            if side not in {"long", "buy"}:
                raise ValueError(f"Riga {row_number}: il tracking manuale supporta solo posizioni Long.")

            qty = _required_float(row, "qty", row_number)
            entry = _required_float(row, "avg_fill_price", row_number)
            target = _required_float(row, "take_profit", row_number)
            stop = _required_float(row, "stop_loss", row_number)
            last_price = _optional_float(row, "last_price")
            entry_date = (row.get("entry_date") or "").strip()
            if not entry_date:
                raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: entry_date")

            currency = (row.get("currency") or "EUR").strip() or "EUR"
            base_currency = (row.get("base_currency") or currency).strip() or currency
            meta = {
                "manual_tracking": True,
                "source": (row.get("source") or "manual").strip() or "manual",
                "currency": currency,
                "base_currency": base_currency,
                "tradingview_symbol": (row.get("tradingview_symbol") or "").strip(),
                "trade_value_display": (row.get("trade_value_display") or "").strip(),
                "market_value_display": (row.get("market_value_display") or "").strip(),
                "unrealized_pnl_display": (row.get("unrealized_pnl_display") or "").strip(),
                "unrealized_pnl_pct_display": (row.get("unrealized_pnl_pct_display") or "").strip(),
            }
            if currency == base_currency:
                meta["fx_to_base"] = 1.0
            if last_price is not None:
                meta["manual_last_price"] = last_price

            signals.append(
                Signal(
                    symbol=symbol,
                    name=(row.get("name") or symbol).strip() or symbol,
                    instrument_type=(row.get("instrument_type") or "stock").strip() or "stock",
                    action="BUY",
                    strategy="manual_tracking",
                    date=entry_date,
                    price=last_price or entry,
                    entry=entry,
                    stop=stop,
                    target=target,
                    reward_risk=_reward_risk(entry, stop, target),
                    qty=qty,
                    notional=round(qty * entry, 2),
                    reason="Posizione manuale importata da TradingView demo.",
                    meta=meta,
                )
            )

    return signals
