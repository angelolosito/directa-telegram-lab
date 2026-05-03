from __future__ import annotations

from pathlib import Path

from .manual_position_importer import MANUAL_POSITION_FIELDNAMES, load_manual_position_rows
from .strategy import Signal


def load_manual_positions(path: Path) -> list[Signal]:
    signals: list[Signal] = []
    for position in load_manual_position_rows(path):
        signals.append(
            Signal(
                symbol=position.symbol,
                name=position.name,
                instrument_type=position.instrument_type,
                action="BUY",
                strategy="manual_tracking",
                date=position.entry_date,
                price=position.last_price or position.entry_price,
                entry=position.entry_price,
                stop=position.stop,
                target=position.target,
                reward_risk=position.reward_risk,
                qty=position.qty,
                notional=round(position.notional, 2),
                reason="Posizione manuale importata da TradingView demo.",
                meta=position.meta(),
            )
        )

    return signals
