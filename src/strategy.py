from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd


SignalAction = Literal["BUY", "WATCH", "SELL", "HOLD", "ERROR"]


@dataclass
class Signal:
    symbol: str
    name: str
    instrument_type: str
    action: SignalAction
    strategy: str
    date: str
    price: float | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reward_risk: float | None = None
    reason: str = ""
    qty: int = 0
    notional: float = 0.0
    estimated_round_trip_cost: float = 0.0
    meta: dict | None = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "instrument_type": self.instrument_type,
            "action": self.action,
            "strategy": self.strategy,
            "date": self.date,
            "price": self.price,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "reward_risk": self.reward_risk,
            "reason": self.reason,
            "qty": self.qty,
            "notional": self.notional,
            "estimated_round_trip_cost": self.estimated_round_trip_cost,
            "meta": self.meta or {},
        }


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _latest_context(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    clean = df.dropna(subset=["Close", "SMA20", "SMA50", "SMA200", "RSI14", "ATR14"])
    if len(clean) < 2:
        return None, None
    return clean.iloc[-1], clean.iloc[-2]


def analyze_buy_signals(
    instrument: dict,
    df: pd.DataFrame,
    strategy_config: dict,
    today: date,
) -> list[Signal]:
    symbol = instrument["symbol"]
    name = instrument.get("name", symbol)
    instrument_type = instrument.get("type", "unknown")
    latest, previous = _latest_context(df)

    if latest is None or previous is None:
        return [
            Signal(
                symbol=symbol,
                name=name,
                instrument_type=instrument_type,
                action="WATCH",
                strategy="data_quality",
                date=today.isoformat(),
                reason="Dati insufficienti per calcolare indicatori affidabili.",
            )
        ]

    close = float(latest["Close"])
    high = float(latest["High"])
    low = float(latest["Low"])
    sma20 = float(latest["SMA20"])
    sma50 = float(latest["SMA50"])
    sma200 = float(latest["SMA200"])
    rsi14 = float(latest["RSI14"])
    atr14 = float(latest["ATR14"])
    volume = _safe_float(latest.get("Volume")) or 0.0
    vol20 = _safe_float(latest.get("VOL20")) or 0.0
    high20_prev = _safe_float(latest.get("HIGH20_PREV"))

    signals: list[Signal] = []
    enabled = strategy_config.get("enabled", {})

    trend_up = close > sma200 and sma50 > sma200

    if enabled.get("trend_pullback", True):
        lookback = int(strategy_config.get("pullback_lookback_days", 8))
        recent = df.dropna(subset=["Low", "SMA20", "SMA50"]).tail(lookback)
        recent_low = float(recent["Low"].min()) if not recent.empty else low
        touched_pullback_zone = recent_low <= max(sma20 * 1.01, sma50 * 1.02)
        recovered_sma20 = close > sma20 and close > float(previous["Close"])
        rsi_ok = float(strategy_config.get("rsi_min_pullback", 40)) <= rsi14 <= float(
            strategy_config.get("rsi_max_pullback", 68)
        )

        if trend_up and touched_pullback_zone and recovered_sma20 and rsi_ok:
            raw_stop = min(recent_low, close - float(strategy_config.get("atr_stop_multiplier", 1.5)) * atr14)
            stop = round(raw_stop, 4)
            entry = round(close, 4)
            risk = entry - stop
            if risk > 0:
                target = round(entry + float(strategy_config.get("min_reward_risk", 2.0)) * risk, 4)
                rr = round((target - entry) / risk, 2)
                signals.append(
                    Signal(
                        symbol=symbol,
                        name=name,
                        instrument_type=instrument_type,
                        action="BUY",
                        strategy="trend_pullback",
                        date=today.isoformat(),
                        price=round(close, 4),
                        entry=entry,
                        stop=stop,
                        target=target,
                        reward_risk=rr,
                        reason=(
                            "Trend primario positivo, recente pullback verso medie brevi/intermedie, "
                            "recupero sopra SMA20 e RSI in area non eccessivamente tirata."
                        ),
                        meta={
                            "close": close,
                            "sma20": sma20,
                            "sma50": sma50,
                            "sma200": sma200,
                            "rsi14": rsi14,
                            "atr14": atr14,
                            "recent_low": recent_low,
                        },
                    )
                )

    if enabled.get("controlled_breakout", True):
        breakout_lookback = int(strategy_config.get("breakout_lookback_days", 20))
        # HIGH20_PREV is already a 20-day previous high; if not available, compute fallback.
        if high20_prev is None:
            high20_prev = float(df["High"].shift(1).rolling(breakout_lookback).max().iloc[-1])

        volume_ok = volume > 0 and vol20 > 0 and volume >= vol20 * float(
            strategy_config.get("volume_breakout_multiplier", 1.10)
        )
        breakout_ok = high20_prev is not None and close > high20_prev * 1.002
        rsi_ok = rsi14 <= float(strategy_config.get("rsi_max_breakout", 75))

        if trend_up and breakout_ok and volume_ok and rsi_ok:
            raw_stop = max(high20_prev - atr14, close - float(strategy_config.get("breakout_atr_stop_multiplier", 2.0)) * atr14)
            stop = round(raw_stop, 4)
            entry = round(close, 4)
            risk = entry - stop
            if risk > 0:
                target = round(entry + float(strategy_config.get("min_reward_risk", 2.0)) * risk, 4)
                rr = round((target - entry) / risk, 2)
                signals.append(
                    Signal(
                        symbol=symbol,
                        name=name,
                        instrument_type=instrument_type,
                        action="BUY",
                        strategy="controlled_breakout",
                        date=today.isoformat(),
                        price=round(close, 4),
                        entry=entry,
                        stop=stop,
                        target=target,
                        reward_risk=rr,
                        reason=(
                            "Breakout sopra massimo recente in trend primario positivo, "
                            "con volume superiore alla media e RSI sotto soglia estrema."
                        ),
                        meta={
                            "close": close,
                            "sma50": sma50,
                            "sma200": sma200,
                            "rsi14": rsi14,
                            "atr14": atr14,
                            "high20_prev": high20_prev,
                            "volume": volume,
                            "vol20": vol20,
                        },
                    )
                )

    if not signals:
        signals.append(
            Signal(
                symbol=symbol,
                name=name,
                instrument_type=instrument_type,
                action="HOLD",
                strategy="none",
                date=today.isoformat(),
                price=round(close, 4),
                reason="Nessun setup valido secondo le regole attive.",
                meta={"close": close, "sma20": sma20, "sma50": sma50, "sma200": sma200, "rsi14": rsi14},
            )
        )

    return signals
