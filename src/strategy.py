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
    score: float | None = None
    score_details: str = ""
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
            "score": self.score,
            "score_details": self.score_details,
            "meta": self.meta or {},
        }


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _score_watch_setup(base: float, *points: float) -> float:
    return round(_clamp(base + sum(points), 0.0, 100.0), 1)


def _latest_context(df: pd.DataFrame) -> tuple[pd.Series, pd.Series] | tuple[None, None]:
    clean = df.dropna(subset=["Close", "SMA20", "SMA50", "SMA200", "RSI14", "ATR14"])
    if len(clean) < 2:
        return None, None
    return clean.iloc[-1], clean.iloc[-2]


def score_signal(signal: Signal, strategy_config: dict) -> Signal:
    """Attach a 0-100 quality score to a sized BUY signal."""
    meta = signal.meta or {}
    close = _safe_float(meta.get("close")) or signal.price
    sma50 = _safe_float(meta.get("sma50"))
    sma200 = _safe_float(meta.get("sma200"))
    rsi14 = _safe_float(meta.get("rsi14"))
    volume = _safe_float(meta.get("volume"))
    vol20 = _safe_float(meta.get("vol20"))

    if close is None or signal.entry is None or signal.stop is None:
        signal.score = 0.0
        signal.score_details = "segnale incompleto"
        return signal

    base = float(strategy_config.get("score_base", 50.0))
    score = base

    strategy_bonus = 10.0 if signal.strategy == "controlled_breakout" else 8.0
    score += strategy_bonus

    trend_points = 0.0
    if sma200 and sma200 > 0:
        trend_points += _clamp(((close / sma200) - 1.0) * 100.0 * 0.45, -8.0, 10.0)
    if sma50 and sma200 and sma200 > 0:
        trend_points += _clamp(((sma50 / sma200) - 1.0) * 100.0 * 0.55, -5.0, 8.0)
    score += trend_points

    rr = signal.reward_risk or 0.0
    min_rr = float(strategy_config.get("min_reward_risk", 2.0))
    rr_points = _clamp((rr - min_rr) * 8.0, -6.0, 12.0)
    score += rr_points

    rsi_points = 0.0
    if rsi14 is not None:
        if 50 <= rsi14 <= 64:
            rsi_points = 10.0
        elif 45 <= rsi14 < 50 or 64 < rsi14 <= 70:
            rsi_points = 5.0
        elif 40 <= rsi14 < 45 or 70 < rsi14 <= 75:
            rsi_points = 0.0
        else:
            rsi_points = -8.0
    score += rsi_points

    unit_risk = signal.entry - signal.stop
    risk_pct = (unit_risk / signal.entry) * 100.0 if signal.entry > 0 else 0.0
    if 1.5 <= risk_pct <= 5.5:
        risk_points = 8.0
    elif 0.8 <= risk_pct < 1.5 or 5.5 < risk_pct <= 8.0:
        risk_points = 2.0
    else:
        risk_points = -8.0
    score += risk_points

    volume_points = 0.0
    if volume and vol20 and vol20 > 0:
        volume_ratio = volume / vol20
        volume_points = _clamp((volume_ratio - 1.0) * 10.0, -5.0, 8.0)
    elif signal.strategy == "controlled_breakout":
        volume_points = -6.0
    score += volume_points

    cost_pct = (
        (signal.estimated_round_trip_cost / signal.notional) * 100.0
        if signal.notional > 0
        else 0.0
    )
    if cost_pct == 0.0:
        cost_points = 0.0
    elif cost_pct <= 0.35:
        cost_points = 4.0
    elif cost_pct <= 0.65:
        cost_points = -2.0
    elif cost_pct <= 1.0:
        cost_points = -8.0
    else:
        cost_points = -15.0
    score += cost_points

    final_score = round(_clamp(score, 0.0, 100.0), 1)
    signal.score = final_score
    signal.score_details = (
        f"strategia {strategy_bonus:+.1f}, trend {trend_points:+.1f}, "
        f"R/R {rr_points:+.1f}, RSI {rsi_points:+.1f}, rischio {risk_points:+.1f}, "
        f"volumi {volume_points:+.1f}, costi {cost_points:+.1f}"
    )
    meta["score_breakdown"] = {
        "base": base,
        "strategy": strategy_bonus,
        "trend": round(trend_points, 2),
        "reward_risk": round(rr_points, 2),
        "rsi": round(rsi_points, 2),
        "risk": round(risk_points, 2),
        "volume": round(volume_points, 2),
        "cost": round(cost_points, 2),
        "risk_pct": round(risk_pct, 2),
        "cost_pct": round(cost_pct, 2),
    }
    signal.meta = meta
    return signal


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
    watch_signals: list[Signal] = []
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
                            "volume": volume,
                            "vol20": vol20,
                            "recent_low": recent_low,
                        },
                    )
                )
        elif trend_up and touched_pullback_zone and rsi_ok:
            distance_sma20_pct = ((close / sma20) - 1.0) * 100.0 if sma20 > 0 else None
            readiness = _score_watch_setup(
                50.0,
                10.0 if close > sma200 else 0.0,
                8.0 if touched_pullback_zone else 0.0,
                6.0 if rsi_ok else -6.0,
                6.0 if recovered_sma20 else -4.0,
            )
            if readiness >= float(strategy_config.get("setup_watch_min_score", 50.0)):
                missing = "attendere recupero sopra SMA20" if not recovered_sma20 else "attendere chiusura più convincente"
                watch_signals.append(
                    Signal(
                        symbol=symbol,
                        name=name,
                        instrument_type=instrument_type,
                        action="WATCH",
                        strategy="trend_pullback_setup",
                        date=today.isoformat(),
                        price=round(close, 4),
                        reason=(
                            "Setup pullback in formazione: trend primario positivo e prezzo tornato in zona utile; "
                            f"{missing}."
                        ),
                        score=readiness,
                        score_details=(
                            f"radar pullback, distanza SMA20 "
                            f"{distance_sma20_pct:.2f}%" if distance_sma20_pct is not None else "radar pullback"
                        ),
                        meta={
                            "close": close,
                            "sma20": sma20,
                            "sma50": sma50,
                            "sma200": sma200,
                            "rsi14": rsi14,
                            "atr14": atr14,
                            "volume": volume,
                            "vol20": vol20,
                            "recent_low": recent_low,
                            "readiness": readiness,
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
        elif trend_up and high20_prev is not None and high20_prev > 0:
            near_breakout_pct = float(strategy_config.get("near_breakout_pct", 1.5))
            distance_to_breakout_pct = ((high20_prev / close) - 1.0) * 100.0 if close > 0 else None
            near_breakout = close >= high20_prev * (1.0 - near_breakout_pct / 100.0) and close <= high20_prev * 1.002
            if near_breakout:
                volume_ratio = volume / vol20 if volume > 0 and vol20 > 0 else 0.0
                readiness = _score_watch_setup(
                    52.0,
                    10.0 if close > sma200 else 0.0,
                    8.0 if rsi_ok else -6.0,
                    _clamp((volume_ratio - 1.0) * 10.0, -4.0, 8.0),
                )
                if readiness >= float(strategy_config.get("setup_watch_min_score", 50.0)):
                    watch_signals.append(
                        Signal(
                            symbol=symbol,
                            name=name,
                            instrument_type=instrument_type,
                            action="WATCH",
                            strategy="near_breakout_setup",
                            date=today.isoformat(),
                            price=round(close, 4),
                            reason=(
                                "Setup breakout in avvicinamento: prezzo vicino al massimo recente; "
                                "attendere chiusura sopra livello con volumi adeguati."
                            ),
                            score=readiness,
                            score_details=(
                                f"radar breakout, distanza livello "
                                f"{distance_to_breakout_pct:.2f}%, volume x{volume_ratio:.2f}"
                                if distance_to_breakout_pct is not None
                                else f"radar breakout, volume x{volume_ratio:.2f}"
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
                                "volume_ratio": round(volume_ratio, 2),
                                "readiness": readiness,
                            },
                        )
                    )

    if not signals:
        if watch_signals:
            return sorted(watch_signals, key=lambda signal: signal.score or 0.0, reverse=True)[:2]
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
