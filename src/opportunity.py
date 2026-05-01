from __future__ import annotations

from typing import Any

import pandas as pd

from .strategy import Signal


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _regime_value(market_regime: Any, key: str, default=None):
    if isinstance(market_regime, dict):
        return market_regime.get(key, default)
    return getattr(market_regime, key, default)


def _add_check(checks: list[dict], name: str, status: str, points: float, note: str) -> None:
    checks.append(
        {
            "name": name,
            "status": status,
            "points": round(points, 2),
            "note": note,
        }
    )


def _append_sentence(text: str, sentence: str) -> str:
    if not text:
        return sentence
    return f"{text.rstrip()} {sentence}"


def _grade(score: float) -> str:
    if score >= 82:
        return "A"
    if score >= 72:
        return "B"
    if score >= 62:
        return "C"
    return "D"


def _cost_pct(signal: Signal) -> float | None:
    if signal.notional <= 0:
        return None
    return (signal.estimated_round_trip_cost / signal.notional) * 100.0


def review_opportunity(signal: Signal, market_regime: Any, config: dict) -> Signal:
    opportunity_cfg = config.get("opportunity", {})
    if not opportunity_cfg.get("enabled", True) or signal.action != "BUY":
        return signal

    meta = signal.meta or {}
    checks: list[dict] = []
    hard_blocks: list[str] = []
    adjustment = 0.0
    technical_score = float(signal.score or 0.0)

    market_state = str(_regime_value(market_regime, "state", "unknown"))
    new_positions_allowed = bool(_regime_value(market_regime, "new_positions_allowed", True))
    active_min_signal_score = float(_regime_value(market_regime, "active_min_signal_score", technical_score))

    if not new_positions_allowed:
        adjustment -= 12.0
        hard_blocks.append("regime mercato difensivo")
        _add_check(checks, "mercato", "fail", -12.0, "regime mercato non consente nuovi ingressi")
    elif market_state == "risk_on":
        adjustment += 4.0
        _add_check(checks, "mercato", "pass", 4.0, "contesto generale favorevole")
    elif market_state == "neutral":
        adjustment -= 3.0
        _add_check(checks, "mercato", "caution", -3.0, "contesto misto, serve più qualità")
    elif market_state == "risk_off":
        adjustment -= 8.0
        _add_check(checks, "mercato", "caution", -8.0, "contesto fragile")
    else:
        _add_check(checks, "mercato", "caution", 0.0, "contesto non leggibile")

    close = _safe_float(meta.get("close")) or signal.price
    sma20 = _safe_float(meta.get("sma20"))
    sma50 = _safe_float(meta.get("sma50"))
    sma200 = _safe_float(meta.get("sma200"))
    atr14 = _safe_float(meta.get("atr14"))
    rsi14 = _safe_float(meta.get("rsi14"))
    high20_prev = _safe_float(meta.get("high20_prev"))
    volume = _safe_float(meta.get("volume"))
    vol20 = _safe_float(meta.get("vol20"))

    if close and sma50 and sma200 and close > sma50 > sma200:
        adjustment += 6.0
        _add_check(checks, "trend", "pass", 6.0, "prezzo sopra SMA50 e SMA200 ordinata")
    elif close and sma50 and sma200 and close > sma200 and sma50 > sma200:
        adjustment += 3.0
        _add_check(checks, "trend", "pass", 3.0, "trend primario positivo")
    else:
        adjustment -= 8.0
        hard_blocks.append("trend non abbastanza pulito")
        _add_check(checks, "trend", "fail", -8.0, "trend non confermato")

    if signal.strategy == "trend_pullback" and close and sma20 and sma20 > 0:
        distance_sma20_pct = ((close / sma20) - 1.0) * 100.0
        meta["distance_sma20_pct"] = round(distance_sma20_pct, 2)
        ideal = float(opportunity_cfg.get("ideal_pullback_distance_pct", 2.5))
        max_distance = float(opportunity_cfg.get("max_pullback_distance_pct", 4.5))
        if 0 <= distance_sma20_pct <= ideal:
            adjustment += 10.0
            _add_check(checks, "timing", "pass", 10.0, "rimbalzo vicino alla media breve")
        elif 0 <= distance_sma20_pct <= max_distance:
            adjustment += 4.0
            _add_check(checks, "timing", "caution", 4.0, "rimbalzo valido ma già un po' esteso")
        else:
            adjustment -= 10.0
            hard_blocks.append("prezzo troppo distante dalla SMA20")
            _add_check(checks, "timing", "fail", -10.0, "pullback già troppo rincorso")
    elif signal.strategy == "controlled_breakout" and close and high20_prev and atr14 and atr14 > 0:
        extension_atr = (close - high20_prev) / atr14
        meta["breakout_extension_atr"] = round(extension_atr, 2)
        ideal = float(opportunity_cfg.get("ideal_breakout_extension_atr", 0.8))
        max_extension = float(opportunity_cfg.get("max_breakout_extension_atr", 1.4))
        if 0 <= extension_atr <= ideal:
            adjustment += 10.0
            _add_check(checks, "timing", "pass", 10.0, "breakout fresco, non ancora tirato")
        elif 0 <= extension_atr <= max_extension:
            adjustment += 4.0
            _add_check(checks, "timing", "caution", 4.0, "breakout valido ma meno fresco")
        else:
            adjustment -= 10.0
            hard_blocks.append("breakout troppo esteso rispetto ad ATR")
            _add_check(checks, "timing", "fail", -10.0, "rischio di inseguire il prezzo")

    if rsi14 is not None:
        if 45 <= rsi14 <= 64:
            adjustment += 4.0
            _add_check(checks, "momentum", "pass", 4.0, "RSI in zona costruttiva")
        elif 40 <= rsi14 <= 72:
            _add_check(checks, "momentum", "caution", 0.0, "RSI accettabile ma non ideale")
        else:
            adjustment -= 5.0
            hard_blocks.append("momentum poco equilibrato")
            _add_check(checks, "momentum", "fail", -5.0, "RSI fuori zona utile")

    if volume and vol20 and vol20 > 0:
        volume_ratio = volume / vol20
        meta["volume_ratio"] = round(volume_ratio, 2)
        if volume_ratio >= 1.25:
            adjustment += 4.0
            _add_check(checks, "partecipazione", "pass", 4.0, "volume sopra media")
        elif volume_ratio >= 0.85:
            _add_check(checks, "partecipazione", "caution", 0.0, "volume nella norma")
        else:
            adjustment -= 3.0
            _add_check(checks, "partecipazione", "caution", -3.0, "volume debole")

    if signal.entry is not None and signal.stop is not None and signal.entry > 0:
        risk_pct = ((signal.entry - signal.stop) / signal.entry) * 100.0
        meta["risk_pct"] = round(risk_pct, 2)
        min_risk_pct = float(opportunity_cfg.get("min_risk_pct", 0.8))
        max_risk_pct = float(opportunity_cfg.get("max_risk_pct", 8.0))
        if 1.5 <= risk_pct <= 5.5:
            adjustment += 5.0
            _add_check(checks, "rischio", "pass", 5.0, "stop abbastanza vicino ma non casuale")
        elif min_risk_pct <= risk_pct <= max_risk_pct:
            adjustment += 1.0
            _add_check(checks, "rischio", "caution", 1.0, "rischio accettabile ma non ideale")
        else:
            adjustment -= 10.0
            hard_blocks.append("distanza stop non efficiente")
            _add_check(checks, "rischio", "fail", -10.0, "stop troppo stretto o troppo largo")

    cost_pct = _cost_pct(signal)
    if cost_pct is not None:
        meta["cost_pct"] = round(cost_pct, 2)
        high_quality_cost_pct = float(opportunity_cfg.get("high_quality_cost_pct", 1.0))
        max_cost_pct = float(opportunity_cfg.get("max_cost_pct", 5.0))
        if cost_pct <= high_quality_cost_pct:
            adjustment += 4.0
            _add_check(checks, "costi", "pass", 4.0, "costi compatibili con il trade")
        elif cost_pct <= max_cost_pct:
            _add_check(checks, "costi", "caution", 0.0, "costi elevati ma sotto limite")
        else:
            adjustment -= 8.0
            hard_blocks.append("costi troppo pesanti")
            _add_check(checks, "costi", "fail", -8.0, "commissioni incidono troppo sul controvalore")

    opportunity_score = round(_clamp(technical_score + adjustment, 0.0, 100.0), 1)
    decision_threshold = max(float(opportunity_cfg.get("min_decision_score", 62.0)), active_min_signal_score)
    grade = _grade(opportunity_score)

    if hard_blocks:
        decision = "NO_GO"
        status_text = "Momento non propizio"
    elif opportunity_score >= decision_threshold:
        decision = "GO"
        status_text = "Momento propizio"
    else:
        decision = "WAIT"
        status_text = "Setup interessante ma non ancora abbastanza forte"

    meta["opportunity"] = {
        "decision": decision,
        "grade": grade,
        "technical_score": round(technical_score, 1),
        "opportunity_score": opportunity_score,
        "adjustment": round(adjustment, 1),
        "threshold": round(decision_threshold, 1),
        "hard_blocks": hard_blocks,
        "checks": checks,
    }
    signal.meta = meta
    signal.score = opportunity_score
    signal.score_details = (
        f"{signal.score_details}; opportunita {adjustment:+.1f}, "
        f"decisione {decision}, grado {grade}"
    )

    if decision != "GO":
        signal.action = "WATCH"
        block_text = ", ".join(hard_blocks) if hard_blocks else f"score opportunità sotto {decision_threshold:.1f}"
        signal.reason = _append_sentence(signal.reason, f"{status_text}: {block_text}.")
    else:
        signal.reason = _append_sentence(signal.reason, f"{status_text}: checklist {grade}, score opportunità {opportunity_score:.1f}.")

    return signal
