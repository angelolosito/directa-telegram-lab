from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .strategy import Signal


@dataclass(frozen=True)
class AllocationResult:
    selected: list[Signal]
    rejected: list[Signal]
    summary: dict[str, Any]


def _cfg(config: dict) -> dict:
    return config.get("allocation", {})


def _enabled(config: dict) -> bool:
    return bool(_cfg(config).get("enabled", True))


def _meta(signal: Signal) -> dict:
    return signal.meta or {}


def _meta_key(signal: Signal, key: str) -> str | None:
    value = _meta(signal).get(key)
    if value is None or value == "":
        return None
    return str(value)


def _context_key(context: dict, key: str) -> str | None:
    value = context.get(key)
    if value is None or value == "":
        return None
    return str(value)


def _cost_pct(signal: Signal) -> float:
    if signal.notional <= 0:
        return 0.0
    return (signal.estimated_round_trip_cost / signal.notional) * 100.0


def _allocation_score(signal: Signal, market_regime: dict, config: dict) -> float:
    cfg = _cfg(config)
    score = float(signal.score or 0.0)

    rr = float(signal.reward_risk or 0.0)
    min_rr = float(config.get("risk", {}).get("min_reward_risk", 2.0))
    score += max(-2.0, min((rr - min_rr) * 2.0, 4.0))

    cost_penalty = min(_cost_pct(signal), float(cfg.get("max_cost_penalty_pct", 2.0)))
    score -= cost_penalty

    priority = _meta_key(signal, "priority")
    if priority in {"core", "leader"}:
        score += float(cfg.get("leader_priority_bonus", 2.0))

    relative = _meta(signal).get("relative_strength", {})
    if isinstance(relative, dict):
        state = relative.get("state")
        if state == "very_strong":
            score += float(cfg.get("very_strong_relative_bonus", 2.0))
        elif state == "strong":
            score += float(cfg.get("strong_relative_bonus", 1.0))
        elif state == "weak":
            score -= float(cfg.get("weak_relative_penalty", 3.0))

    cautious_states = set(cfg.get("cautious_regime_states", ["neutral"]))
    if market_regime.get("state") in cautious_states:
        if signal.instrument_type == "etf":
            score += float(cfg.get("cautious_etf_bonus", 4.0))
        else:
            score -= float(cfg.get("cautious_stock_penalty", 3.0))

    return round(max(0.0, min(score, 100.0)), 1)


def _count_contexts(open_contexts: list[dict], selected: list[Signal], key: str) -> Counter:
    counts: Counter = Counter()
    for context in open_contexts:
        value = _context_key(context, key)
        if value:
            counts[value] += 1
    for signal in selected:
        value = _meta_key(signal, key)
        if value:
            counts[value] += 1
    return counts


def _constraint_failure(
    signal: Signal,
    selected: list[Signal],
    open_contexts: list[dict],
    config: dict,
) -> str | None:
    cfg = _cfg(config)

    sector = _meta_key(signal, "sector")
    max_sector = int(cfg.get("max_same_sector_open", 1))
    if sector and max_sector > 0 and _count_contexts(open_contexts, selected, "sector")[sector] >= max_sector:
        return f"settore gia coperto: {sector}"

    role = _meta_key(signal, "role")
    max_role = int(cfg.get("max_same_role_open", 1))
    if role and max_role > 0 and _count_contexts(open_contexts, selected, "role")[role] >= max_role:
        return f"ruolo gia coperto: {role}"

    region = _meta_key(signal, "region")
    max_region = int(cfg.get("max_same_region_open", 2))
    if region and max_region > 0 and _count_contexts(open_contexts, selected, "region")[region] >= max_region:
        return f"area gia abbastanza rappresentata: {region}"

    return None


def _annotate(signal: Signal, allocation: dict[str, Any]) -> None:
    meta = dict(signal.meta or {})
    meta["allocation"] = allocation
    signal.meta = meta


def _mark_selected(signal: Signal, rank: int, score: float) -> None:
    _annotate(
        signal,
        {
            "decision": "SELECTED",
            "rank": rank,
            "score": score,
            "reason": "miglior combinazione tra qualita, rischio, costo e diversificazione",
        },
    )
    signal.reason += (
        " Allocazione: selezionato dal motore finale per qualita del segnale e coerenza del portafoglio."
    )


def _mark_rejected(signal: Signal, reason: str, score: float) -> None:
    _annotate(
        signal,
        {
            "decision": "SKIPPED",
            "score": score,
            "reason": reason,
        },
    )
    signal.action = "WATCH"
    signal.reason += f" Allocazione: non scelto oggi ({reason})."


def select_portfolio_candidates(
    candidates: list[Signal],
    open_contexts: list[dict],
    market_regime: dict,
    config: dict,
    max_new_positions: int | None = None,
) -> AllocationResult:
    if not candidates:
        return AllocationResult(
            selected=[],
            rejected=[],
            summary={
                "enabled": _enabled(config),
                "candidates": 0,
                "selected": 0,
                "rejected": 0,
                "reason": "nessun candidato operativo",
            },
        )

    cfg = _cfg(config)
    if not _enabled(config):
        limit = max_new_positions or int(cfg.get("max_new_positions_per_run", 1))
        selected = candidates[:limit]
        for idx, signal in enumerate(selected, start=1):
            _mark_selected(signal, idx, float(signal.score or 0.0))
        return AllocationResult(
            selected=selected,
            rejected=[],
            summary={
                "enabled": False,
                "candidates": len(candidates),
                "selected": len(selected),
                "rejected": 0,
                "reason": "motore allocazione disattivato",
            },
        )

    limit = int(max_new_positions or cfg.get("max_new_positions_per_run", 1))
    limit = max(0, limit)
    scored = [
        (
            _allocation_score(signal, market_regime, config),
            float(signal.score or 0.0),
            float(signal.reward_risk or 0.0),
            -_cost_pct(signal),
            signal,
        )
        for signal in candidates
    ]
    scored.sort(reverse=True, key=lambda item: item[:4])

    selected: list[Signal] = []
    rejected: list[Signal] = []
    rejection_reasons: Counter = Counter()

    for allocation_score, _score, _rr, _cost, signal in scored:
        if len(selected) >= limit:
            reason = "limite nuovi ingressi gia raggiunto"
            _mark_rejected(signal, reason, allocation_score)
            rejected.append(signal)
            rejection_reasons[reason] += 1
            continue

        reason = _constraint_failure(signal, selected, open_contexts, config)
        if reason:
            _mark_rejected(signal, reason, allocation_score)
            rejected.append(signal)
            rejection_reasons[reason] += 1
            continue

        selected.append(signal)
        _mark_selected(signal, len(selected), allocation_score)

    if not selected and rejected:
        top_reason = rejected[0].meta.get("allocation", {}).get("reason", "vincoli di portafoglio")
        reason = f"nessun ingresso: {top_reason}"
    elif selected:
        reason = f"selezionato {selected[0].symbol}"
    else:
        reason = "nessun candidato operativo"

    summary = {
        "enabled": True,
        "candidates": len(candidates),
        "selected": len(selected),
        "rejected": len(rejected),
        "reason": reason,
        "selected_symbols": [signal.symbol for signal in selected],
        "top_rejections": rejection_reasons.most_common(3),
        "open_contexts": len(open_contexts),
    }
    return AllocationResult(selected=selected, rejected=rejected, summary=summary)
