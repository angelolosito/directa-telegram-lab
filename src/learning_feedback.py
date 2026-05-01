from __future__ import annotations

import csv
from pathlib import Path

from .strategy import Signal


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _signal_bucket(signal: Signal, market_regime: dict | None) -> str:
    opportunity = (signal.meta or {}).get("opportunity", {})
    if not isinstance(opportunity, dict):
        opportunity = {}
    market_state = (market_regime or {}).get("state", "") or "unknown_regime"
    return "|".join(
        [
            signal.instrument_type or "unknown",
            signal.strategy or "unknown",
            opportunity.get("decision") or "no_decision",
            opportunity.get("grade") or "no_grade",
            market_state,
        ]
    )


def _row_bucket(row: dict) -> str:
    return "|".join(
        [
            row.get("instrument_type", "") or "unknown",
            row.get("strategy", "") or "unknown",
            row.get("opportunity_decision", "") or "no_decision",
            row.get("opportunity_grade", "") or "no_grade",
            row.get("market_regime", "") or "unknown_regime",
        ]
    )


def load_learning_stats(evaluations_path: Path, config: dict) -> dict[str, dict]:
    learning_cfg = config.get("learning", {})
    if not learning_cfg.get("enabled", True) or not learning_cfg.get("adaptive_feedback_enabled", True):
        return {}

    primary_horizon = int(learning_cfg.get("primary_horizon_sessions", 20))
    rows = [
        row
        for row in _read_csv(evaluations_path)
        if str(row.get("horizon_sessions")) == str(primary_horizon)
    ]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _safe_float(row.get("close_return_pct"))
        if value is None:
            continue
        grouped.setdefault(_row_bucket(row), []).append(value)

    stats: dict[str, dict] = {}
    for bucket, values in grouped.items():
        positives = len([value for value in values if value > 0])
        stats[bucket] = {
            "bucket": bucket,
            "count": len(values),
            "positive_rate": round((positives / len(values)) * 100.0, 1),
            "avg_close_return_pct": round(sum(values) / len(values), 2),
        }
    return stats


def apply_learning_feedback(signal: Signal, learning_stats: dict[str, dict], market_regime: dict | None, config: dict) -> Signal:
    if signal.action != "BUY" or not learning_stats:
        return signal

    learning_cfg = config.get("learning", {})
    min_samples = int(learning_cfg.get("adaptive_min_samples", 5))
    positive_floor = float(learning_cfg.get("adaptive_positive_rate_floor", 45.0))
    avg_floor = float(learning_cfg.get("adaptive_avg_return_floor_pct", -0.5))
    positive_good = float(learning_cfg.get("adaptive_positive_rate_good", 58.0))
    avg_good = float(learning_cfg.get("adaptive_avg_return_good_pct", 0.5))
    penalty = float(learning_cfg.get("adaptive_penalty_points", 6.0))
    bonus = float(learning_cfg.get("adaptive_bonus_points", 3.0))

    bucket = _signal_bucket(signal, market_regime)
    stats = learning_stats.get(bucket)
    if not stats or int(stats.get("count", 0)) < min_samples:
        return signal

    adjustment = 0.0
    verdict = "neutral"
    if stats["positive_rate"] < positive_floor or stats["avg_close_return_pct"] < avg_floor:
        adjustment = -penalty
        verdict = "weak"
    elif stats["positive_rate"] >= positive_good and stats["avg_close_return_pct"] >= avg_good:
        adjustment = bonus
        verdict = "strong"

    if adjustment == 0.0:
        return signal

    meta = signal.meta or {}
    meta["learning_feedback"] = {
        "bucket": bucket,
        "verdict": verdict,
        "adjustment": round(adjustment, 1),
        "count": stats["count"],
        "positive_rate": stats["positive_rate"],
        "avg_close_return_pct": stats["avg_close_return_pct"],
    }
    signal.meta = meta
    signal.score = round(max(0.0, min(100.0, float(signal.score or 0.0) + adjustment)), 1)
    signal.score_details = (
        f"{signal.score_details}; feedback diario {adjustment:+.1f} "
        f"({stats['positive_rate']:.1f}% positivi, {stats['avg_close_return_pct']:+.2f}% medio, n={stats['count']})"
    )
    if verdict == "weak":
        signal.reason += (
            " Diario segnali prudente: questa famiglia storica ha risultati deboli "
            f"({stats['positive_rate']:.1f}% positivi, {stats['avg_close_return_pct']:+.2f}% medio)."
        )
    return signal
