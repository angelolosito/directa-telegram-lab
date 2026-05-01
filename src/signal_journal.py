from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .strategy import Signal


JOURNAL_FIELDNAMES = [
    "signal_id",
    "date",
    "symbol",
    "name",
    "instrument_type",
    "action",
    "strategy",
    "score",
    "opportunity_decision",
    "opportunity_grade",
    "market_regime",
    "price",
    "entry",
    "stop",
    "target",
    "reward_risk",
    "cost_pct",
    "reason",
    "meta_json",
    "created_at",
]


EVALUATION_FIELDNAMES = [
    "evaluation_id",
    "signal_id",
    "date",
    "symbol",
    "instrument_type",
    "strategy",
    "action",
    "score",
    "opportunity_decision",
    "opportunity_grade",
    "market_regime",
    "horizon_sessions",
    "entry_price",
    "end_date",
    "close_return_pct",
    "max_gain_pct",
    "max_drawdown_pct",
    "hit_target",
    "hit_stop",
    "outcome",
    "updated_at",
]


def _safe_float(value) -> float | None:
    try:
        if value in ("", None):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _format_float(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _opportunity(signal: Signal) -> dict[str, Any]:
    meta = signal.meta or {}
    opportunity = meta.get("opportunity")
    return opportunity if isinstance(opportunity, dict) else {}


def _signal_id(signal: Signal) -> str:
    raw = "|".join(
        [
            str(signal.date),
            signal.symbol,
            signal.strategy,
            signal.action,
            _format_float(signal.price),
            _format_float(signal.entry),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _market_state(market_regime: dict | None) -> str:
    if not market_regime:
        return ""
    return str(market_regime.get("state", ""))


def signal_to_journal_row(signal: Signal, market_regime: dict | None, run_date: date) -> dict:
    opportunity = _opportunity(signal)
    cost_pct = None
    if signal.notional > 0:
        cost_pct = (signal.estimated_round_trip_cost / signal.notional) * 100.0

    return {
        "signal_id": _signal_id(signal),
        "date": signal.date,
        "symbol": signal.symbol,
        "name": signal.name,
        "instrument_type": signal.instrument_type,
        "action": signal.action,
        "strategy": signal.strategy,
        "score": _format_float(signal.score, 1),
        "opportunity_decision": opportunity.get("decision", ""),
        "opportunity_grade": opportunity.get("grade", ""),
        "market_regime": _market_state(market_regime),
        "price": _format_float(signal.price),
        "entry": _format_float(signal.entry),
        "stop": _format_float(signal.stop),
        "target": _format_float(signal.target),
        "reward_risk": _format_float(signal.reward_risk, 2),
        "cost_pct": _format_float(cost_pct, 2),
        "reason": signal.reason,
        "meta_json": json.dumps(signal.meta or {}, ensure_ascii=False, sort_keys=True),
        "created_at": run_date.isoformat(),
    }


def append_signal_journal(
    path: Path,
    signals: list[Signal],
    market_regime: dict | None,
    run_date: date,
) -> int:
    if not signals:
        return 0

    existing_rows = _read_csv(path)
    rows_by_id = {row.get("signal_id", ""): row for row in existing_rows if row.get("signal_id")}
    added = 0
    for signal in signals:
        if signal.action not in {"BUY", "WATCH"}:
            continue
        row = signal_to_journal_row(signal, market_regime, run_date)
        if row["signal_id"] in rows_by_id:
            continue
        rows_by_id[row["signal_id"]] = row
        added += 1

    if added:
        ordered_rows = sorted(rows_by_id.values(), key=lambda row: (row.get("date", ""), row.get("symbol", "")))
        _write_csv(path, JOURNAL_FIELDNAMES, ordered_rows)
    return added


def _entry_price(row: dict) -> float | None:
    return _safe_float(row.get("entry")) or _safe_float(row.get("price"))


def _window_after_signal(df: pd.DataFrame, signal_date: str, horizon: int) -> pd.DataFrame:
    if df.empty:
        return df
    clean = df.dropna(subset=["Close", "High", "Low"]).copy()
    clean.index = pd.to_datetime(clean.index)
    start = pd.Timestamp(signal_date)
    forward = clean.loc[clean.index > start]
    return forward.head(horizon)


def _first_hit_day(window: pd.DataFrame, column: str, threshold: float, direction: str) -> pd.Timestamp | None:
    if direction == "above":
        hits = window[window[column] >= threshold]
    else:
        hits = window[window[column] <= threshold]
    if hits.empty:
        return None
    return pd.Timestamp(hits.index[0])


def _classify_outcome(
    close_return_pct: float,
    target_day: pd.Timestamp | None,
    stop_day: pd.Timestamp | None,
) -> str:
    if target_day is not None and stop_day is not None:
        if target_day < stop_day:
            return "target_before_stop"
        if stop_day < target_day:
            return "stop_before_target"
        return "target_and_stop_same_day"
    if target_day is not None:
        return "target_hit"
    if stop_day is not None:
        return "stop_hit"
    if close_return_pct > 0:
        return "positive"
    if close_return_pct < 0:
        return "negative"
    return "flat"


def _evaluation_row(row: dict, df: pd.DataFrame, horizon: int, updated_at: date) -> dict | None:
    entry = _entry_price(row)
    if entry is None or entry <= 0:
        return None

    window = _window_after_signal(df, row.get("date", ""), horizon)
    if len(window) < horizon:
        return None

    end_row = window.iloc[-1]
    close = float(end_row["Close"])
    high = float(window["High"].max())
    low = float(window["Low"].min())
    target = _safe_float(row.get("target"))
    stop = _safe_float(row.get("stop"))
    target_day = _first_hit_day(window, "High", target, "above") if target else None
    stop_day = _first_hit_day(window, "Low", stop, "below") if stop else None
    close_return_pct = ((close / entry) - 1.0) * 100.0

    evaluation_id = f"{row.get('signal_id')}:{horizon}"
    return {
        "evaluation_id": evaluation_id,
        "signal_id": row.get("signal_id", ""),
        "date": row.get("date", ""),
        "symbol": row.get("symbol", ""),
        "instrument_type": row.get("instrument_type", ""),
        "strategy": row.get("strategy", ""),
        "action": row.get("action", ""),
        "score": row.get("score", ""),
        "opportunity_decision": row.get("opportunity_decision", ""),
        "opportunity_grade": row.get("opportunity_grade", ""),
        "market_regime": row.get("market_regime", ""),
        "horizon_sessions": str(horizon),
        "entry_price": _format_float(entry),
        "end_date": pd.Timestamp(window.index[-1]).date().isoformat(),
        "close_return_pct": _format_float(close_return_pct, 2),
        "max_gain_pct": _format_float(((high / entry) - 1.0) * 100.0, 2),
        "max_drawdown_pct": _format_float(((low / entry) - 1.0) * 100.0, 2),
        "hit_target": "true" if target_day is not None else "false",
        "hit_stop": "true" if stop_day is not None else "false",
        "outcome": _classify_outcome(close_return_pct, target_day, stop_day),
        "updated_at": updated_at.isoformat(),
    }


def _summarize_evaluations(rows: list[dict], primary_horizon: int, min_bucket_count: int) -> dict:
    primary = [row for row in rows if str(row.get("horizon_sessions")) == str(primary_horizon)]
    returns = [_safe_float(row.get("close_return_pct")) for row in primary]
    returns = [value for value in returns if value is not None]
    if not returns:
        return {
            "primary_horizon": primary_horizon,
            "completed": 0,
            "positive_rate": None,
            "avg_close_return_pct": None,
            "best_bucket": "",
            "weak_bucket": "",
        }

    positive = len([value for value in returns if value > 0])
    buckets: dict[str, list[float]] = {}
    for row in primary:
        value = _safe_float(row.get("close_return_pct"))
        if value is None:
            continue
        bucket = "|".join(
            [
                row.get("strategy", ""),
                row.get("opportunity_decision", "") or "no_decision",
                row.get("opportunity_grade", "") or "no_grade",
            ]
        )
        buckets.setdefault(bucket, []).append(value)

    eligible = {key: values for key, values in buckets.items() if len(values) >= min_bucket_count}
    best_bucket = ""
    weak_bucket = ""
    if eligible:
        averages = {key: sum(values) / len(values) for key, values in eligible.items()}
        best_key = max(averages, key=averages.get)
        weak_key = min(averages, key=averages.get)
        best_bucket = f"{best_key} ({averages[best_key]:+.2f}%, n={len(eligible[best_key])})"
        weak_bucket = f"{weak_key} ({averages[weak_key]:+.2f}%, n={len(eligible[weak_key])})"

    return {
        "primary_horizon": primary_horizon,
        "completed": len(returns),
        "positive_rate": round((positive / len(returns)) * 100.0, 1),
        "avg_close_return_pct": round(sum(returns) / len(returns), 2),
        "best_bucket": best_bucket,
        "weak_bucket": weak_bucket,
    }


def _same_evaluation(existing: dict, evaluated: dict) -> bool:
    return all(
        existing.get(field, "") == evaluated.get(field, "")
        for field in EVALUATION_FIELDNAMES
        if field != "updated_at"
    )


def update_signal_evaluations(
    journal_path: Path,
    evaluations_path: Path,
    market_data: dict[str, pd.DataFrame],
    config: dict,
    run_date: date,
) -> dict:
    learning_cfg = config.get("learning", {})
    horizons = [int(value) for value in learning_cfg.get("horizons_sessions", [5, 10, 20, 40])]
    primary_horizon = int(learning_cfg.get("primary_horizon_sessions", 20))
    min_bucket_count = int(learning_cfg.get("min_bucket_count", 2))

    journal_rows = _read_csv(journal_path)
    existing_rows = _read_csv(evaluations_path)
    rows_by_id = {row.get("evaluation_id", ""): row for row in existing_rows if row.get("evaluation_id")}
    new_or_updated = 0

    for row in journal_rows:
        symbol = row.get("symbol", "")
        df = market_data.get(symbol)
        if df is None:
            continue
        for horizon in horizons:
            evaluated = _evaluation_row(row, df, horizon, run_date)
            if evaluated is None:
                continue
            existing = rows_by_id.get(evaluated["evaluation_id"])
            if existing is None or not _same_evaluation(existing, evaluated):
                rows_by_id[evaluated["evaluation_id"]] = evaluated
                new_or_updated += 1

    ordered_rows = sorted(
        rows_by_id.values(),
        key=lambda row: (row.get("date", ""), row.get("symbol", ""), int(row.get("horizon_sessions", 0) or 0)),
    )
    if new_or_updated:
        _write_csv(evaluations_path, EVALUATION_FIELDNAMES, ordered_rows)

    summary = _summarize_evaluations(ordered_rows, primary_horizon, min_bucket_count)
    summary["new_or_updated"] = new_or_updated
    summary["journal_size"] = len(journal_rows)
    return summary


def _bucket_key(row: dict, include_market: bool = True) -> str:
    parts = [
        row.get("instrument_type", "") or "unknown",
        row.get("strategy", "") or "unknown",
        row.get("opportunity_decision", "") or "no_decision",
        row.get("opportunity_grade", "") or "no_grade",
    ]
    if include_market:
        parts.append(row.get("market_regime", "") or "unknown_regime")
    return "|".join(parts)


def _bucket_stats(rows: list[dict], primary_horizon: int, min_bucket_count: int) -> list[dict]:
    primary = [row for row in rows if str(row.get("horizon_sessions")) == str(primary_horizon)]
    buckets: dict[str, list[float]] = {}
    for row in primary:
        value = _safe_float(row.get("close_return_pct"))
        if value is None:
            continue
        buckets.setdefault(_bucket_key(row), []).append(value)

    stats: list[dict] = []
    for bucket, values in buckets.items():
        if len(values) < min_bucket_count:
            continue
        positives = len([value for value in values if value > 0])
        stats.append(
            {
                "bucket": bucket,
                "count": len(values),
                "positive_rate": round((positives / len(values)) * 100.0, 1),
                "avg_close_return_pct": round(sum(values) / len(values), 2),
                "best_return_pct": round(max(values), 2),
                "worst_return_pct": round(min(values), 2),
            }
        )
    return stats


def build_learning_report(journal_path: Path, evaluations_path: Path, config: dict) -> str:
    learning_cfg = config.get("learning", {})
    primary_horizon = int(learning_cfg.get("primary_horizon_sessions", 20))
    min_bucket_count = int(learning_cfg.get("min_bucket_count", 2))
    journal_rows = _read_csv(journal_path)
    evaluation_rows = _read_csv(evaluations_path)
    summary = _summarize_evaluations(evaluation_rows, primary_horizon, min_bucket_count)
    stats = _bucket_stats(evaluation_rows, primary_horizon, min_bucket_count)
    best = sorted(stats, key=lambda row: row["avg_close_return_pct"], reverse=True)[:5]
    weak = sorted(stats, key=lambda row: row["avg_close_return_pct"])[:5]
    recent = sorted(
        [row for row in evaluation_rows if str(row.get("horizon_sessions")) == str(primary_horizon)],
        key=lambda row: (row.get("end_date", ""), row.get("symbol", "")),
        reverse=True,
    )[:10]

    lines = [
        "# Diario intelligente segnali",
        "",
        f"Segnali in memoria: {len(journal_rows)}",
        f"Valutazioni totali: {len(evaluation_rows)}",
        f"Orizzonte principale: {primary_horizon} sedute",
        f"Valutazioni complete sull'orizzonte principale: {summary.get('completed', 0)}",
        f"Tasso positivi: {summary.get('positive_rate', 'n/d')}%",
        f"Rendimento medio: {summary.get('avg_close_return_pct', 'n/d')}%",
        "",
    ]

    if best:
        lines.extend(["## Setup migliori", ""])
        for row in best:
            lines.append(
                f"- {row['bucket']}: {row['avg_close_return_pct']:+.2f}% medio, "
                f"{row['positive_rate']:.1f}% positivi, n={row['count']}"
            )
        lines.append("")

    if weak:
        lines.extend(["## Setup più deboli", ""])
        for row in weak:
            lines.append(
                f"- {row['bucket']}: {row['avg_close_return_pct']:+.2f}% medio, "
                f"{row['positive_rate']:.1f}% positivi, n={row['count']}"
            )
        lines.append("")

    if recent:
        lines.extend(["## Ultime valutazioni", ""])
        for row in recent:
            lines.append(
                f"- {row.get('end_date')} {row.get('symbol')} {row.get('strategy')} "
                f"{row.get('opportunity_decision')}/{row.get('opportunity_grade')}: "
                f"{row.get('close_return_pct')}% ({row.get('outcome')})"
            )
        lines.append("")

    if not evaluation_rows:
        lines.append("Non ci sono ancora valutazioni: servono alcuni run e abbastanza sedute dopo i segnali.")

    lines.append("Nota: statistiche su segnali paper/didattici, non garanzia di performance futura.")
    return "\n".join(lines).strip()
