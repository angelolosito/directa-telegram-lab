from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path
from queue import Empty
from typing import Any

from .strategy import Signal


METRIC_KEYS = [
    "market_cap",
    "trailing_pe",
    "forward_pe",
    "price_to_sales",
    "price_to_book",
    "ev_to_ebitda",
    "profit_margin",
    "operating_margin",
    "roe",
    "revenue_growth",
    "earnings_growth",
    "debt_to_equity",
    "current_ratio",
    "operating_cashflow",
    "free_cashflow",
    "dividend_yield",
    "payout_ratio",
    "beta",
    "recommendation_mean",
    "analyst_count",
    "eps_revision_balance",
    "target_mean_price",
    "current_price",
]

EVENT_KEYS = [
    "next_earnings_date",
    "latest_quarter",
]


@dataclass(frozen=True)
class FundamentalSnapshot:
    symbol: str
    provider: str
    source_symbol: str
    fetched_at: str
    metrics: dict[str, float | None]
    events: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "provider": self.provider,
            "source_symbol": self.source_symbol,
            "fetched_at": self.fetched_at,
            "metrics": self.metrics,
            "events": self.events,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FundamentalSnapshot":
        return cls(
            symbol=str(data.get("symbol", "")),
            provider=str(data.get("provider", "")),
            source_symbol=str(data.get("source_symbol", "")),
            fetched_at=str(data.get("fetched_at", "")),
            metrics={key: _parse_float((data.get("metrics") or {}).get(key)) for key in METRIC_KEYS},
            events={key: _parse_event((data.get("events") or {}).get(key)) for key in EVENT_KEYS},
            error=data.get("error"),
        )


@dataclass(frozen=True)
class FundamentalReview:
    score: float | None
    state: str
    adjustment: float
    reason: str
    source: str
    metrics: dict[str, float | None]
    subscores: dict[str, float | None]
    events: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "state": self.state,
            "adjustment": self.adjustment,
            "reason": self.reason,
            "source": self.source,
            "metrics": self.metrics,
            "subscores": self.subscores,
            "events": self.events,
            "error": self.error,
        }


class FundamentalDataError(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "None", "null", "-", "NoneType"}:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:
        return None
    return parsed


def _parse_event(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"None", "null", "-", "NaT"}:
        return None
    return text


def _parse_iso_date(value: Any):
    text = _parse_event(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except Exception:
        return None


def _normalize_percent(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) > 2:
        return value / 100.0
    return value


def _avg(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 1)


def _score_margin(value: float | None) -> float | None:
    value = _normalize_percent(value)
    if value is None:
        return None
    if value >= 0.25:
        return 95.0
    if value >= 0.15:
        return 82.0
    if value >= 0.08:
        return 68.0
    if value >= 0.02:
        return 52.0
    if value >= 0:
        return 38.0
    return 15.0


def _score_roe(value: float | None) -> float | None:
    value = _normalize_percent(value)
    if value is None:
        return None
    if value >= 0.25:
        return 95.0
    if value >= 0.15:
        return 82.0
    if value >= 0.08:
        return 64.0
    if value > 0:
        return 45.0
    return 15.0


def _score_growth(value: float | None) -> float | None:
    value = _normalize_percent(value)
    if value is None:
        return None
    if value >= 0.20:
        return 95.0
    if value >= 0.10:
        return 82.0
    if value >= 0.04:
        return 66.0
    if value >= 0:
        return 50.0
    if value >= -0.08:
        return 32.0
    return 15.0


def _score_debt_to_equity(value: float | None) -> float | None:
    if value is None:
        return None
    # yfinance normally reports this as percent, while some APIs return a ratio.
    ratio = value / 100.0 if value > 10 else value
    if ratio <= 0.3:
        return 92.0
    if ratio <= 0.8:
        return 78.0
    if ratio <= 1.5:
        return 58.0
    if ratio <= 3.0:
        return 35.0
    return 15.0


def _score_current_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 1.5:
        return 85.0
    if value >= 1.0:
        return 68.0
    if value >= 0.7:
        return 42.0
    return 20.0


def _score_cashflow(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 0:
        return 80.0
    if value == 0:
        return 45.0
    return 20.0


def _score_pe(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if 8 <= value <= 24:
        return 86.0
    if 24 < value <= 35:
        return 70.0
    if 4 <= value < 8:
        return 64.0
    if 35 < value <= 55:
        return 45.0
    return 25.0


def _score_price_to_sales(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= 3:
        return 82.0
    if value <= 7:
        return 64.0
    if value <= 12:
        return 42.0
    return 22.0


def _score_ev_to_ebitda(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= 12:
        return 84.0
    if value <= 20:
        return 64.0
    if value <= 30:
        return 42.0
    return 22.0


def _score_payout(value: float | None) -> float | None:
    value = _normalize_percent(value)
    if value is None:
        return None
    if value <= 0:
        return None
    if value <= 0.65:
        return 78.0
    if value <= 0.9:
        return 52.0
    return 28.0


def _score_analyst(value: float | None) -> float | None:
    if value is None:
        return None
    # Lower is better for Yahoo recommendationMean.
    if value <= 2.0:
        return 82.0
    if value <= 2.7:
        return 64.0
    if value <= 3.2:
        return 48.0
    return 30.0


def _score_revision_balance(value: float | None) -> float | None:
    if value is None:
        return None
    if value >= 4:
        return 86.0
    if value > 0:
        return 70.0
    if value == 0:
        return 55.0
    if value >= -3:
        return 40.0
    return 20.0


def _subscores(metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "profitability": _avg(
            [
                _score_margin(metrics.get("profit_margin")),
                _score_margin(metrics.get("operating_margin")),
                _score_roe(metrics.get("roe")),
            ]
        ),
        "growth": _avg(
            [
                _score_growth(metrics.get("revenue_growth")),
                _score_growth(metrics.get("earnings_growth")),
            ]
        ),
        "balance_sheet": _avg(
            [
                _score_debt_to_equity(metrics.get("debt_to_equity")),
                _score_current_ratio(metrics.get("current_ratio")),
            ]
        ),
        "cashflow": _avg(
            [
                _score_cashflow(metrics.get("operating_cashflow")),
                _score_cashflow(metrics.get("free_cashflow")),
            ]
        ),
        "valuation": _avg(
            [
                _score_pe(metrics.get("forward_pe") or metrics.get("trailing_pe")),
                _score_price_to_sales(metrics.get("price_to_sales")),
                _score_ev_to_ebitda(metrics.get("ev_to_ebitda")),
            ]
        ),
        "shareholder": _avg(
            [
                _score_payout(metrics.get("payout_ratio")),
            ]
        ),
        "analyst": _avg(
            [
                _score_analyst(metrics.get("recommendation_mean")),
                _score_revision_balance(metrics.get("eps_revision_balance")),
            ]
        ),
    }


def _weighted_score(subscores: dict[str, float | None], config: dict) -> float | None:
    default_weights = {
        "profitability": 0.25,
        "growth": 0.20,
        "balance_sheet": 0.18,
        "cashflow": 0.17,
        "valuation": 0.15,
        "shareholder": 0.03,
        "analyst": 0.02,
    }
    weights = {**default_weights, **(config.get("fundamentals", {}).get("weights") or {})}
    total_weight = 0.0
    total_score = 0.0
    for key, weight in weights.items():
        value = subscores.get(key)
        if value is None:
            continue
        total_weight += float(weight)
        total_score += float(value) * float(weight)
    if total_weight <= 0:
        return None
    return round(total_score / total_weight, 1)


def evaluate_fundamentals(snapshot: FundamentalSnapshot, config: dict) -> FundamentalReview:
    if snapshot.error:
        return FundamentalReview(
            score=None,
            state="unknown",
            adjustment=0.0,
            reason=f"fondamentali non disponibili: {snapshot.error}",
            source=snapshot.provider,
            metrics=snapshot.metrics,
            subscores={},
            events=snapshot.events,
            error=snapshot.error,
        )

    subscores = _subscores(snapshot.metrics)
    score = _weighted_score(subscores, config)
    if score is None:
        return FundamentalReview(
            score=None,
            state="unknown",
            adjustment=0.0,
            reason="fondamentali insufficienti per una valutazione affidabile",
            source=snapshot.provider,
            metrics=snapshot.metrics,
            subscores=subscores,
            events=snapshot.events,
        )

    cfg = config.get("fundamentals", {})
    strong = float(cfg.get("strong_score", 72.0))
    healthy = float(cfg.get("healthy_score", 60.0))
    weak = float(cfg.get("weak_score", 45.0))
    adjustments = cfg.get("score_adjustments", {})

    if score >= strong:
        state = "strong"
        adjustment = float(adjustments.get("strong", 6.0))
        reason = "qualita fondamentale forte"
    elif score >= healthy:
        state = "healthy"
        adjustment = float(adjustments.get("healthy", 3.0))
        reason = "fondamentali sani"
    elif score >= weak:
        state = "mixed"
        adjustment = float(adjustments.get("mixed", 0.0))
        reason = "fondamentali misti, senza vantaggio chiaro"
    else:
        state = "weak"
        adjustment = float(adjustments.get("weak", -8.0))
        reason = "fondamentali deboli o deteriorati"

    return FundamentalReview(
        score=score,
        state=state,
        adjustment=adjustment,
        reason=reason,
        source=snapshot.provider,
        metrics=snapshot.metrics,
        subscores=subscores,
        events=snapshot.events,
    )


def _quality_gate(review: FundamentalReview, cfg: dict) -> dict[str, Any]:
    gate_cfg = cfg.get("quality_gate", {})
    if not gate_cfg.get("enabled", True) or review.score is None:
        return {"enabled": bool(gate_cfg.get("enabled", True)), "blocked": False, "weak_subscores": []}

    thresholds = gate_cfg.get("min_subscores") or {
        "profitability": 35.0,
        "balance_sheet": 35.0,
        "cashflow": 35.0,
    }
    weak_subscores = []
    for key, threshold in thresholds.items():
        value = review.subscores.get(key)
        if value is not None and value < float(threshold):
            weak_subscores.append({"name": key, "score": value, "threshold": float(threshold)})

    max_failures = int(gate_cfg.get("max_critical_failures", 0))
    blocked = bool(gate_cfg.get("block_on_critical_subscores", True)) and len(weak_subscores) > max_failures
    return {
        "enabled": True,
        "blocked": blocked,
        "weak_subscores": weak_subscores,
    }


def _earnings_blackout(events: dict[str, str | None], signal_date: str, cfg: dict) -> dict[str, Any]:
    blackout_cfg = cfg.get("earnings_blackout", {})
    if not blackout_cfg.get("enabled", True):
        return {"enabled": False, "active": False}

    earnings_date = _parse_iso_date(events.get("next_earnings_date"))
    run_date = _parse_iso_date(signal_date)
    if earnings_date is None or run_date is None:
        return {"enabled": True, "active": False}

    days_to_earnings = (earnings_date - run_date).days
    days_before = int(blackout_cfg.get("days_before", 5))
    days_after = int(blackout_cfg.get("days_after", 1))
    active = -days_after <= days_to_earnings <= days_before
    return {
        "enabled": True,
        "active": active,
        "earnings_date": earnings_date.isoformat(),
        "days_to_earnings": days_to_earnings,
        "action": str(blackout_cfg.get("action", "watch")),
        "penalty_points": float(blackout_cfg.get("penalty_points", 5.0)),
    }


def apply_fundamental_review(
    signal: Signal,
    instrument: dict[str, Any],
    snapshot: FundamentalSnapshot | None,
    config: dict,
) -> Signal:
    cfg = config.get("fundamentals", {})
    if not cfg.get("enabled", False) or signal.action != "BUY":
        return signal
    if instrument.get("type") not in set(cfg.get("apply_to_types", ["stock"])):
        return signal

    if snapshot is None:
        review = FundamentalReview(
            score=None,
            state="unknown",
            adjustment=0.0,
            reason="snapshot fondamentali assente",
            source="none",
            metrics={},
            subscores={},
            events={},
        )
    else:
        review = evaluate_fundamentals(snapshot, config)

    quality_gate = _quality_gate(review, cfg)
    earnings_blackout = _earnings_blackout(review.events, signal.date, cfg)

    if review.adjustment:
        signal.score = round(max(0.0, min(100.0, float(signal.score or 0.0) + review.adjustment)), 1)
        signal.score_details = f"{signal.score_details}; fondamentali {review.adjustment:+.1f}"

    if earnings_blackout.get("active") and earnings_blackout.get("action") == "penalty":
        penalty = float(earnings_blackout.get("penalty_points", 5.0))
        signal.score = round(max(0.0, float(signal.score or 0.0) - penalty), 1)
        signal.score_details = f"{signal.score_details}; trimestrale vicina -{penalty:.1f}"

    review_payload = review.to_dict()
    review_payload["quality_gate"] = quality_gate
    review_payload["earnings_blackout"] = earnings_blackout
    meta = dict(signal.meta or {})
    meta["fundamentals"] = review_payload
    signal.meta = meta

    block_score = float(cfg.get("block_below_score", cfg.get("weak_score", 45.0)))
    block_weak = bool(cfg.get("block_when_weak", True))
    block_missing = bool(cfg.get("block_when_missing", False))
    if review.score is not None and review.score < block_score and block_weak:
        signal.action = "WATCH"
        signal.reason += (
            f" Fondamentali deboli: score {review.score:.1f}/100, sotto blocco {block_score:.1f}."
        )
    elif quality_gate.get("blocked"):
        weak_names = ", ".join(item["name"] for item in quality_gate.get("weak_subscores", []))
        signal.action = "WATCH"
        signal.reason += f" Quality gate fondamentale non superato: {weak_names} sotto soglia."
    elif earnings_blackout.get("active") and earnings_blackout.get("action") == "watch":
        days = int(earnings_blackout.get("days_to_earnings", 0))
        label = "tra" if days >= 0 else "da"
        signal.action = "WATCH"
        signal.reason += (
            f" Trimestrale troppo vicina: {earnings_blackout.get('earnings_date')} "
            f"({label} {abs(days)} giorni)."
        )
    elif review.score is None and block_missing:
        signal.action = "WATCH"
        signal.reason += " Fondamentali mancanti: segnale sospeso per prudenza."
    elif review.score is not None:
        signal.reason += f" Fondamentali: {review.state} ({review.score:.1f}/100, {review.reason})."
    else:
        signal.reason += f" Fondamentali: {review.reason}."

    return signal


def _normal_metrics() -> dict[str, float | None]:
    return {key: None for key in METRIC_KEYS}


def _normal_events() -> dict[str, str | None]:
    return {key: None for key in EVENT_KEYS}


def _format_date_like(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(float(value), timezone.utc).date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    text = str(value).strip()
    if not text or text in {"None", "NaT", "nan"}:
        return None
    if " " in text:
        text = text.split(" ")[0]
    return text[:10]


def _extract_earnings_date(info: dict[str, Any]) -> str | None:
    for key in ("earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd", "nextEarningsDate"):
        parsed = _format_date_like(info.get(key))
        if parsed:
            return parsed
    return None


def _extract_revision_balance(payload: Any) -> float | None:
    up = 0.0
    down = 0.0
    found = False

    def walk(value: Any, label: str = "") -> None:
        nonlocal up, down, found
        if isinstance(value, dict):
            for key, nested in value.items():
                walk(nested, f"{label} {key}".lower())
            return
        if isinstance(value, list):
            for nested in value:
                walk(nested, label)
            return
        parsed = _parse_float(value)
        if parsed is None:
            return
        clean_label = label.lower()
        if "up" in clean_label or "positive" in clean_label:
            up += parsed
            found = True
        elif "down" in clean_label or "negative" in clean_label:
            down += parsed
            found = True

    if hasattr(payload, "to_dict"):
        try:
            payload = payload.to_dict()
        except Exception:
            return None
    walk(payload)
    if not found:
        return None
    return round(up - down, 2)


def _supplement_yfinance_snapshot(snapshot: FundamentalSnapshot, ticker: Any) -> FundamentalSnapshot:
    metrics = dict(snapshot.metrics)
    events = dict(snapshot.events)

    try:
        revisions = ticker.get_eps_revisions()
        metrics["eps_revision_balance"] = _extract_revision_balance(revisions)
    except Exception:
        pass

    try:
        calendar = ticker.get_calendar()
        if hasattr(calendar, "to_dict"):
            calendar = calendar.to_dict()
        if isinstance(calendar, dict):
            for key, value in calendar.items():
                if "earn" in str(key).lower():
                    if isinstance(value, list) and value:
                        events["next_earnings_date"] = _format_date_like(value[0])
                    else:
                        events["next_earnings_date"] = _format_date_like(value)
                    break
    except Exception:
        pass

    try:
        dates = ticker.get_earnings_dates(limit=4)
        if events.get("next_earnings_date") is None and getattr(dates, "empty", True) is False:
            index = getattr(dates, "index", [])
            if len(index):
                events["next_earnings_date"] = _format_date_like(index[0])
    except Exception:
        pass

    return FundamentalSnapshot(
        symbol=snapshot.symbol,
        provider=snapshot.provider,
        source_symbol=snapshot.source_symbol,
        fetched_at=snapshot.fetched_at,
        metrics=metrics,
        events=events,
        error=snapshot.error,
    )


def _from_yfinance_info(symbol: str, source_symbol: str, info: dict[str, Any]) -> FundamentalSnapshot:
    mapping = {
        "market_cap": "marketCap",
        "trailing_pe": "trailingPE",
        "forward_pe": "forwardPE",
        "price_to_sales": "priceToSalesTrailing12Months",
        "price_to_book": "priceToBook",
        "ev_to_ebitda": "enterpriseToEbitda",
        "profit_margin": "profitMargins",
        "operating_margin": "operatingMargins",
        "roe": "returnOnEquity",
        "revenue_growth": "revenueGrowth",
        "earnings_growth": "earningsGrowth",
        "debt_to_equity": "debtToEquity",
        "current_ratio": "currentRatio",
        "operating_cashflow": "operatingCashflow",
        "free_cashflow": "freeCashflow",
        "dividend_yield": "dividendYield",
        "payout_ratio": "payoutRatio",
        "beta": "beta",
        "recommendation_mean": "recommendationMean",
        "analyst_count": "numberOfAnalystOpinions",
        "target_mean_price": "targetMeanPrice",
        "current_price": "currentPrice",
    }
    metrics = _normal_metrics()
    for target, source in mapping.items():
        metrics[target] = _parse_float(info.get(source))
    events = _normal_events()
    events["next_earnings_date"] = _extract_earnings_date(info)
    return FundamentalSnapshot(
        symbol=symbol,
        provider="yfinance",
        source_symbol=source_symbol,
        fetched_at=_now_iso(),
        metrics=metrics,
        events=events,
    )


def _from_alpha_vantage(symbol: str, source_symbol: str, data: dict[str, Any]) -> FundamentalSnapshot:
    mapping = {
        "market_cap": "MarketCapitalization",
        "trailing_pe": "PERatio",
        "forward_pe": "ForwardPE",
        "price_to_sales": "PriceToSalesRatioTTM",
        "price_to_book": "PriceToBookRatio",
        "ev_to_ebitda": "EVToEBITDA",
        "profit_margin": "ProfitMargin",
        "operating_margin": "OperatingMarginTTM",
        "roe": "ReturnOnEquityTTM",
        "revenue_growth": "QuarterlyRevenueGrowthYOY",
        "earnings_growth": "QuarterlyEarningsGrowthYOY",
        "dividend_yield": "DividendYield",
        "payout_ratio": "PayoutRatio",
        "beta": "Beta",
        "target_mean_price": "AnalystTargetPrice",
    }
    metrics = _normal_metrics()
    for target, source in mapping.items():
        metrics[target] = _parse_float(data.get(source))
    events = _normal_events()
    events["latest_quarter"] = _parse_event(data.get("LatestQuarter"))
    return FundamentalSnapshot(
        symbol=symbol,
        provider="alpha_vantage",
        source_symbol=source_symbol,
        fetched_at=_now_iso(),
        metrics=metrics,
        events=events,
    )


def _yfinance_worker(queue, symbol: str, source_symbol: str) -> None:
    try:
        import yfinance as yf

        ticker = yf.Ticker(source_symbol)
        info = ticker.info or {}
        if not info:
            raise FundamentalDataError("risposta yfinance vuota")
        snapshot = _from_yfinance_info(symbol, source_symbol, info)
        snapshot = _supplement_yfinance_snapshot(snapshot, ticker)
        queue.put(("ok", snapshot.to_dict()))
    except Exception as e:  # noqa: BLE001
        queue.put(("error", repr(e)))


def _alpha_vantage_worker(queue, symbol: str, source_symbol: str, api_key: str, timeout: int) -> None:
    try:
        import requests

        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "OVERVIEW", "symbol": source_symbol, "apikey": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not data or data.get("Note") or data.get("Information") or data.get("Error Message"):
            raise FundamentalDataError(str(data.get("Note") or data.get("Information") or data.get("Error Message") or "risposta vuota"))
        queue.put(("ok", _from_alpha_vantage(symbol, source_symbol, data).to_dict()))
    except Exception as e:  # noqa: BLE001
        queue.put(("error", repr(e)))


def _fetch_with_deadline(provider: str, symbol: str, source_symbol: str, config: dict) -> FundamentalSnapshot:
    cfg = config.get("fundamentals", {})
    request_timeout = int(cfg.get("request_timeout_seconds", 8))
    deadline_seconds = int(cfg.get("process_timeout_seconds", 15))
    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    ctx = get_context(start_method)
    queue = ctx.Queue(maxsize=1)

    if provider == "alpha_vantage":
        api_key = os.getenv(str(cfg.get("alpha_vantage_api_key_env", "ALPHA_VANTAGE_API_KEY")), "")
        if not api_key:
            raise FundamentalDataError("ALPHA_VANTAGE_API_KEY non configurata")
        process = ctx.Process(target=_alpha_vantage_worker, args=(queue, symbol, source_symbol, api_key, request_timeout))
    else:
        process = ctx.Process(target=_yfinance_worker, args=(queue, symbol, source_symbol))

    process.start()
    process.join(max(deadline_seconds, request_timeout))

    if process.is_alive():
        process.terminate()
        process.join(2)
        queue.close()
        raise TimeoutError(f"fondamentali oltre {deadline_seconds} secondi")

    try:
        status, payload = queue.get(timeout=1)
    except Empty as e:
        raise RuntimeError(f"download fondamentali terminato senza dati, exit code {process.exitcode}") from e
    finally:
        queue.close()

    if status == "error":
        raise FundamentalDataError(payload)
    return FundamentalSnapshot.from_dict(payload)


def _cache_key(provider: str, source_symbol: str) -> str:
    return f"{provider}:{source_symbol}"


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _is_cache_fresh(snapshot: FundamentalSnapshot, ttl_hours: int) -> bool:
    try:
        fetched_at = datetime.fromisoformat(snapshot.fetched_at)
    except Exception:
        return False
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - fetched_at <= timedelta(hours=ttl_hours)


def _provider(config: dict, source_symbol: str) -> str:
    cfg = config.get("fundamentals", {})
    provider = str(cfg.get("provider", "yfinance"))
    if provider == "auto":
        api_key = os.getenv(str(cfg.get("alpha_vantage_api_key_env", "ALPHA_VANTAGE_API_KEY")), "")
        return "alpha_vantage" if api_key and "." not in source_symbol else "yfinance"
    return provider


def fetch_fundamental_snapshot(
    instrument: dict[str, Any],
    config: dict,
    cache_path: Path | None = None,
) -> FundamentalSnapshot | None:
    cfg = config.get("fundamentals", {})
    if not cfg.get("enabled", False):
        return None
    if instrument.get("type") not in set(cfg.get("apply_to_types", ["stock"])):
        return None

    symbol = str(instrument["symbol"])
    source_symbol = str(instrument.get("fundamental_symbol") or symbol)
    provider = _provider(config, source_symbol)
    ttl_hours = int(cfg.get("cache_ttl_hours", 24))

    if cache_path is not None:
        cache = _load_cache(cache_path)
        cached = cache.get(_cache_key(provider, source_symbol))
        if cached:
            snapshot = FundamentalSnapshot.from_dict(cached)
            if _is_cache_fresh(snapshot, ttl_hours):
                return snapshot
    else:
        cache = {}

    try:
        snapshot = _fetch_with_deadline(provider, symbol, source_symbol, config)
    except Exception as e:  # noqa: BLE001
        snapshot = FundamentalSnapshot(
            symbol=symbol,
            provider=provider,
            source_symbol=source_symbol,
            fetched_at=_now_iso(),
            metrics=_normal_metrics(),
            events=_normal_events(),
            error=str(e),
        )

    if cache_path is not None:
        cache[_cache_key(provider, source_symbol)] = snapshot.to_dict()
        _save_cache(cache_path, cache)

    return snapshot
