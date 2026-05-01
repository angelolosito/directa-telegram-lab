from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategy import Signal


@dataclass(frozen=True)
class RelativeStrengthResult:
    benchmark_symbol: str
    symbol_return_pct: float | None
    benchmark_return_pct: float | None
    relative_strength_pct: float | None
    state: str
    adjustment: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "benchmark_symbol": self.benchmark_symbol,
            "symbol_return_pct": self.symbol_return_pct,
            "benchmark_return_pct": self.benchmark_return_pct,
            "relative_strength_pct": self.relative_strength_pct,
            "state": self.state,
            "adjustment": self.adjustment,
            "reason": self.reason,
        }


def configured_relative_strength_benchmarks(config: dict, watchlist: list[dict] | None = None) -> list[dict]:
    cfg = config.get("relative_strength", {})
    if not cfg.get("enabled", False):
        return []

    symbols: list[str] = []
    default = cfg.get("default_benchmark")
    if default:
        symbols.append(default)
    for symbol in (cfg.get("benchmark_by_type") or {}).values():
        if symbol:
            symbols.append(symbol)
    for symbol in (cfg.get("benchmark_by_region") or {}).values():
        if symbol:
            symbols.append(symbol)
    for instrument in watchlist or []:
        symbol = instrument.get("benchmark")
        if symbol:
            symbols.append(symbol)

    unique_symbols = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)
    return [{"symbol": symbol, "name": symbol} for symbol in unique_symbols]


def benchmark_for_instrument(instrument: dict, config: dict) -> str | None:
    cfg = config.get("relative_strength", {})
    if not cfg.get("enabled", False):
        return None
    instrument_benchmark = instrument.get("benchmark")
    if instrument_benchmark:
        return instrument_benchmark
    region = instrument.get("region")
    by_region = cfg.get("benchmark_by_region") or {}
    if region and by_region.get(region):
        return by_region[region]
    instrument_type = instrument.get("type", "unknown")
    by_type = cfg.get("benchmark_by_type") or {}
    return by_type.get(instrument_type) or cfg.get("default_benchmark")


def _safe_return_pct(df: pd.DataFrame, lookback_sessions: int) -> float | None:
    if df.empty or "Close" not in df.columns:
        return None
    clean = df.dropna(subset=["Close"])
    if len(clean) <= lookback_sessions:
        return None
    latest = float(clean["Close"].iloc[-1])
    previous = float(clean["Close"].iloc[-lookback_sessions - 1])
    if previous <= 0:
        return None
    return round(((latest / previous) - 1.0) * 100.0, 2)


def evaluate_relative_strength(
    instrument: dict,
    instrument_df: pd.DataFrame,
    benchmark_data: dict[str, pd.DataFrame],
    config: dict,
) -> RelativeStrengthResult | None:
    cfg = config.get("relative_strength", {})
    if not cfg.get("enabled", False):
        return None

    benchmark_symbol = benchmark_for_instrument(instrument, config)
    if not benchmark_symbol:
        return None

    benchmark_df = benchmark_data.get(benchmark_symbol)
    if benchmark_df is None:
        return RelativeStrengthResult(
            benchmark_symbol=benchmark_symbol,
            symbol_return_pct=None,
            benchmark_return_pct=None,
            relative_strength_pct=None,
            state="unknown",
            adjustment=0.0,
            reason="benchmark forza relativa non disponibile",
        )

    lookback_sessions = int(cfg.get("lookback_sessions", 60))
    symbol_return = _safe_return_pct(instrument_df, lookback_sessions)
    benchmark_return = _safe_return_pct(benchmark_df, lookback_sessions)
    if symbol_return is None or benchmark_return is None:
        return RelativeStrengthResult(
            benchmark_symbol=benchmark_symbol,
            symbol_return_pct=symbol_return,
            benchmark_return_pct=benchmark_return,
            relative_strength_pct=None,
            state="unknown",
            adjustment=0.0,
            reason="dati insufficienti per forza relativa",
        )

    relative_strength = round(symbol_return - benchmark_return, 2)
    weak_threshold = float(cfg.get("weak_threshold_pct", -2.0))
    strong_threshold = float(cfg.get("strong_threshold_pct", 2.0))
    very_strong_threshold = float(cfg.get("very_strong_threshold_pct", 5.0))
    penalty = float(cfg.get("penalty_points", 8.0))
    bonus = float(cfg.get("bonus_points", 4.0))
    strong_bonus = float(cfg.get("strong_bonus_points", 7.0))

    if relative_strength <= weak_threshold:
        return RelativeStrengthResult(
            benchmark_symbol=benchmark_symbol,
            symbol_return_pct=symbol_return,
            benchmark_return_pct=benchmark_return,
            relative_strength_pct=relative_strength,
            state="weak",
            adjustment=-penalty,
            reason="strumento più debole del benchmark nel periodo recente",
        )
    if relative_strength >= very_strong_threshold:
        return RelativeStrengthResult(
            benchmark_symbol=benchmark_symbol,
            symbol_return_pct=symbol_return,
            benchmark_return_pct=benchmark_return,
            relative_strength_pct=relative_strength,
            state="very_strong",
            adjustment=strong_bonus,
            reason="strumento nettamente più forte del benchmark",
        )
    if relative_strength >= strong_threshold:
        return RelativeStrengthResult(
            benchmark_symbol=benchmark_symbol,
            symbol_return_pct=symbol_return,
            benchmark_return_pct=benchmark_return,
            relative_strength_pct=relative_strength,
            state="strong",
            adjustment=bonus,
            reason="strumento più forte del benchmark",
        )

    return RelativeStrengthResult(
        benchmark_symbol=benchmark_symbol,
        symbol_return_pct=symbol_return,
        benchmark_return_pct=benchmark_return,
        relative_strength_pct=relative_strength,
        state="neutral",
        adjustment=0.0,
        reason="forza relativa in linea con il benchmark",
    )


def apply_relative_strength(
    signal: Signal,
    instrument: dict,
    instrument_df: pd.DataFrame,
    benchmark_data: dict[str, pd.DataFrame],
    config: dict,
) -> Signal:
    if signal.action != "BUY":
        return signal

    result = evaluate_relative_strength(instrument, instrument_df, benchmark_data, config)
    if result is None:
        return signal

    meta = signal.meta or {}
    meta["relative_strength"] = result.to_dict()
    signal.meta = meta

    if result.adjustment:
        signal.score = round(max(0.0, min(100.0, float(signal.score or 0.0) + result.adjustment)), 1)
        signal.score_details = (
            f"{signal.score_details}; forza relativa {result.adjustment:+.1f} "
            f"({result.relative_strength_pct:+.2f}% vs {result.benchmark_symbol})"
        )

    block_weak = bool(config.get("relative_strength", {}).get("block_when_weak", False))
    if result.state == "weak":
        signal.reason += (
            f" Forza relativa debole: {result.relative_strength_pct:+.2f}% "
            f"rispetto a {result.benchmark_symbol}."
        )
        if block_weak:
            signal.action = "WATCH"
            signal.reason += " Segnale sospeso: lo strumento non è tra i leader relativi."
    elif result.state in {"strong", "very_strong"}:
        signal.reason += (
            f" Forza relativa favorevole: {result.relative_strength_pct:+.2f}% "
            f"rispetto a {result.benchmark_symbol}."
        )
    return signal
