from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class BenchmarkRegime:
    symbol: str
    name: str
    state: str
    close: float | None
    sma50: float | None
    sma200: float | None
    change_20d_pct: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "state": self.state,
            "close": self.close,
            "sma50": self.sma50,
            "sma200": self.sma200,
            "change_20d_pct": self.change_20d_pct,
            "reason": self.reason,
        }


@dataclass
class MarketRegime:
    enabled: bool
    state: str
    score: float | None
    base_min_signal_score: float
    active_min_signal_score: float
    new_positions_allowed: bool
    reason: str
    benchmarks: list[BenchmarkRegime] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "score": self.score,
            "base_min_signal_score": self.base_min_signal_score,
            "active_min_signal_score": self.active_min_signal_score,
            "new_positions_allowed": self.new_positions_allowed,
            "reason": self.reason,
            "benchmarks": [benchmark.to_dict() for benchmark in self.benchmarks],
        }


def configured_benchmarks(config: dict) -> list[dict]:
    regime_cfg = config.get("market_regime", {})
    if not regime_cfg.get("enabled", False):
        return []
    return [item for item in regime_cfg.get("benchmarks", []) if item.get("symbol")]


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _latest_clean_slice(df: pd.DataFrame, as_of: date | None) -> pd.DataFrame:
    if df.empty:
        return df
    required_columns = {"Close", "SMA50", "SMA200"}
    if not required_columns.issubset(df.columns):
        return df.iloc[0:0]
    sliced = df
    if as_of is not None:
        sliced = df.loc[: pd.Timestamp(as_of)]
    return sliced.dropna(subset=["Close", "SMA50", "SMA200"])


def _classify_benchmark(symbol: str, name: str, df: pd.DataFrame, as_of: date | None) -> BenchmarkRegime:
    clean = _latest_clean_slice(df, as_of)
    if clean.empty:
        return BenchmarkRegime(
            symbol=symbol,
            name=name,
            state="unknown",
            close=None,
            sma50=None,
            sma200=None,
            change_20d_pct=None,
            reason="dati insufficienti",
        )

    latest = clean.iloc[-1]
    close = _safe_float(latest.get("Close"))
    sma50 = _safe_float(latest.get("SMA50"))
    sma200 = _safe_float(latest.get("SMA200"))
    change_20d_pct = None
    if close is not None and len(clean) > 20:
        close_20d = _safe_float(clean["Close"].iloc[-21])
        if close_20d and close_20d > 0:
            change_20d_pct = round(((close / close_20d) - 1.0) * 100, 2)

    if close is None or sma50 is None or sma200 is None:
        return BenchmarkRegime(
            symbol=symbol,
            name=name,
            state="unknown",
            close=close,
            sma50=sma50,
            sma200=sma200,
            change_20d_pct=change_20d_pct,
            reason="indicatori incompleti",
        )

    if close > sma200 and sma50 > sma200:
        state = "risk_on"
        reason = "prezzo e SMA50 sopra SMA200"
    elif close > sma200 or sma50 > sma200:
        state = "neutral"
        reason = "quadro misto tra prezzo, SMA50 e SMA200"
    else:
        state = "risk_off"
        reason = "prezzo e SMA50 sotto SMA200"

    return BenchmarkRegime(
        symbol=symbol,
        name=name,
        state=state,
        close=round(close, 4),
        sma50=round(sma50, 4),
        sma200=round(sma200, 4),
        change_20d_pct=change_20d_pct,
        reason=reason,
    )


def evaluate_market_regime(
    benchmark_data: dict[str, pd.DataFrame],
    config: dict,
    base_min_signal_score: float,
    as_of: date | None = None,
) -> MarketRegime:
    regime_cfg = config.get("market_regime", {})
    base_score = float(base_min_signal_score)

    if not regime_cfg.get("enabled", False):
        return MarketRegime(
            enabled=False,
            state="disabled",
            score=None,
            base_min_signal_score=base_score,
            active_min_signal_score=base_score,
            new_positions_allowed=True,
            reason="Filtro regime di mercato disattivato.",
        )

    benchmarks: list[BenchmarkRegime] = []
    for item in configured_benchmarks(config):
        symbol = item["symbol"]
        df = benchmark_data.get(symbol)
        if df is None:
            benchmarks.append(
                BenchmarkRegime(
                    symbol=symbol,
                    name=item.get("name", symbol),
                    state="unknown",
                    close=None,
                    sma50=None,
                    sma200=None,
                    change_20d_pct=None,
                    reason="dato benchmark non disponibile",
                )
            )
            continue
        benchmarks.append(_classify_benchmark(symbol, item.get("name", symbol), df, as_of))

    valid_states = [benchmark.state for benchmark in benchmarks if benchmark.state != "unknown"]
    if not valid_states:
        boost = float(regime_cfg.get("unknown_score_boost", 0))
        allow = not bool(regime_cfg.get("block_new_positions_when_unknown", False))
        return MarketRegime(
            enabled=True,
            state="unknown",
            score=None,
            base_min_signal_score=base_score,
            active_min_signal_score=min(100.0, base_score + boost),
            new_positions_allowed=allow,
            reason="Nessun benchmark valido: il filtro resta prudente ma non forza uno stop operativo.",
            benchmarks=benchmarks,
        )

    state_points = {"risk_on": 1.0, "neutral": 0.5, "risk_off": 0.0}
    score = round(sum(state_points[state] for state in valid_states) / len(valid_states), 2)
    risk_on_threshold = float(regime_cfg.get("risk_on_threshold", 0.75))
    neutral_threshold = float(regime_cfg.get("neutral_threshold", 0.45))

    if score >= risk_on_threshold:
        state = "risk_on"
        boost = 0.0
        reason = "Mercato generale favorevole: i benchmark principali sono in trend positivo."
    elif score >= neutral_threshold:
        state = "neutral"
        boost = float(regime_cfg.get("neutral_score_boost", 5))
        reason = "Mercato generale misto: il bot alza la soglia qualità dei segnali."
    else:
        state = "risk_off"
        boost = float(regime_cfg.get("risk_off_score_boost", 15))
        reason = "Mercato generale fragile: il bot protegge il capitale da nuovi ingressi deboli."

    allow = True
    if state == "risk_off" and bool(regime_cfg.get("block_new_positions_when_risk_off", True)):
        allow = False

    return MarketRegime(
        enabled=True,
        state=state,
        score=score,
        base_min_signal_score=base_score,
        active_min_signal_score=min(100.0, base_score + boost),
        new_positions_allowed=allow,
        reason=reason,
        benchmarks=benchmarks,
    )
