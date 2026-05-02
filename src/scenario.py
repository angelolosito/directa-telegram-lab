from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .backtest import BacktestResult, run_backtest


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    changes: dict[str, Any]


@dataclass(frozen=True)
class ScenarioResult:
    spec: ScenarioSpec
    result: BacktestResult
    robustness_score: float
    notes: list[str]


def default_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name="baseline",
            description="Configurazione attuale.",
            changes={},
        ),
        ScenarioSpec(
            name="quality_65",
            description="Alza la qualita minima senza cambiare rischio.",
            changes={"strategy.min_signal_score": 65},
        ),
        ScenarioSpec(
            name="quality_70",
            description="Versione molto selettiva sui segnali.",
            changes={"strategy.min_signal_score": 70},
        ),
        ScenarioSpec(
            name="low_risk_quality",
            description="Rischio piu basso e soglia qualita piu alta.",
            changes={"strategy.min_signal_score": 65, "risk.risk_per_trade": 15},
        ),
        ScenarioSpec(
            name="one_position_defensive",
            description="Una sola posizione aperta, rischio ridotto e piu selezione.",
            changes={
                "strategy.min_signal_score": 65,
                "risk.risk_per_trade": 20,
                "risk.max_open_positions": 1,
            },
        ),
        ScenarioSpec(
            name="balanced_two_positions",
            description="Due posizioni, rischio moderato e qualita sopra baseline.",
            changes={
                "strategy.min_signal_score": 65,
                "risk.risk_per_trade": 20,
                "risk.max_open_positions": 2,
            },
        ),
        ScenarioSpec(
            name="etf_neutral_defensive",
            description="In mercato neutrale preferisce ETF e penalizza le singole azioni.",
            changes={
                "allocation.cautious_etf_bonus": 7,
                "allocation.cautious_stock_penalty": 5,
                "market_regime.neutral_score_boost": 8,
            },
        ),
        ScenarioSpec(
            name="strict_diversification",
            description="Diversificazione piu stretta per area, settore e ruolo.",
            changes={
                "strategy.min_signal_score": 65,
                "allocation.max_same_sector_open": 1,
                "allocation.max_same_role_open": 1,
                "allocation.max_same_region_open": 1,
            },
        ),
        ScenarioSpec(
            name="opportunistic",
            description="Piu permissivo, utile solo per vedere se aumenta troppo il rumore.",
            changes={
                "strategy.min_signal_score": 60,
                "risk.risk_per_trade": 25,
                "market_regime.neutral_score_boost": 3,
            },
        ),
    ]


def _set_path(config: dict, dotted_path: str, value: Any) -> None:
    current = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_scenario(config: dict, spec: ScenarioSpec) -> dict:
    scenario_config = deepcopy(config)
    for dotted_path, value in spec.changes.items():
        _set_path(scenario_config, dotted_path, value)
    return scenario_config


def _months_between(result: BacktestResult) -> float:
    if result.start_date is None or result.end_date is None:
        return 0.0
    days = max((result.end_date - result.start_date).days, 1)
    return max(days / 30.4, 1.0)


def _trade_frequency(result: BacktestResult) -> float:
    months = _months_between(result)
    if months <= 0:
        return 0.0
    return result.closed_trades / months


def _effective_profit_factor(result: BacktestResult) -> float:
    if result.profit_factor is not None:
        return result.profit_factor
    if result.closed_trades > 0 and result.losing_trades == 0 and result.winning_trades > 0:
        return 3.0
    return 0.0


def _scenario_score(result: BacktestResult, config: dict) -> float:
    cfg = config.get("scenario_report", {})
    min_trades = int(cfg.get("min_trades_for_ranking", 5))
    max_monthly_trades = float(cfg.get("max_monthly_trades", 4.0))
    drawdown_weight = float(cfg.get("drawdown_weight", 1.25))
    profit_factor_weight = float(cfg.get("profit_factor_weight", 6.0))
    trade_penalty = float(cfg.get("small_sample_penalty", 2.0))
    overtrade_penalty = float(cfg.get("overtrade_penalty", 2.0))

    profit_factor = min(_effective_profit_factor(result), 3.0)
    frequency = _trade_frequency(result)
    small_sample = max(0, min_trades - result.closed_trades) * trade_penalty
    overtrade = max(0.0, frequency - max_monthly_trades) * overtrade_penalty
    drawdown = abs(min(result.max_drawdown_pct, 0.0))

    score = (
        result.total_return_pct
        + profit_factor * profit_factor_weight
        - drawdown * drawdown_weight
        - small_sample
        - overtrade
    )
    return round(score, 2)


def _scenario_notes(result: BacktestResult, config: dict) -> list[str]:
    cfg = config.get("scenario_report", {})
    min_trades = int(cfg.get("min_trades_for_ranking", 5))
    max_drawdown = float(cfg.get("max_drawdown_warn_pct", 8.0))
    notes: list[str] = []

    if result.closed_trades < min_trades:
        notes.append("campione piccolo")
    if result.max_drawdown_pct <= -max_drawdown:
        notes.append("drawdown alto")
    if result.profit_factor is not None and result.profit_factor < 1.0:
        notes.append("profit factor debole")
    if result.total_return_pct < 0:
        notes.append("rendimento negativo")
    if not notes:
        notes.append("profilo regolare")
    return notes


def run_scenario_grid(
    watchlist: list[dict[str, Any]],
    market_data: dict[str, pd.DataFrame],
    config: dict,
    regime_data: dict[str, pd.DataFrame] | None = None,
    relative_strength_data: dict[str, pd.DataFrame] | None = None,
) -> list[ScenarioResult]:
    cfg = config.get("scenario_report", {})
    max_scenarios = int(cfg.get("max_scenarios", 9))
    results: list[ScenarioResult] = []

    for spec in default_scenarios()[:max_scenarios]:
        scenario_config = apply_scenario(config, spec)
        result = run_backtest(
            watchlist,
            market_data,
            scenario_config,
            regime_data=regime_data,
            relative_strength_data=relative_strength_data,
        )
        results.append(
            ScenarioResult(
                spec=spec,
                result=result,
                robustness_score=_scenario_score(result, scenario_config),
                notes=_scenario_notes(result, scenario_config),
            )
        )
    return sorted(results, key=lambda item: item.robustness_score, reverse=True)


def _fmt(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value:.{decimals}f}"


def _change_lines(spec: ScenarioSpec) -> list[str]:
    if not spec.changes:
        return ["- Nessuna modifica: baseline attuale."]
    return [f"- `{path}` -> `{value}`" for path, value in sorted(spec.changes.items())]


def build_scenario_report(
    watchlist: list[dict[str, Any]],
    market_data: dict[str, pd.DataFrame],
    config: dict,
    regime_data: dict[str, pd.DataFrame] | None = None,
    relative_strength_data: dict[str, pd.DataFrame] | None = None,
    data_errors: list[str] | None = None,
) -> str:
    scenario_results = run_scenario_grid(
        watchlist,
        market_data,
        config,
        regime_data=regime_data,
        relative_strength_data=relative_strength_data,
    )
    best = scenario_results[0] if scenario_results else None

    lines = [
        "# Scenario Report",
        "",
        "Obiettivo: confrontare piu configurazioni sullo stesso storico senza applicare modifiche automatiche.",
        "",
        "## Classifica",
        "",
        "| # | Scenario | Robustezza | Return | Drawdown | PF | Trade | Win rate | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for rank, item in enumerate(scenario_results, start=1):
        result = item.result
        lines.append(
            f"| {rank} | {item.spec.name} | {item.robustness_score:.2f} | "
            f"{result.total_return_pct:.2f}% | {result.max_drawdown_pct:.2f}% | "
            f"{_fmt(result.profit_factor, 2)} | {result.closed_trades} | "
            f"{_fmt(result.win_rate, 1)}% | {', '.join(item.notes)} |"
        )

    if best is not None:
        lines.extend(
            [
                "",
                "## Miglior scenario",
                "",
                f"Scenario: `{best.spec.name}`",
                f"Descrizione: {best.spec.description}",
                f"Robustezza: {best.robustness_score:.2f}",
                "",
                "Modifiche testate:",
                *_change_lines(best.spec),
                "",
                "Lettura prudenziale:",
                "- Non applicare automaticamente: verifica che resti buono anche su altri periodi.",
                "- Preferisci scenari con drawdown controllato, profit factor accettabile e campione sufficiente.",
            ]
        )

    lines.extend(["", "## Dettaglio scenari", ""])
    for item in scenario_results:
        result = item.result
        frequency = _trade_frequency(result)
        lines.extend(
            [
                f"### {item.spec.name}",
                "",
                item.spec.description,
                "",
                f"- Trade chiusi: {result.closed_trades} ({frequency:.2f}/mese)",
                f"- Rendimento: {result.total_return_pct:.2f}%",
                f"- Drawdown: {result.max_drawdown_pct:.2f}%",
                f"- Profit factor: {_fmt(result.profit_factor, 2)}",
                f"- Win rate: {_fmt(result.win_rate, 1)}%",
                "- Modifiche:",
                *_change_lines(item.spec),
                "",
            ]
        )

    if data_errors:
        lines.extend(["## Errori dati", ""])
        for error in data_errors[:20]:
            lines.append(f"- {error}")
        lines.append("")

    lines.append("Nota: scenario report didattico su dati Yahoo Finance, non raccomandazione finanziaria.")
    return "\n".join(lines).strip()
