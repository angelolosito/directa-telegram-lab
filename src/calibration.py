from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .backtest import BacktestResult, BacktestTrade


@dataclass(frozen=True)
class BucketStats:
    key: str
    count: int
    net_pnl: float
    win_rate: float | None
    avg_pnl: float | None


def _fmt_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value:.{decimals}f}"


def _months_between(result: BacktestResult) -> float:
    if result.start_date is None or result.end_date is None:
        return 0.0
    days = max((result.end_date - result.start_date).days, 1)
    return max(days / 30.4, 1.0)


def _bucket_value(trade: BacktestTrade, key: str) -> str:
    if key == "strategy":
        return trade.strategy or "unknown"
    if key == "instrument_type":
        return trade.instrument_type or "unknown"
    value = (trade.meta or {}).get(key)
    if value is None or value == "":
        return "unknown"
    return str(value)


def _bucket_stats(trades: list[BacktestTrade], key: str, min_count: int = 1) -> list[BucketStats]:
    groups: dict[str, list[BacktestTrade]] = defaultdict(list)
    for trade in trades:
        groups[_bucket_value(trade, key)].append(trade)

    stats: list[BucketStats] = []
    for bucket, bucket_trades in groups.items():
        if len(bucket_trades) < min_count:
            continue
        values = [trade.net_pnl for trade in bucket_trades]
        wins = [value for value in values if value > 0]
        win_rate = (len(wins) / len(values)) * 100 if values else None
        avg_pnl = sum(values) / len(values) if values else None
        stats.append(
            BucketStats(
                key=bucket,
                count=len(values),
                net_pnl=round(sum(values), 2),
                win_rate=round(win_rate, 1) if win_rate is not None else None,
                avg_pnl=round(avg_pnl, 2) if avg_pnl is not None else None,
            )
        )
    return sorted(stats, key=lambda item: (item.net_pnl, item.avg_pnl or 0.0), reverse=True)


def _trade_frequency(result: BacktestResult) -> float:
    months = _months_between(result)
    if months <= 0:
        return 0.0
    return round(result.closed_trades / months, 2)


def _recommendations(result: BacktestResult, config: dict) -> list[str]:
    calibration_cfg = config.get("calibration", {})
    min_trades = int(calibration_cfg.get("min_trades_for_confidence", 8))
    max_drawdown_warn = float(calibration_cfg.get("max_drawdown_warn_pct", 8.0))
    min_profit_factor = float(calibration_cfg.get("min_profit_factor", 1.15))
    min_monthly_trades = float(calibration_cfg.get("min_monthly_trades", 0.4))
    max_monthly_trades = float(calibration_cfg.get("max_monthly_trades", 4.0))
    frequency = _trade_frequency(result)

    recommendations: list[str] = []
    if result.closed_trades < min_trades:
        recommendations.append(
            f"Campione ancora piccolo: {result.closed_trades} trade chiusi. "
            "Prima di cambiare soglie in modo aggressivo conviene accumulare piu osservazioni."
        )
    if frequency < min_monthly_trades:
        recommendations.append(
            f"Operativita molto selettiva: circa {frequency:.2f} trade/mese. "
            "Se il bot resta troppo fermo, valuta una soglia score leggermente piu bassa o piu capitale per posizione."
        )
    elif frequency > max_monthly_trades:
        recommendations.append(
            f"Operativita intensa: circa {frequency:.2f} trade/mese. "
            "Valuta soglie piu alte o vincoli di allocazione piu severi."
        )

    if result.profit_factor is not None and result.profit_factor < min_profit_factor:
        recommendations.append(
            f"Profit factor sotto soglia ({result.profit_factor}). "
            "Il bot deve diventare piu selettivo su setup con costi elevati o forza relativa debole."
        )
    if result.max_drawdown_pct <= -max_drawdown_warn:
        recommendations.append(
            f"Drawdown rilevante ({result.max_drawdown_pct:.2f}%). "
            "Riduci rischio per trade o aumenta la soglia minima nei regimi neutrali."
        )
    if result.win_rate is not None and result.win_rate < 45:
        recommendations.append(
            f"Win rate basso ({result.win_rate:.1f}%). "
            "Rafforza filtri di ingresso, soprattutto su breakout estesi e settori gia coperti."
        )
    if result.closed_trades and result.avg_trade_pnl is not None and result.avg_trade_pnl <= 0:
        recommendations.append(
            f"P/L medio non positivo ({result.avg_trade_pnl:.2f} EUR). "
            "La qualita media dei segnali non compensa costi e stop."
        )
    if not recommendations:
        recommendations.append(
            "Nessun allarme principale: mantieni la configurazione e continua a raccogliere dati paper."
        )
    return recommendations


def _format_bucket_section(title: str, buckets: list[BucketStats], limit: int = 6) -> list[str]:
    lines = ["", f"## {title}", ""]
    if not buckets:
        lines.append("- Dati insufficienti.")
        return lines
    for bucket in buckets[:limit]:
        lines.append(
            f"- {bucket.key}: trade {bucket.count}, P/L {bucket.net_pnl:.2f} EUR, "
            f"win rate {_fmt_float(bucket.win_rate, 1)}%, media {_fmt_float(bucket.avg_pnl, 2)} EUR"
        )
    return lines


def build_calibration_report(
    result: BacktestResult,
    watchlist: list[dict[str, Any]],
    config: dict,
) -> str:
    frequency = _trade_frequency(result)
    min_bucket_count = int(config.get("calibration", {}).get("min_bucket_count", 2))
    active_symbols = {trade.symbol for trade in result.trades}
    unused_symbols = [item["symbol"] for item in watchlist if item.get("symbol") not in active_symbols]

    lines = [
        "# Calibration Report",
        "",
        "Obiettivo: capire se il bot e troppo severo, troppo permissivo o troppo concentrato.",
        "",
        "## Sintesi",
        "",
        f"Periodo: {result.start_date or 'n/d'} -> {result.end_date or 'n/d'}",
        f"Strumenti in watchlist: {len(watchlist)}",
        f"Strumenti che hanno generato trade chiusi: {len(active_symbols)}",
        f"Trade chiusi: {result.closed_trades}",
        f"Trade/mese stimati: {frequency:.2f}",
        f"Win rate: {_fmt_float(result.win_rate, 1)}%",
        f"Profit factor: {_fmt_float(result.profit_factor, 2)}",
        f"P/L medio: {_fmt_float(result.avg_trade_pnl, 2)} EUR",
        f"Rendimento totale: {result.total_return_pct:.2f}%",
        f"Max drawdown: {result.max_drawdown_pct:.2f}%",
        f"Posizioni ancora aperte: {len(result.open_positions)}",
    ]

    lines.extend(_format_bucket_section("Strategie", _bucket_stats(result.trades, "strategy", min_bucket_count)))
    lines.extend(_format_bucket_section("Aree geografiche", _bucket_stats(result.trades, "region", min_bucket_count)))
    lines.extend(_format_bucket_section("Settori", _bucket_stats(result.trades, "sector", min_bucket_count)))
    lines.extend(_format_bucket_section("Tipi strumento", _bucket_stats(result.trades, "instrument_type", 1)))

    lines.extend(["", "## Raccomandazioni", ""])
    for recommendation in _recommendations(result, config):
        lines.append(f"- {recommendation}")

    if unused_symbols:
        preview = ", ".join(unused_symbols[:12])
        suffix = "..." if len(unused_symbols) > 12 else ""
        lines.extend(
            [
                "",
                "## Watchlist silenziosa",
                "",
                f"- {len(unused_symbols)} strumenti non hanno generato trade chiusi nel periodo.",
                f"- Esempi: {preview}{suffix}",
                "- Non e automaticamente un problema: possono restare come radar, ma vanno osservati nel tempo.",
            ]
        )

    if result.regime_counts:
        lines.extend(["", "## Regime mercato", ""])
        total_days = sum(result.regime_counts.values()) or 1
        for state, count in sorted(result.regime_counts.items()):
            pct = (count / total_days) * 100
            lines.append(f"- {state}: {count} sedute ({pct:.1f}%)")

    if result.errors:
        lines.extend(["", "## Errori dati", ""])
        for error in result.errors[:20]:
            lines.append(f"- {error}")

    lines.append("")
    lines.append("Nota: calibrazione didattica su dati Yahoo Finance, non raccomandazione finanziaria.")
    return "\n".join(lines)
