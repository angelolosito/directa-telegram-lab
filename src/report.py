from __future__ import annotations

from datetime import date
from pathlib import Path

from .strategy import Signal


def _format_optional_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value:.{decimals}f}"


def _cost_pct(signal: Signal) -> float | None:
    if signal.notional <= 0:
        return None
    return (signal.estimated_round_trip_cost / signal.notional) * 100


def _currency(signal: Signal) -> str:
    return str((signal.meta or {}).get("currency") or "EUR")


def _base_currency(signal: Signal) -> str:
    return str((signal.meta or {}).get("base_currency") or "EUR")


def _opportunity(signal: Signal) -> dict:
    return (signal.meta or {}).get("opportunity", {})


def _opportunity_line(signal: Signal) -> str:
    opportunity = _opportunity(signal)
    if not opportunity:
        return ""
    return (
        f"Decisione: {opportunity.get('decision', 'n/d')} "
        f"| Grado: {opportunity.get('grade', 'n/d')} "
        f"| Soglia: {_format_optional_float(opportunity.get('threshold'), 1)}/100\n"
    )


def _learning_feedback(signal: Signal) -> dict:
    feedback = (signal.meta or {}).get("learning_feedback", {})
    return feedback if isinstance(feedback, dict) else {}


def _learning_line(signal: Signal) -> str:
    feedback = _learning_feedback(signal)
    if not feedback:
        return ""
    return (
        f"Feedback diario: {feedback.get('verdict')} "
        f"{_format_optional_float(feedback.get('adjustment'), 1)} punti "
        f"(n={feedback.get('count')}, positivi {feedback.get('positive_rate')}%)\n"
    )


def _relative_strength(signal: Signal) -> dict:
    relative = (signal.meta or {}).get("relative_strength", {})
    return relative if isinstance(relative, dict) else {}


def _relative_strength_line(signal: Signal) -> str:
    relative = _relative_strength(signal)
    if not relative or relative.get("relative_strength_pct") is None:
        return ""
    return (
        f"Forza relativa: {relative.get('state')} "
        f"{relative.get('relative_strength_pct'):+.2f}% vs {relative.get('benchmark_symbol')}\n"
    )


def _fundamentals(signal: Signal) -> dict:
    fundamentals = (signal.meta or {}).get("fundamentals", {})
    return fundamentals if isinstance(fundamentals, dict) else {}


def _fundamentals_line(signal: Signal) -> str:
    fundamentals = _fundamentals(signal)
    if not fundamentals:
        return ""
    score = fundamentals.get("score")
    score_text = _format_optional_float(score, 1) if score is not None else "n/d"
    state = fundamentals.get("state", "n/d")
    source = fundamentals.get("source", "n/d")
    events = fundamentals.get("events") or {}
    earnings = ""
    if isinstance(events, dict) and events.get("next_earnings_date"):
        earnings = f" | trimestrale: {events.get('next_earnings_date')}"
    elif isinstance(events, dict) and events.get("latest_quarter"):
        earnings = f" | ultimo trimestre: {events.get('latest_quarter')}"
    return f"Fondamentali: {state} {score_text}/100 | fonte {source}{earnings}\n"


def _allocation(signal: Signal) -> dict:
    allocation = (signal.meta or {}).get("allocation", {})
    return allocation if isinstance(allocation, dict) else {}


def _allocation_line(signal: Signal) -> str:
    allocation = _allocation(signal)
    if not allocation:
        return ""
    decision = allocation.get("decision", "n/d")
    score = _format_optional_float(allocation.get("score"), 1)
    reason = allocation.get("reason", "")
    return f"Allocazione: {decision} | score finale {score}/100 | {reason}\n"


def format_signal(signal: Signal) -> str:
    if signal.action == "BUY":
        currency = _currency(signal)
        base_currency = _base_currency(signal)
        return (
            f"🟢 <b>POSSIBILE ACQUISTO PAPER</b>\n"
            f"<b>{signal.name}</b> ({signal.symbol})\n"
            f"Strategia: {signal.strategy}\n"
            f"Score: {_format_optional_float(signal.score, 1)}/100\n"
            f"{_opportunity_line(signal)}"
            f"{_learning_line(signal)}"
            f"{_relative_strength_line(signal)}"
            f"{_fundamentals_line(signal)}"
            f"{_allocation_line(signal)}"
            f"Prezzo: {signal.price:.4f} {currency}\n"
            f"Entry simulata: {signal.entry:.4f} {currency}\n"
            f"Stop: {signal.stop:.4f} {currency}\n"
            f"Target: {signal.target:.4f} {currency}\n"
            f"R/R: {signal.reward_risk}\n"
            f"Qty simulata: {signal.qty}\n"
            f"Controvalore: {signal.notional:.2f} {base_currency}\n"
            f"Costi round-trip stimati: {signal.estimated_round_trip_cost:.2f} {base_currency} "
            f"({_format_optional_float(_cost_pct(signal), 2)}%)\n"
            f"Dettaglio score: {signal.score_details}\n"
            f"Motivo: {signal.reason}"
        )
    if signal.action == "WATCH":
        score = f"\nScore: {_format_optional_float(signal.score, 1)}/100" if signal.score is not None else ""
        return f"🟡 <b>WATCH</b> {signal.name} ({signal.symbol}){score}\n{signal.reason}"
    if signal.action == "ERROR":
        return f"⚠️ <b>ERRORE</b> {signal.name} ({signal.symbol})\n{signal.reason}"
    return f"⚪ <b>HOLD</b> {signal.name} ({signal.symbol})\nPrezzo: {signal.price}\n{signal.reason}"


def format_candidate_signal(signal: Signal, rank: int) -> str:
    status = "operativo" if signal.action == "BUY" else "watch"
    opportunity = _opportunity(signal)
    decision = (
        f" | {opportunity.get('decision')}/{opportunity.get('grade')}"
        if opportunity
        else ""
    )
    feedback = _learning_feedback(signal)
    learning = (
        f" | diario {feedback.get('adjustment'):+.1f}"
        if feedback.get("adjustment") is not None
        else ""
    )
    relative = _relative_strength(signal)
    relative_text = (
        f" | RS {relative.get('relative_strength_pct'):+.1f}%"
        if relative.get("relative_strength_pct") is not None
        else ""
    )
    fundamentals = _fundamentals(signal)
    fundamentals_text = ""
    if fundamentals:
        score = fundamentals.get("score")
        score_text = _format_optional_float(score, 1) if score is not None else "n/d"
        fundamentals_text = f" | FND {fundamentals.get('state', 'n/d')} {score_text}"
    allocation = _allocation(signal)
    allocation_text = ""
    if allocation:
        label = "scelto" if allocation.get("decision") == "SELECTED" else "scartato"
        allocation_text = f" | alloc {label}"
    return (
        f"{rank}. <b>{signal.symbol}</b> {signal.name} | "
        f"score {_format_optional_float(signal.score, 1)}/100 | "
        f"{signal.strategy} | R/R {_format_optional_float(signal.reward_risk, 2)} | "
        f"costi {_format_optional_float(_cost_pct(signal), 2)}% | {status}{decision}{learning}{relative_text}"
        f"{fundamentals_text}{allocation_text}"
    )


def format_close_event(event: dict) -> str:
    emoji = "🔴" if event.get("net_pnl", 0) < 0 else "🟢"
    return (
        f"{emoji} <b>CHIUSURA PAPER</b>\n"
        f"<b>{event.get('name', event.get('symbol'))}</b> ({event.get('symbol')})\n"
        f"Motivo: {event.get('reason')}\n"
        f"Entry: {event.get('entry_price'):.4f} €\n"
        f"Exit: {event.get('exit_price'):.4f} €\n"
        f"Qty: {event.get('qty')}\n"
        f"P/L lordo: {event.get('gross_pnl'):.2f} €\n"
        f"P/L netto stimato: {event.get('net_pnl'):.2f} €"
    )


def build_daily_message(
    run_date: date,
    buy_signals: list[Signal],
    candidate_signals: list[Signal],
    close_events: list[dict],
    trail_events: list[dict],
    summary: dict,
    errors: list[str],
    dry_run: bool = False,
    market_regime: dict | None = None,
    signal_learning: dict | None = None,
    allocation: dict | None = None,
) -> str:
    lines: list[str] = [
        f"📊 <b>Directa Telegram Trading Lab</b>",
        f"Data: {run_date.isoformat()}",
    ]
    if dry_run:
        lines.append("Modalità: DRY-RUN, nessuna modifica salvata.")
    if market_regime and market_regime.get("enabled"):
        allowed = "sì" if market_regime.get("new_positions_allowed") else "no"
        lines.extend(
            [
                "",
                "<b>Regime di mercato</b>",
                f"Stato: {market_regime.get('state', 'n/d')}",
                f"Soglia score attiva: {_format_optional_float(market_regime.get('active_min_signal_score'), 1)}/100",
                f"Nuovi ingressi permessi: {allowed}",
                str(market_regime.get("reason", "")),
            ]
        )
        benchmarks = market_regime.get("benchmarks") or []
        for benchmark in benchmarks[:3]:
            change_20d = _format_optional_float(benchmark.get("change_20d_pct"), 2)
            lines.append(
                f"- {benchmark.get('symbol')}: {benchmark.get('state')} "
                f"(20 sedute {change_20d}%)"
            )
    lines.extend(
        [
            "",
            f"Cash paper: {summary.get('cash', 0):.2f} €",
            f"Valore posizioni: {summary.get('open_market_value', 0):.2f} €",
            f"Equity stimata: {summary.get('equity', 0):.2f} €",
            f"Posizioni aperte: {summary.get('open_positions', 0)}",
            f"P/L realizzato netto stimato: {summary.get('realized_pnl', 0):.2f} €",
            f"P/L non realizzato netto stimato: {summary.get('unrealized_pnl', 0):.2f} €",
            f"P/L totale stimato: {summary.get('total_pnl', 0):.2f} € ({summary.get('total_return_pct', 0):.2f}%)",
            f"Rischio aperto fino agli stop: {summary.get('open_risk_to_stop', 0):.2f} €",
            "",
        ]
    )

    if summary.get("closed_trades", 0):
        lines.extend(
            [
                "<b>Statistiche laboratorio</b>",
                f"Trade chiusi: {summary.get('closed_trades', 0)}",
                f"Win rate: {_format_optional_float(summary.get('win_rate'), 1)}%",
                f"Profit factor: {_format_optional_float(summary.get('profit_factor'), 2)}",
                f"P/L medio per trade: {_format_optional_float(summary.get('avg_trade_pnl'), 2)} €",
                f"Migliore/peggiore: {_format_optional_float(summary.get('best_trade_pnl'), 2)} € / "
                f"{_format_optional_float(summary.get('worst_trade_pnl'), 2)} €",
                "",
            ]
        )

    if signal_learning and signal_learning.get("completed", 0):
        lines.extend(
            [
                "<b>Diario intelligente segnali</b>",
                f"Segnali in memoria: {signal_learning.get('journal_size', 0)}",
                f"Nuove valutazioni aggiornate: {signal_learning.get('new_or_updated', 0)}",
                f"Orizzonte principale: {signal_learning.get('primary_horizon')} sedute",
                f"Valutazioni complete: {signal_learning.get('completed', 0)}",
                f"Tasso positivi: {_format_optional_float(signal_learning.get('positive_rate'), 1)}%",
                f"Rendimento medio: {_format_optional_float(signal_learning.get('avg_close_return_pct'), 2)}%",
                "",
            ]
        )
        if signal_learning.get("best_bucket"):
            lines.append(f"Setup migliori finora: {signal_learning.get('best_bucket')}")
        if signal_learning.get("weak_bucket"):
            lines.append(f"Setup più deboli finora: {signal_learning.get('weak_bucket')}")
        if signal_learning.get("best_bucket") or signal_learning.get("weak_bucket"):
            lines.append("")

    if allocation and allocation.get("enabled"):
        selected = ", ".join(allocation.get("selected_symbols") or []) or "nessuno"
        lines.extend(
            [
                "<b>Selettore portafoglio</b>",
                f"Candidati operativi: {allocation.get('candidates', 0)}",
                f"Selezionati: {selected}",
                f"Scartati per diversificazione/allocazione: {allocation.get('rejected', 0)}",
                f"Motivo: {allocation.get('reason', 'n/d')}",
                "",
            ]
        )
        for reason, count in allocation.get("top_rejections", [])[:3]:
            lines.append(f"- {reason}: {count}")
        if allocation.get("top_rejections"):
            lines.append("")

    if close_events:
        lines.append("<b>Uscite / vendite simulate</b>")
        for event in close_events:
            lines.append(format_close_event(event))
            lines.append("")

    if trail_events:
        lines.append("<b>Aggiornamenti trailing stop</b>")
        for event in trail_events:
            lines.append(f"🔧 {event.get('message')}")
        lines.append("")

    if buy_signals:
        lines.append("<b>Segnale selezionato</b>")
        for signal in buy_signals:
            lines.append(format_signal(signal))
            lines.append("")
    else:
        lines.append("Nessun nuovo segnale operativo oggi.")
        lines.append("")

    if candidate_signals:
        lines.append("<b>Classifica candidati</b>")
        for rank, signal in enumerate(candidate_signals[:5], start=1):
            lines.append(format_candidate_signal(signal, rank))
        lines.append("")

    if errors:
        lines.append("<b>Errori dati</b>")
        for err in errors[:10]:
            lines.append(f"⚠️ {err}")

    lines.append("Nota: segnali didattici/paper trading, non ordini reali.")
    return "\n".join(lines).strip()


def save_markdown_report(
    reports_dir: Path,
    run_date: date,
    message_html: str,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    text = message_html.replace("<b>", "**").replace("</b>", "**")
    path = reports_dir / f"report_{run_date.isoformat()}.md"
    path.write_text(text, encoding="utf-8")
    return path
