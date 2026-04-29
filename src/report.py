from __future__ import annotations

from datetime import date
from pathlib import Path

from .strategy import Signal


def format_signal(signal: Signal) -> str:
    if signal.action == "BUY":
        return (
            f"🟢 <b>POSSIBILE ACQUISTO PAPER</b>\n"
            f"<b>{signal.name}</b> ({signal.symbol})\n"
            f"Strategia: {signal.strategy}\n"
            f"Prezzo: {signal.price:.4f} €\n"
            f"Entry simulata: {signal.entry:.4f} €\n"
            f"Stop: {signal.stop:.4f} €\n"
            f"Target: {signal.target:.4f} €\n"
            f"R/R: {signal.reward_risk}\n"
            f"Qty simulata: {signal.qty}\n"
            f"Controvalore: {signal.notional:.2f} €\n"
            f"Costi round-trip stimati: {signal.estimated_round_trip_cost:.2f} €\n"
            f"Motivo: {signal.reason}"
        )
    if signal.action == "WATCH":
        return f"🟡 <b>WATCH</b> {signal.name} ({signal.symbol})\n{signal.reason}"
    if signal.action == "ERROR":
        return f"⚠️ <b>ERRORE</b> {signal.name} ({signal.symbol})\n{signal.reason}"
    return f"⚪ <b>HOLD</b> {signal.name} ({signal.symbol})\nPrezzo: {signal.price}\n{signal.reason}"


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
    close_events: list[dict],
    trail_events: list[dict],
    summary: dict,
    errors: list[str],
) -> str:
    lines: list[str] = [
        f"📊 <b>Directa Telegram Trading Lab</b>",
        f"Data: {run_date.isoformat()}",
        "",
        f"Cash paper: {summary.get('cash', 0):.2f} €",
        f"Posizioni aperte: {summary.get('open_positions', 0)}",
        f"P/L realizzato netto stimato: {summary.get('realized_pnl', 0):.2f} €",
        "",
    ]

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
        lines.append("<b>Nuovi segnali</b>")
        for signal in buy_signals:
            lines.append(format_signal(signal))
            lines.append("")
    else:
        lines.append("Nessun nuovo segnale operativo oggi.")
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
