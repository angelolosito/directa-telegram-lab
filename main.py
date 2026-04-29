from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import load_config, load_watchlist
from src.data_provider import DataProviderError, fetch_daily_data
from src.paper_portfolio import PaperPortfolio
from src.report import build_daily_message, save_markdown_report
from src.strategy import Signal, analyze_buy_signals
from src.telegram_notifier import TelegramNotifier


def append_signals_csv(path: Path, signals: list[Signal]) -> None:
    if not signals:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "symbol",
                "name",
                "instrument_type",
                "action",
                "strategy",
                "price",
                "entry",
                "stop",
                "target",
                "reward_risk",
                "qty",
                "notional",
                "estimated_round_trip_cost",
                "reason",
            ],
        )
        if not exists:
            writer.writeheader()
        for s in signals:
            writer.writerow(
                {
                    "date": s.date,
                    "symbol": s.symbol,
                    "name": s.name,
                    "instrument_type": s.instrument_type,
                    "action": s.action,
                    "strategy": s.strategy,
                    "price": s.price,
                    "entry": s.entry,
                    "stop": s.stop,
                    "target": s.target,
                    "reward_risk": s.reward_risk,
                    "qty": s.qty,
                    "notional": s.notional,
                    "estimated_round_trip_cost": s.estimated_round_trip_cost,
                    "reason": s.reason,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Directa Telegram Trading Lab")
    parser.add_argument("--base-dir", default=".", help="Project base directory")
    parser.add_argument("--dry-run", action="store_true", help="Do not open new paper positions and do not send Telegram")
    parser.add_argument("--send-test", action="store_true", help="Send only a Telegram test message")
    args = parser.parse_args()

    app = load_config(args.base_dir)
    cfg = app.raw
    timezone = cfg["project"].get("timezone", "Europe/Rome")
    today = datetime.now(ZoneInfo(timezone)).date()

    notifier = TelegramNotifier()

    if args.send_test:
        notifier.send("✅ Test Directa Telegram Trading Lab riuscito.")
        return 0

    watchlist = load_watchlist(app.base_dir)
    portfolio = PaperPortfolio(app.database_path, cfg)

    market_data = {}
    errors: list[str] = []
    min_rows = int(cfg["run"].get("min_rows_required", 220))
    lookback_days = int(cfg["run"].get("lookback_days", 430))

    for instrument in watchlist:
        symbol = instrument["symbol"]
        try:
            df = fetch_daily_data(symbol, lookback_days=lookback_days, timezone=timezone)
            if len(df.dropna(subset=["Close", "SMA200"])) < min_rows:
                errors.append(f"{symbol}: storico insufficiente dopo calcolo SMA200.")
                continue
            market_data[symbol] = df
        except DataProviderError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{symbol}: errore imprevisto: {e}")

    close_and_trail_events = portfolio.update_open_positions(market_data, today)
    close_events = [e for e in close_and_trail_events if e.get("type") == "CLOSE"]
    trail_events = [e for e in close_and_trail_events if e.get("type") == "TRAIL_UPDATE"]

    actionable_buy_signals: list[Signal] = []
    all_logged_signals: list[Signal] = []

    for instrument in watchlist:
        symbol = instrument["symbol"]
        df = market_data.get(symbol)
        if df is None:
            continue
        signals = analyze_buy_signals(instrument, df, cfg["strategy"], today)
        for signal in signals:
            if signal.action != "BUY":
                continue
            signal = portfolio.size_signal(signal)
            can_open, reason = portfolio.can_open_new_position(signal, today)
            if signal.qty <= 0:
                signal.action = "WATCH"
                signal.reason += " Segnale non eseguito in paper: capitale/size non sufficiente con i limiti attuali."
                all_logged_signals.append(signal)
                continue
            if not can_open:
                signal.action = "WATCH"
                signal.reason += f" Segnale non eseguito in paper: {reason}"
                all_logged_signals.append(signal)
                continue
            actionable_buy_signals.append(signal)
            all_logged_signals.append(signal)

    # Priorità: massimo una nuova posizione per run, scegliendo il rapporto rischio/rendimento migliore e poi il minor costo stimato.
    actionable_buy_signals.sort(key=lambda s: (s.reward_risk or 0, -s.estimated_round_trip_cost), reverse=True)
    opened_signals: list[Signal] = []
    if actionable_buy_signals:
        best_signal = actionable_buy_signals[0]
        if not args.dry_run:
            opened = portfolio.open_position(best_signal)
            if opened:
                opened_signals.append(best_signal)
        else:
            opened_signals.append(best_signal)

    append_signals_csv(app.signals_csv, all_logged_signals)
    summary = portfolio.summary()
    message = build_daily_message(
        run_date=today,
        buy_signals=opened_signals,
        close_events=close_events,
        trail_events=trail_events,
        summary=summary,
        errors=errors,
    )

    if cfg["run"].get("save_reports", True):
        save_markdown_report(app.reports_dir, today, message)

    should_send = bool(cfg["run"].get("send_telegram", True)) and not args.dry_run
    if should_send and notifier.enabled:
        notifier.send(message)
    else:
        print(message)
        if should_send and not notifier.enabled:
            print("\nTelegram non configurato: imposto solo output console.")

    portfolio.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
