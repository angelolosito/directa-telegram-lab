from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.allocation import select_portfolio_candidates
from src.backtest import format_backtest_report, run_backtest
from src.calibration import build_calibration_report
from src.config import load_config, load_watchlist
from src.currency import (
    configured_currency_pairs,
    enrich_watchlist_with_fx,
    latest_fx_rates,
)
from src.data_provider import DataProviderError, fetch_daily_data
from src.learning_feedback import apply_learning_feedback, load_learning_stats
from src.market_regime import configured_benchmarks, evaluate_market_regime
from src.opportunity import review_opportunity
from src.paper_portfolio import PaperPortfolio
from src.relative_strength import apply_relative_strength, configured_relative_strength_benchmarks
from src.report import build_daily_message, save_markdown_report
from src.signal_journal import append_signal_journal, build_learning_report, update_signal_evaluations
from src.strategy import Signal, analyze_buy_signals, score_signal
from src.telegram_notifier import TelegramNotifier


SIGNAL_CSV_FIELDNAMES = [
    "date",
    "symbol",
    "name",
    "instrument_type",
    "action",
    "strategy",
    "score",
    "score_details",
    "price",
    "entry",
    "stop",
    "target",
    "reward_risk",
    "qty",
    "notional",
    "estimated_round_trip_cost",
    "reason",
]


def append_signals_csv(path: Path, signals: list[Signal]) -> None:
    if not signals:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames != SIGNAL_CSV_FIELDNAMES:
                with path.open("w", newline="", encoding="utf-8") as out:
                    writer = csv.DictWriter(out, fieldnames=SIGNAL_CSV_FIELDNAMES)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({field: row.get(field, "") for field in SIGNAL_CSV_FIELDNAMES})
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_CSV_FIELDNAMES)
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
                    "score": s.score,
                    "score_details": s.score_details,
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


def fetch_market_regime_data(
    cfg: dict,
    known_market_data: dict,
    timezone: str,
    lookback_days: int,
    request_timeout: int,
    download_retries: int,
    process_timeout: int,
) -> tuple[dict, list[str]]:
    regime_data = {}
    errors: list[str] = []
    benchmarks = configured_benchmarks(cfg)
    if not benchmarks:
        return regime_data, errors

    regime_cfg = cfg.get("market_regime", {})
    regime_lookback_days = int(regime_cfg.get("lookback_days", max(lookback_days, 320)))
    min_rows = int(regime_cfg.get("min_rows_required", 220))

    for benchmark in benchmarks:
        symbol = benchmark["symbol"]
        if symbol in known_market_data:
            df = known_market_data[symbol]
            if len(df.dropna(subset=["Close", "SMA200"])) < min_rows:
                errors.append(f"{symbol}: storico insufficiente per filtro regime mercato.")
            else:
                regime_data[symbol] = df
            continue
        try:
            df = fetch_daily_data(
                symbol,
                lookback_days=regime_lookback_days,
                timezone=timezone,
                request_timeout=request_timeout,
                retries=download_retries,
                process_timeout=process_timeout,
            )
            if len(df.dropna(subset=["Close", "SMA200"])) < min_rows:
                errors.append(f"{symbol}: storico insufficiente per filtro regime mercato.")
                continue
            regime_data[symbol] = df
        except DataProviderError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{symbol}: errore imprevisto nel filtro regime mercato: {e}")
    return regime_data, errors


def fetch_relative_strength_data(
    cfg: dict,
    watchlist: list[dict],
    known_market_data: dict,
    timezone: str,
    lookback_days: int,
    request_timeout: int,
    download_retries: int,
    process_timeout: int,
) -> tuple[dict, list[str]]:
    relative_data = {}
    errors: list[str] = []
    benchmarks = configured_relative_strength_benchmarks(cfg, watchlist)
    if not benchmarks:
        return relative_data, errors

    relative_cfg = cfg.get("relative_strength", {})
    relative_lookback_days = int(relative_cfg.get("lookback_days", max(lookback_days, 220)))
    min_rows = int(relative_cfg.get("min_rows_required", int(relative_cfg.get("lookback_sessions", 60)) + 1))

    for benchmark in benchmarks:
        symbol = benchmark["symbol"]
        if symbol in known_market_data:
            df = known_market_data[symbol]
            if len(df.dropna(subset=["Close"])) < min_rows:
                errors.append(f"{symbol}: storico insufficiente per forza relativa.")
            else:
                relative_data[symbol] = df
            continue
        try:
            df = fetch_daily_data(
                symbol,
                lookback_days=relative_lookback_days,
                timezone=timezone,
                request_timeout=request_timeout,
                retries=download_retries,
                process_timeout=process_timeout,
            )
            if len(df.dropna(subset=["Close"])) < min_rows:
                errors.append(f"{symbol}: storico insufficiente per forza relativa.")
                continue
            relative_data[symbol] = df
        except DataProviderError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{symbol}: errore imprevisto nella forza relativa: {e}")
    return relative_data, errors


def fetch_currency_data(
    cfg: dict,
    watchlist: list[dict],
    timezone: str,
    request_timeout: int,
    download_retries: int,
    process_timeout: int,
) -> tuple[dict, list[str]]:
    fx_data = {}
    errors: list[str] = []
    pairs = configured_currency_pairs(watchlist, cfg)
    if not pairs:
        return fx_data, errors

    currency_cfg = cfg.get("currency", {})
    lookback_days = int(currency_cfg.get("lookback_days", 14))

    for pair in pairs:
        symbol = pair["symbol"]
        try:
            df = fetch_daily_data(
                symbol,
                lookback_days=lookback_days,
                timezone=timezone,
                request_timeout=request_timeout,
                retries=download_retries,
                process_timeout=process_timeout,
            )
            fx_data[symbol] = df
        except DataProviderError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{symbol}: errore imprevisto nel cambio valuta: {e}")

    return fx_data, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Directa Telegram Trading Lab")
    parser.add_argument("--base-dir", default=".", help="Project base directory")
    parser.add_argument("--dry-run", action="store_true", help="Do not open new paper positions and do not send Telegram")
    parser.add_argument("--send-test", action="store_true", help="Send only a Telegram test message")
    parser.add_argument("--backtest", action="store_true", help="Run a historical paper backtest and exit")
    parser.add_argument("--backtest-days", type=int, default=None, help="Override backtest lookback days")
    parser.add_argument("--calibration-report", action="store_true", help="Run backtest diagnostics and tuning report")
    parser.add_argument("--learning-report", action="store_true", help="Print the signal learning journal report and exit")
    args = parser.parse_args()

    app = load_config(args.base_dir)
    cfg = app.raw
    timezone = cfg["project"].get("timezone", "Europe/Rome")
    today = datetime.now(ZoneInfo(timezone)).date()

    notifier = TelegramNotifier()

    if args.send_test:
        notifier.send("✅ Test Directa Telegram Trading Lab riuscito.")
        return 0

    if args.learning_report:
        report = build_learning_report(app.signal_journal_csv, app.signal_evaluations_csv, cfg)
        if cfg["run"].get("save_reports", True):
            app.reports_dir.mkdir(parents=True, exist_ok=True)
            path = app.reports_dir / f"learning_{today.isoformat()}.md"
            path.write_text(report, encoding="utf-8")
        print(report)
        return 0

    watchlist = load_watchlist(app.base_dir)
    data_cfg = cfg.get("data", {})
    request_timeout = int(data_cfg.get("request_timeout_seconds", 8))
    download_retries = int(data_cfg.get("download_retries", 1))
    process_timeout = int(data_cfg.get("process_timeout_seconds", max(request_timeout, 20)))

    if args.backtest or args.calibration_report:
        backtest_cfg = cfg.get("backtest", {})
        lookback_days = int(
            args.backtest_days
            or backtest_cfg.get("lookback_days", max(int(cfg["run"].get("lookback_days", 430)), 900))
        )
        min_rows = int(backtest_cfg.get("min_rows_required", cfg["run"].get("min_rows_required", 220)))
        market_data = {}
        errors: list[str] = []

        for instrument in watchlist:
            symbol = instrument["symbol"]
            try:
                df = fetch_daily_data(
                    symbol,
                    lookback_days=lookback_days,
                    timezone=timezone,
                    request_timeout=request_timeout,
                    retries=download_retries,
                    process_timeout=process_timeout,
                )
                if len(df.dropna(subset=["Close", "SMA200"])) < min_rows:
                    errors.append(f"{symbol}: storico insufficiente per backtest.")
                    continue
                market_data[symbol] = df
            except DataProviderError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{symbol}: errore imprevisto: {e}")

        fx_data, fx_errors = fetch_currency_data(
            cfg,
            watchlist,
            timezone,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(fx_errors)
        fx_pairs = configured_currency_pairs(watchlist, cfg)
        fx_rates, fx_rate_errors = latest_fx_rates(fx_data, fx_pairs, cfg)
        errors.extend(fx_rate_errors)
        active_watchlist, fx_watchlist_errors = enrich_watchlist_with_fx(watchlist, fx_rates, cfg)
        errors.extend(fx_watchlist_errors)

        regime_data, regime_errors = fetch_market_regime_data(
            cfg,
            market_data,
            timezone,
            lookback_days,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(regime_errors)
        relative_strength_data, relative_errors = fetch_relative_strength_data(
            cfg,
            active_watchlist,
            market_data,
            timezone,
            lookback_days,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(relative_errors)

        result = run_backtest(
            active_watchlist,
            market_data,
            cfg,
            regime_data=regime_data,
            relative_strength_data=relative_strength_data,
        )
        result.errors.extend(errors)
        report = (
            build_calibration_report(result, active_watchlist, cfg)
            if args.calibration_report
            else format_backtest_report(result)
        )
        if cfg["run"].get("save_reports", True):
            app.reports_dir.mkdir(parents=True, exist_ok=True)
            prefix = "calibration" if args.calibration_report else "backtest"
            path = app.reports_dir / f"{prefix}_{today.isoformat()}.md"
            path.write_text(report, encoding="utf-8")
        print(report)
        return 0

    dry_run = args.dry_run or bool(cfg["run"].get("dry_run_default", False))
    strategy_cfg = dict(cfg["strategy"])
    strategy_cfg.setdefault("min_reward_risk", cfg.get("risk", {}).get("min_reward_risk", 2.0))
    min_signal_score = float(strategy_cfg.get("min_signal_score", 0.0))

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    database_path = app.database_path
    if dry_run:
        temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(temp_dir.name) / app.database_path.name
        if app.database_path.exists():
            shutil.copy2(app.database_path, database_path)

    portfolio = PaperPortfolio(database_path, cfg)

    try:
        market_data = {}
        errors: list[str] = []
        min_rows = int(cfg["run"].get("min_rows_required", 220))
        lookback_days = int(cfg["run"].get("lookback_days", 430))

        for instrument in watchlist:
            symbol = instrument["symbol"]
            try:
                df = fetch_daily_data(
                    symbol,
                    lookback_days=lookback_days,
                    timezone=timezone,
                    request_timeout=request_timeout,
                    retries=download_retries,
                    process_timeout=process_timeout,
                )
                if len(df.dropna(subset=["Close", "SMA200"])) < min_rows:
                    errors.append(f"{symbol}: storico insufficiente dopo calcolo SMA200.")
                    continue
                market_data[symbol] = df
            except DataProviderError as e:
                errors.append(str(e))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{symbol}: errore imprevisto: {e}")

        fx_data, fx_errors = fetch_currency_data(
            cfg,
            watchlist,
            timezone,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(fx_errors)
        fx_pairs = configured_currency_pairs(watchlist, cfg)
        fx_rates, fx_rate_errors = latest_fx_rates(fx_data, fx_pairs, cfg)
        errors.extend(fx_rate_errors)
        active_watchlist, fx_watchlist_errors = enrich_watchlist_with_fx(watchlist, fx_rates, cfg)
        errors.extend(fx_watchlist_errors)

        regime_data, regime_errors = fetch_market_regime_data(
            cfg,
            market_data,
            timezone,
            lookback_days,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(regime_errors)
        relative_strength_data, relative_errors = fetch_relative_strength_data(
            cfg,
            active_watchlist,
            market_data,
            timezone,
            lookback_days,
            request_timeout,
            download_retries,
            process_timeout,
        )
        errors.extend(relative_errors)
        market_regime = evaluate_market_regime(regime_data, cfg, min_signal_score, today)
        min_signal_score = market_regime.active_min_signal_score

        close_and_trail_events = portfolio.update_open_positions(market_data, today)
        close_events = [e for e in close_and_trail_events if e.get("type") == "CLOSE"]
        trail_events = [e for e in close_and_trail_events if e.get("type") == "TRAIL_UPDATE"]

        actionable_buy_signals: list[Signal] = []
        all_logged_signals: list[Signal] = []
        learning_enabled = bool(cfg.get("learning", {}).get("enabled", True))
        learning_stats = load_learning_stats(app.signal_evaluations_csv, cfg) if learning_enabled else {}

        for instrument in active_watchlist:
            symbol = instrument["symbol"]
            df = market_data.get(symbol)
            if df is None:
                continue
            signals = analyze_buy_signals(instrument, df, strategy_cfg, today)
            for signal in signals:
                if signal.action == "WATCH":
                    all_logged_signals.append(signal)
                    continue
                if signal.action != "BUY":
                    continue
                signal = portfolio.size_signal(signal)
                signal = score_signal(signal, strategy_cfg)
                if signal.qty <= 0:
                    signal.action = "WATCH"
                    signal.reason += " Segnale non eseguito in paper: capitale/size non sufficiente con i limiti attuali."
                    all_logged_signals.append(signal)
                    continue
                signal = apply_relative_strength(signal, instrument, df, relative_strength_data, cfg)
                if signal.action != "BUY":
                    all_logged_signals.append(signal)
                    continue
                signal = review_opportunity(signal, market_regime, cfg)
                if signal.action != "BUY":
                    all_logged_signals.append(signal)
                    continue
                signal = apply_learning_feedback(signal, learning_stats, market_regime.to_dict(), cfg)
                if signal.score is not None and signal.score < min_signal_score:
                    signal.action = "WATCH"
                    signal.reason += f" Segnale non eseguito in paper: score {signal.score:.1f} sotto soglia {min_signal_score:.1f}."
                    all_logged_signals.append(signal)
                    continue
                can_open, reason = portfolio.can_open_new_position(signal, today)
                if not can_open:
                    signal.action = "WATCH"
                    signal.reason += f" Segnale non eseguito in paper: {reason}"
                    all_logged_signals.append(signal)
                    continue
                actionable_buy_signals.append(signal)
                all_logged_signals.append(signal)

        allocation_result = select_portfolio_candidates(
            actionable_buy_signals,
            portfolio.open_position_contexts(),
            market_regime.to_dict(),
            cfg,
        )

        opened_signals: list[Signal] = []
        for best_signal in allocation_result.selected:
            if not dry_run:
                opened = portfolio.open_position(best_signal)
                if opened:
                    opened_signals.append(best_signal)
            else:
                opened_signals.append(best_signal)

        ranked_signals = sorted(
            all_logged_signals,
            key=lambda s: (
                (s.meta or {}).get("allocation", {}).get("score", s.score or 0.0),
                s.score or 0.0,
                s.reward_risk or 0.0,
                -s.estimated_round_trip_cost,
            ),
            reverse=True,
        )

        if not dry_run:
            append_signals_csv(app.signals_csv, all_logged_signals)
            if learning_enabled:
                append_signal_journal(app.signal_journal_csv, all_logged_signals, market_regime.to_dict(), today)
                learning_summary = update_signal_evaluations(
                    app.signal_journal_csv,
                    app.signal_evaluations_csv,
                    market_data,
                    cfg,
                    today,
                )
            else:
                learning_summary = None
        else:
            learning_summary = None
        summary = portfolio.summary(market_data)
        message = build_daily_message(
            run_date=today,
            buy_signals=opened_signals,
            candidate_signals=ranked_signals,
            close_events=close_events,
            trail_events=trail_events,
            summary=summary,
            errors=errors,
            dry_run=dry_run,
            market_regime=market_regime.to_dict(),
            signal_learning=learning_summary,
            allocation=allocation_result.summary,
        )

        if cfg["run"].get("save_reports", True) and not dry_run:
            save_markdown_report(app.reports_dir, today, message)

        should_send = bool(cfg["run"].get("send_telegram", True)) and not dry_run
        if should_send and notifier.enabled:
            notifier.send(message)
        else:
            print(message)
            if should_send and not notifier.enabled:
                print("\nTelegram non configurato: imposto solo output console.")

        return 0
    finally:
        portfolio.close()
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
