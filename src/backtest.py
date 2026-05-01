from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .costs import estimate_commission, estimate_round_trip_cost, max_affordable_quantity
from .market_regime import evaluate_market_regime
from .strategy import Signal, analyze_buy_signals, score_signal


@dataclass
class BacktestPosition:
    symbol: str
    name: str
    instrument_type: str
    strategy: str
    entry_date: date
    entry_price: float
    qty: int
    stop: float
    target: float
    highest_close: float
    entry_commission: float
    meta: dict = field(default_factory=dict)


@dataclass
class BacktestTrade:
    symbol: str
    name: str
    strategy: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    qty: int
    exit_reason: str
    gross_pnl: float
    net_pnl: float


@dataclass
class BacktestResult:
    start_date: date | None
    end_date: date | None
    initial_capital: float
    ending_equity: float
    realized_pnl: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: list[BacktestTrade]
    open_positions: list[BacktestPosition]
    errors: list[str]
    equity_curve: list[dict]
    regime_counts: dict[str, int] = field(default_factory=dict)

    @property
    def closed_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return len([trade for trade in self.trades if trade.net_pnl > 0])

    @property
    def losing_trades(self) -> int:
        return len([trade for trade in self.trades if trade.net_pnl < 0])

    @property
    def win_rate(self) -> float | None:
        if not self.trades:
            return None
        return round((self.winning_trades / len(self.trades)) * 100, 1)

    @property
    def profit_factor(self) -> float | None:
        gross_profit = sum(trade.net_pnl for trade in self.trades if trade.net_pnl > 0)
        gross_loss = abs(sum(trade.net_pnl for trade in self.trades if trade.net_pnl < 0))
        if gross_loss == 0:
            return None
        return round(gross_profit / gross_loss, 2)

    @property
    def avg_trade_pnl(self) -> float | None:
        if not self.trades:
            return None
        return round(sum(trade.net_pnl for trade in self.trades) / len(self.trades), 2)


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _row_on_or_before(df: pd.DataFrame, day: pd.Timestamp) -> pd.Series | None:
    sliced = df.loc[:day]
    if sliced.empty:
        return None
    row = sliced.iloc[-1]
    if pd.isna(row.get("Close")):
        return None
    return row


def _estimate_equity(
    cash: float,
    positions: list[BacktestPosition],
    market_data: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    costs_cfg: dict,
) -> float:
    equity = cash
    for pos in positions:
        row = _row_on_or_before(market_data[pos.symbol], day)
        close = _safe_float(row.get("Close")) if row is not None else pos.entry_price
        exit_notional = (close or pos.entry_price) * pos.qty
        equity += exit_notional - estimate_commission(exit_notional, costs_cfg)
    return round(equity, 2)


def _size_signal(signal: Signal, cash: float, risk_cfg: dict, costs_cfg: dict) -> Signal:
    if signal.entry is None or signal.stop is None:
        return signal

    unit_risk = signal.entry - signal.stop
    if unit_risk <= 0:
        signal.reason += " Rischio unitario non valido."
        return signal

    risk_per_trade = float(risk_cfg.get("risk_per_trade", 25.0))
    max_allocation = float(risk_cfg.get("max_allocation_per_trade", 500.0))
    qty_by_risk = int(risk_per_trade // unit_risk)
    qty_by_allocation = max_affordable_quantity(signal.entry, cash, max_allocation, costs_cfg)
    qty = max(0, min(qty_by_risk, qty_by_allocation))

    signal.qty = qty
    signal.notional = round(qty * signal.entry, 2) if qty > 0 else 0.0
    signal.estimated_round_trip_cost = estimate_round_trip_cost(signal.notional, costs_cfg) if qty > 0 else 0.0
    return signal


def _close_position(
    pos: BacktestPosition,
    exit_date: date,
    exit_price: float,
    exit_reason: str,
    costs_cfg: dict,
) -> tuple[BacktestTrade, float]:
    gross_pnl = (exit_price - pos.entry_price) * pos.qty
    exit_notional = exit_price * pos.qty
    exit_commission = estimate_commission(exit_notional, costs_cfg)
    net_pnl = gross_pnl - pos.entry_commission - exit_commission
    cash_in = exit_notional - exit_commission
    trade = BacktestTrade(
        symbol=pos.symbol,
        name=pos.name,
        strategy=pos.strategy,
        entry_date=pos.entry_date,
        exit_date=exit_date,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        qty=pos.qty,
        exit_reason=exit_reason,
        gross_pnl=round(gross_pnl, 2),
        net_pnl=round(net_pnl, 2),
    )
    return trade, cash_in


def run_backtest(
    watchlist: list[dict],
    market_data: dict[str, pd.DataFrame],
    config: dict,
    regime_data: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    risk_cfg = config["risk"]
    costs_cfg = config["costs"]
    strategy_cfg = dict(config["strategy"])
    strategy_cfg.setdefault("min_reward_risk", risk_cfg.get("min_reward_risk", 2.0))
    backtest_cfg = config.get("backtest", {})

    initial_capital = float(risk_cfg.get("initial_capital", 1000.0))
    cash = initial_capital
    positions: list[BacktestPosition] = []
    trades: list[BacktestTrade] = []
    equity_curve: list[dict] = []
    errors: list[str] = []
    regime_counts: dict[str, int] = {}
    last_stop_by_symbol: dict[str, date] = {}
    trades_by_month: dict[str, int] = {}

    max_open_positions = int(risk_cfg.get("max_open_positions", 2))
    max_trades_per_month = int(risk_cfg.get("max_trades_per_month", 0))
    cooldown_after_stop_days = int(risk_cfg.get("cooldown_after_stop_days", 0))
    max_holding_days = int(risk_cfg.get("max_holding_days", 45))
    monthly_loss_limit = float(risk_cfg.get("monthly_loss_limit", 100.0))
    trailing_mult = float(strategy_cfg.get("trailing_atr_multiplier", 2.0))
    min_signal_score = float(strategy_cfg.get("min_signal_score", 0.0))
    max_new_positions_per_day = int(backtest_cfg.get("max_new_positions_per_day", 1))

    dates = sorted({idx for df in market_data.values() for idx in df.index})
    if not dates:
        return BacktestResult(
            start_date=None,
            end_date=None,
            initial_capital=initial_capital,
            ending_equity=initial_capital,
            realized_pnl=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            trades=[],
            open_positions=[],
            errors=["Nessun dato di mercato disponibile per il backtest."],
            equity_curve=[],
            regime_counts={},
        )

    instruments_by_symbol = {item["symbol"]: item for item in watchlist}

    for day_ts in dates:
        day = day_ts.date()
        month_key = day.strftime("%Y-%m")
        market_regime = evaluate_market_regime(regime_data or {}, config, min_signal_score, day)
        active_min_signal_score = market_regime.active_min_signal_score
        if market_regime.enabled:
            regime_counts[market_regime.state] = regime_counts.get(market_regime.state, 0) + 1

        remaining_positions: list[BacktestPosition] = []
        for pos in positions:
            df = market_data.get(pos.symbol)
            if df is None or day_ts not in df.index:
                remaining_positions.append(pos)
                continue

            row = df.loc[day_ts]
            close = _safe_float(row.get("Close"))
            low = _safe_float(row.get("Low"))
            high = _safe_float(row.get("High"))
            sma50 = _safe_float(row.get("SMA50"))
            atr14 = _safe_float(row.get("ATR14"))
            if close is None or low is None or high is None or sma50 is None or atr14 is None:
                remaining_positions.append(pos)
                continue

            exit_price = None
            exit_reason = None
            if low <= pos.stop:
                exit_price = pos.stop
                exit_reason = "stop_loss"
            elif high >= pos.target:
                exit_price = pos.target
                exit_reason = "target_reached"
            elif close < sma50:
                exit_price = close
                exit_reason = "trend_failure_close_below_sma50"
            elif (day - pos.entry_date).days >= max_holding_days:
                exit_price = close
                exit_reason = "time_exit"

            if exit_price is not None and exit_reason is not None:
                trade, cash_in = _close_position(pos, day, exit_price, exit_reason, costs_cfg)
                trades.append(trade)
                cash += cash_in
                if exit_reason == "stop_loss":
                    last_stop_by_symbol[pos.symbol] = day
                continue

            highest_close = max(pos.highest_close, close)
            initial_risk = pos.entry_price - pos.stop
            if initial_risk > 0 and close >= pos.entry_price + initial_risk:
                stop = round(max(pos.stop, highest_close - trailing_mult * atr14, pos.entry_price), 4)
            else:
                stop = pos.stop
            pos.highest_close = highest_close
            pos.stop = stop
            remaining_positions.append(pos)
        positions = remaining_positions

        month_pnl = sum(
            trade.net_pnl for trade in trades if trade.exit_date.strftime("%Y-%m") == month_key
        )
        monthly_trades = trades_by_month.get(month_key, 0)
        candidates: list[Signal] = []
        open_symbols = {pos.symbol for pos in positions}

        if (
            market_regime.new_positions_allowed
            and month_pnl > -monthly_loss_limit
            and len(positions) < max_open_positions
        ):
            for symbol, df in market_data.items():
                if symbol in open_symbols or day_ts not in df.index:
                    continue
                if max_trades_per_month > 0 and monthly_trades >= max_trades_per_month:
                    break
                last_stop = last_stop_by_symbol.get(symbol)
                if last_stop is not None and cooldown_after_stop_days > 0:
                    if 0 <= (day - last_stop).days < cooldown_after_stop_days:
                        continue

                instrument = instruments_by_symbol.get(symbol, {"symbol": symbol, "name": symbol, "type": "unknown"})
                df_slice = df.loc[:day_ts]
                signals = analyze_buy_signals(instrument, df_slice, strategy_cfg, day)
                for signal in signals:
                    if signal.action != "BUY":
                        continue
                    signal = _size_signal(signal, cash, risk_cfg, costs_cfg)
                    signal = score_signal(signal, strategy_cfg)
                    if signal.qty <= 0:
                        continue
                    if signal.score is not None and signal.score < active_min_signal_score:
                        continue
                    candidates.append(signal)

        candidates.sort(
            key=lambda signal: (signal.score or 0.0, signal.reward_risk or 0.0, -signal.estimated_round_trip_cost),
            reverse=True,
        )
        opened_today = 0
        for signal in candidates:
            if opened_today >= max_new_positions_per_day or len(positions) >= max_open_positions:
                break
            if signal.symbol in {pos.symbol for pos in positions}:
                continue
            if signal.entry is None or signal.stop is None or signal.target is None:
                continue
            notional = signal.qty * signal.entry
            commission = estimate_commission(notional, costs_cfg)
            total_cost = notional + commission
            if cash < total_cost:
                continue
            cash -= total_cost
            positions.append(
                BacktestPosition(
                    symbol=signal.symbol,
                    name=signal.name,
                    instrument_type=signal.instrument_type,
                    strategy=signal.strategy,
                    entry_date=day,
                    entry_price=signal.entry,
                    qty=signal.qty,
                    stop=signal.stop,
                    target=signal.target,
                    highest_close=signal.price or signal.entry,
                    entry_commission=commission,
                    meta=signal.meta or {},
                )
            )
            trades_by_month[month_key] = trades_by_month.get(month_key, 0) + 1
            monthly_trades = trades_by_month[month_key]
            opened_today += 1

        equity_curve.append(
            {
                "date": day.isoformat(),
                "equity": _estimate_equity(cash, positions, market_data, day_ts, costs_cfg),
                "cash": round(cash, 2),
                "open_positions": len(positions),
                "market_regime": market_regime.state,
            }
        )

    ending_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
    realized_pnl = round(sum(trade.net_pnl for trade in trades), 2)
    total_return_pct = round(((ending_equity - initial_capital) / initial_capital) * 100, 2)

    peak = initial_capital
    max_drawdown_pct = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            drawdown = ((equity - peak) / peak) * 100
            max_drawdown_pct = min(max_drawdown_pct, drawdown)

    return BacktestResult(
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
        initial_capital=initial_capital,
        ending_equity=round(ending_equity, 2),
        realized_pnl=realized_pnl,
        total_return_pct=total_return_pct,
        max_drawdown_pct=round(max_drawdown_pct, 2),
        trades=trades,
        open_positions=positions,
        errors=errors,
        equity_curve=equity_curve,
        regime_counts=regime_counts,
    )


def format_backtest_report(result: BacktestResult) -> str:
    lines = [
        "# Backtest Directa Telegram Trading Lab",
        "",
        f"Periodo: {result.start_date or 'n/d'} -> {result.end_date or 'n/d'}",
        f"Capitale iniziale: {result.initial_capital:.2f} EUR",
        f"Equity finale stimata: {result.ending_equity:.2f} EUR",
        f"P/L realizzato netto: {result.realized_pnl:.2f} EUR",
        f"Rendimento totale stimato: {result.total_return_pct:.2f}%",
        f"Max drawdown stimato: {result.max_drawdown_pct:.2f}%",
        "",
        "## Statistiche trade",
        "",
        f"Trade chiusi: {result.closed_trades}",
        f"Trade vincenti: {result.winning_trades}",
        f"Trade perdenti: {result.losing_trades}",
        f"Win rate: {result.win_rate if result.win_rate is not None else 'n/d'}%",
        f"Profit factor: {result.profit_factor if result.profit_factor is not None else 'n/d'}",
        f"P/L medio per trade: {result.avg_trade_pnl if result.avg_trade_pnl is not None else 'n/d'} EUR",
        f"Posizioni ancora aperte: {len(result.open_positions)}",
    ]

    if result.regime_counts:
        lines.extend(["", "## Regime di mercato", ""])
        for state, count in sorted(result.regime_counts.items()):
            lines.append(f"- {state}: {count} sedute")

    if result.trades:
        lines.extend(["", "## Ultimi trade", ""])
        for trade in result.trades[-10:]:
            lines.append(
                f"- {trade.exit_date.isoformat()} {trade.symbol} {trade.exit_reason}: "
                f"{trade.net_pnl:.2f} EUR ({trade.entry_date.isoformat()} -> {trade.exit_date.isoformat()})"
            )

    if result.errors:
        lines.extend(["", "## Errori", ""])
        for error in result.errors:
            lines.append(f"- {error}")

    lines.append("")
    lines.append("Nota: backtest didattico su dati Yahoo Finance, con costi stimati e senza garanzia di esecuzione reale.")
    return "\n".join(lines)
