from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .costs import estimate_commission, estimate_round_trip_cost
from .strategy import Signal


@dataclass
class Position:
    id: int
    symbol: str
    name: str
    strategy: str
    entry_date: str
    entry_price: float
    qty: int
    stop: float
    target: float
    highest_close: float
    status: str


class PaperPortfolio:
    def __init__(self, db_path: Path, config: dict):
        self.db_path = db_path
        self.config = config
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._init_account_if_needed()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL,
                initial_capital REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                instrument_type TEXT,
                strategy TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                qty INTEGER NOT NULL,
                stop REAL NOT NULL,
                target REAL NOT NULL,
                highest_close REAL NOT NULL,
                entry_commission REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                exit_commission REAL,
                gross_pnl REAL,
                net_pnl REAL,
                meta TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                event_type TEXT NOT NULL,
                symbol TEXT,
                message TEXT NOT NULL,
                payload TEXT
            )
            """
        )
        self.conn.commit()

    def _init_account_if_needed(self) -> None:
        cur = self.conn.cursor()
        row = cur.execute("SELECT id FROM account WHERE id = 1").fetchone()
        if row is None:
            initial_capital = float(self.config["risk"].get("initial_capital", 1000.0))
            cur.execute(
                "INSERT INTO account (id, cash, initial_capital, created_at) VALUES (1, ?, ?, ?)",
                (initial_capital, initial_capital, datetime.utcnow().isoformat()),
            )
            self.conn.commit()

    def log_event(self, event_type: str, message: str, symbol: str | None = None, payload: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events (event_date, event_type, symbol, message, payload) VALUES (?, ?, ?, ?, ?)",
            (date.today().isoformat(), event_type, symbol, message, json.dumps(payload or {}, ensure_ascii=False)),
        )
        self.conn.commit()

    def cash(self) -> float:
        row = self.conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
        return float(row["cash"])

    def open_positions(self) -> list[Position]:
        rows = self.conn.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY entry_date").fetchall()
        return [
            Position(
                id=int(r["id"]),
                symbol=r["symbol"],
                name=r["name"],
                strategy=r["strategy"],
                entry_date=r["entry_date"],
                entry_price=float(r["entry_price"]),
                qty=int(r["qty"]),
                stop=float(r["stop"]),
                target=float(r["target"]),
                highest_close=float(r["highest_close"]),
                status=r["status"],
            )
            for r in rows
        ]

    def has_open_position(self, symbol: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM positions WHERE symbol = ? AND status = 'OPEN' LIMIT 1", (symbol,)
        ).fetchone()
        return row is not None

    def realized_monthly_pnl(self, reference_date: date) -> float:
        start = reference_date.replace(day=1).isoformat()
        end = reference_date.isoformat()
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(net_pnl), 0) AS pnl
            FROM positions
            WHERE status = 'CLOSED' AND exit_date >= ? AND exit_date <= ?
            """,
            (start, end),
        ).fetchone()
        return float(row["pnl"])

    def can_open_new_position(self, signal: Signal, reference_date: date) -> tuple[bool, str]:
        risk_cfg = self.config["risk"]
        max_open = int(risk_cfg.get("max_open_positions", 2))
        monthly_loss_limit = float(risk_cfg.get("monthly_loss_limit", 100.0))
        open_count = len(self.open_positions())

        if signal.entry is None or signal.stop is None or signal.target is None:
            return False, "Segnale incompleto: entry/stop/target mancanti."

        if open_count >= max_open:
            return False, f"Limite posizioni aperte raggiunto: {open_count}/{max_open}."

        if self.has_open_position(signal.symbol):
            return False, "Esiste già una posizione aperta su questo strumento."

        realized = self.realized_monthly_pnl(reference_date)
        if realized <= -monthly_loss_limit:
            return False, f"Limite perdita mensile raggiunto: {realized:.2f} €."

        return True, "OK"

    def size_signal(self, signal: Signal) -> Signal:
        if signal.entry is None or signal.stop is None:
            return signal

        risk_cfg = self.config["risk"]
        costs_cfg = self.config["costs"]
        risk_per_trade = float(risk_cfg.get("risk_per_trade", 25.0))
        max_allocation = float(risk_cfg.get("max_allocation_per_trade", 500.0))
        available_cash = self.cash()

        unit_risk = signal.entry - signal.stop
        if unit_risk <= 0:
            signal.reason += " Rischio unitario non valido."
            return signal

        qty_by_risk = int(risk_per_trade // unit_risk)
        qty_by_allocation = int(min(max_allocation, available_cash) // signal.entry)
        qty = max(0, min(qty_by_risk, qty_by_allocation))

        notional = round(qty * signal.entry, 2) if qty > 0 else 0.0
        signal.qty = qty
        signal.notional = notional
        signal.estimated_round_trip_cost = estimate_round_trip_cost(notional, costs_cfg) if qty > 0 else 0.0
        return signal

    def open_position(self, signal: Signal) -> bool:
        if signal.qty <= 0 or signal.entry is None or signal.stop is None or signal.target is None:
            self.log_event("SKIP", "Quantità pari a zero: posizione non aperta.", signal.symbol, signal.to_dict())
            return False

        notional = signal.qty * signal.entry
        commission = estimate_commission(notional, self.config["costs"])
        total_cost = notional + commission
        if self.cash() < total_cost:
            self.log_event("SKIP", "Liquidità insufficiente per aprire posizione simulata.", signal.symbol, signal.to_dict())
            return False

        cur = self.conn.cursor()
        cur.execute("UPDATE account SET cash = cash - ? WHERE id = 1", (total_cost,))
        cur.execute(
            """
            INSERT INTO positions (
                symbol, name, instrument_type, strategy, entry_date, entry_price, qty,
                stop, target, highest_close, entry_commission, status, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """,
            (
                signal.symbol,
                signal.name,
                signal.instrument_type,
                signal.strategy,
                signal.date,
                signal.entry,
                signal.qty,
                signal.stop,
                signal.target,
                signal.price or signal.entry,
                commission,
                json.dumps(signal.meta or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        self.log_event("OPEN", f"Aperta posizione paper su {signal.symbol}", signal.symbol, signal.to_dict())
        return True

    def update_open_positions(
        self,
        market_data: dict[str, pd.DataFrame],
        reference_date: date,
    ) -> list[dict]:
        events: list[dict] = []
        risk_cfg = self.config["risk"]
        max_holding_days = int(risk_cfg.get("max_holding_days", 45))
        trailing_mult = float(self.config["strategy"].get("trailing_atr_multiplier", 2.0))

        for pos in self.open_positions():
            df = market_data.get(pos.symbol)
            if df is None or df.empty:
                continue
            latest = df.dropna(subset=["Close", "Low", "High", "SMA50", "ATR14"]).iloc[-1]
            close = float(latest["Close"])
            low = float(latest["Low"])
            high = float(latest["High"])
            sma50 = float(latest["SMA50"])
            atr14 = float(latest["ATR14"])
            new_highest = max(pos.highest_close, close)

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
            else:
                try:
                    days_open = (reference_date - datetime.fromisoformat(pos.entry_date).date()).days
                except Exception:
                    days_open = 0
                if days_open >= max_holding_days:
                    exit_price = close
                    exit_reason = "time_exit"

            if exit_price is not None and exit_reason is not None:
                event = self.close_position(pos.id, exit_price, exit_reason, reference_date)
                events.append(event)
                continue

            # Trailing stop: after price has moved at least 1R in favor, raise stop but never lower it.
            initial_risk = pos.entry_price - pos.stop
            if initial_risk > 0 and close >= pos.entry_price + initial_risk:
                candidate_stop = round(max(pos.stop, new_highest - trailing_mult * atr14, pos.entry_price), 4)
            else:
                candidate_stop = pos.stop

            if candidate_stop != pos.stop or new_highest != pos.highest_close:
                self.conn.execute(
                    "UPDATE positions SET stop = ?, highest_close = ? WHERE id = ?",
                    (candidate_stop, new_highest, pos.id),
                )
                self.conn.commit()
                if candidate_stop != pos.stop:
                    events.append(
                        {
                            "type": "TRAIL_UPDATE",
                            "symbol": pos.symbol,
                            "message": f"Trailing stop aggiornato su {pos.symbol}: {pos.stop:.4f} → {candidate_stop:.4f}",
                        }
                    )

        return events

    def close_position(self, position_id: int, exit_price: float, reason: str, reference_date: date) -> dict:
        row = self.conn.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        if row is None:
            raise ValueError(f"Posizione non trovata: {position_id}")

        qty = int(row["qty"])
        entry_price = float(row["entry_price"])
        entry_commission = float(row["entry_commission"])
        gross_pnl = (exit_price - entry_price) * qty
        exit_notional = exit_price * qty
        exit_commission = estimate_commission(exit_notional, self.config["costs"])
        net_pnl = gross_pnl - entry_commission - exit_commission
        cash_in = exit_notional - exit_commission

        self.conn.execute("UPDATE account SET cash = cash + ? WHERE id = 1", (cash_in,))
        self.conn.execute(
            """
            UPDATE positions
            SET status = 'CLOSED', exit_date = ?, exit_price = ?, exit_reason = ?,
                exit_commission = ?, gross_pnl = ?, net_pnl = ?
            WHERE id = ?
            """,
            (
                reference_date.isoformat(),
                exit_price,
                reason,
                exit_commission,
                round(gross_pnl, 2),
                round(net_pnl, 2),
                position_id,
            ),
        )
        self.conn.commit()

        event = {
            "type": "CLOSE",
            "symbol": row["symbol"],
            "name": row["name"],
            "reason": reason,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
        }
        self.log_event("CLOSE", f"Chiusa posizione paper su {row['symbol']}: {reason}", row["symbol"], event)
        return event

    def summary(self) -> dict:
        cash = self.cash()
        open_rows = self.conn.execute("SELECT * FROM positions WHERE status = 'OPEN'").fetchall()
        closed_row = self.conn.execute(
            "SELECT COALESCE(SUM(net_pnl), 0) AS realized FROM positions WHERE status = 'CLOSED'"
        ).fetchone()
        return {
            "cash": round(cash, 2),
            "open_positions": len(open_rows),
            "realized_pnl": round(float(closed_row["realized"]), 2),
        }
