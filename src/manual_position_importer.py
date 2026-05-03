from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python builds.
    ZoneInfo = None


MANUAL_POSITION_FIELDNAMES = [
    "symbol",
    "name",
    "instrument_type",
    "side",
    "qty",
    "avg_fill_price",
    "take_profit",
    "stop_loss",
    "last_price",
    "entry_date",
    "currency",
    "base_currency",
    "tradingview_symbol",
    "trade_value_display",
    "market_value_display",
    "unrealized_pnl_display",
    "unrealized_pnl_pct_display",
    "source",
]


@dataclass(frozen=True)
class ManualPosition:
    symbol: str
    name: str
    instrument_type: str
    qty: float
    entry_price: float
    stop: float
    target: float
    last_price: float | None
    entry_date: str
    currency: str
    base_currency: str
    tradingview_symbol: str
    trade_value_display: str
    market_value_display: str
    unrealized_pnl_display: str
    unrealized_pnl_pct_display: str
    source: str

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price if self.last_price is not None else self.entry_price)

    @property
    def reward_risk(self) -> float | None:
        risk = self.entry_price - self.stop
        reward = self.target - self.entry_price
        if risk <= 0 or reward <= 0:
            return None
        return round(reward / risk, 2)

    def meta(self) -> dict:
        meta = {
            "manual_tracking": True,
            "source": self.source,
            "currency": self.currency,
            "base_currency": self.base_currency,
            "tradingview_symbol": self.tradingview_symbol,
            "trade_value_display": self.trade_value_display,
            "market_value_display": self.market_value_display,
            "unrealized_pnl_display": self.unrealized_pnl_display,
            "unrealized_pnl_pct_display": self.unrealized_pnl_pct_display,
        }
        if self.currency == self.base_currency:
            meta["fx_to_base"] = 1.0
        if self.last_price is not None:
            meta["manual_last_price"] = self.last_price
        rr = self.reward_risk
        if rr is not None:
            meta["manual_reward_risk"] = rr
        return meta


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: {field}")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Riga {row_number}: valore numerico non valido per {field}: {value}") from exc


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_manual_position_rows(path: Path) -> list[ManualPosition]:
    if not path.exists():
        raise FileNotFoundError(f"File posizioni manuali non trovato: {path}")

    positions: list[ManualPosition] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):
            symbol = (row.get("symbol") or "").strip()
            if not symbol:
                raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: symbol")

            side = (row.get("side") or "Long").strip().lower()
            if side not in {"long", "buy"}:
                raise ValueError(f"Riga {row_number}: il tracking manuale supporta solo posizioni Long.")

            entry_date = (row.get("entry_date") or "").strip()
            if not entry_date:
                raise ValueError(f"Riga {row_number}: campo obbligatorio mancante: entry_date")

            currency = (row.get("currency") or "EUR").strip() or "EUR"
            base_currency = (row.get("base_currency") or currency).strip() or currency
            positions.append(
                ManualPosition(
                    symbol=symbol,
                    name=(row.get("name") or symbol).strip() or symbol,
                    instrument_type=(row.get("instrument_type") or "stock").strip() or "stock",
                    qty=_required_float(row, "qty", row_number),
                    entry_price=_required_float(row, "avg_fill_price", row_number),
                    target=_required_float(row, "take_profit", row_number),
                    stop=_required_float(row, "stop_loss", row_number),
                    last_price=_optional_float(row, "last_price"),
                    entry_date=entry_date,
                    currency=currency,
                    base_currency=base_currency,
                    tradingview_symbol=(row.get("tradingview_symbol") or "").strip(),
                    trade_value_display=(row.get("trade_value_display") or "").strip(),
                    market_value_display=(row.get("market_value_display") or "").strip(),
                    unrealized_pnl_display=(row.get("unrealized_pnl_display") or "").strip(),
                    unrealized_pnl_pct_display=(row.get("unrealized_pnl_pct_display") or "").strip(),
                    source=(row.get("source") or "manual").strip() or "manual",
                )
            )

    return positions


def _strip_inline_comment(value: str) -> str:
    if "#" in value:
        value = value.split("#", 1)[0]
    return value.strip().strip("\"'")


def _read_simple_yaml_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#") or ":" not in raw_line:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, raw_value = raw_line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        path_parts = [part for _, part in stack] + [key.strip()]
        value = _strip_inline_comment(raw_value)
        if value:
            values[".".join(path_parts)] = value
        else:
            stack.append((indent, key.strip()))
    return values


def _float_config(values: dict[str, str], key: str, default: float) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return default


def archive_database(db_path: Path) -> Path | None:
    if not db_path.exists():
        return None
    archive_dir = db_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{db_path.stem}_{timestamp}{db_path.suffix}"
    shutil.move(str(db_path), archive_path)
    return archive_path


def _init_db(conn: sqlite3.Connection, initial_capital: float) -> None:
    cur = conn.cursor()
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
            qty REAL NOT NULL,
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
    cur.execute(
        "INSERT OR REPLACE INTO account (id, cash, initial_capital, created_at) VALUES (1, ?, ?, ?)",
        (initial_capital, initial_capital, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def import_manual_positions_to_sqlite(
    db_path: Path,
    csv_path: Path,
    initial_capital: float,
    fixed_commission: float,
    reference_date: date,
) -> dict:
    positions = load_manual_position_rows(csv_path)
    if not positions:
        raise ValueError(f"Nessuna posizione trovata in {csv_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    archived = archive_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        _init_db(conn, initial_capital)
        cash = initial_capital
        total_notional = 0.0
        total_entry_commissions = 0.0
        total_market_value = 0.0

        cur = conn.cursor()
        for position in positions:
            commission = max(0.0, fixed_commission)
            total_cost = position.notional + commission
            if cash < total_cost:
                raise ValueError(
                    f"Liquidita insufficiente per importare {position.symbol}: "
                    f"servono {total_cost:.2f}, disponibili {cash:.2f}."
                )
            cash -= total_cost
            total_notional += position.notional
            total_entry_commissions += commission
            total_market_value += position.market_value
            highest_close = max(position.last_price or position.entry_price, position.entry_price)

            cur.execute(
                """
                INSERT INTO positions (
                    symbol, name, instrument_type, strategy, entry_date, entry_price, qty,
                    stop, target, highest_close, entry_commission, status, meta
                ) VALUES (?, ?, ?, 'manual_tracking', ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
                """,
                (
                    position.symbol,
                    position.name,
                    position.instrument_type,
                    position.entry_date,
                    position.entry_price,
                    position.qty,
                    position.stop,
                    position.target,
                    highest_close,
                    commission,
                    json.dumps(position.meta(), ensure_ascii=False),
                ),
            )

        exit_commissions = len(positions) * max(0.0, fixed_commission)
        equity = cash + total_market_value - exit_commissions
        cur.execute("UPDATE account SET cash = ? WHERE id = 1", (round(cash, 2),))
        cur.execute(
            "INSERT INTO events (event_date, event_type, symbol, message, payload) VALUES (?, ?, ?, ?, ?)",
            (
                reference_date.isoformat(),
                "MANUAL_IMPORT",
                None,
                f"Importate {len(positions)} posizioni manuali nel paper portfolio.",
                json.dumps(
                    {
                        "symbols": [position.symbol for position in positions],
                        "notional": round(total_notional, 2),
                        "entry_commissions": round(total_entry_commissions, 2),
                        "cash": round(cash, 2),
                    },
                    ensure_ascii=False,
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "archived_database": archived,
        "imported": len(positions),
        "notional": round(total_notional, 2),
        "entry_commissions": round(total_entry_commissions, 2),
        "cash": round(cash, 2),
        "open_market_value": round(total_market_value, 2),
        "equity": round(equity, 2),
        "total_pnl": round(equity - initial_capital, 2),
        "total_return_pct": round(((equity - initial_capital) / initial_capital) * 100, 2)
        if initial_capital
        else 0.0,
    }


def _today(timezone_name: str) -> date:
    if ZoneInfo is None:
        return date.today()
    return datetime.now(ZoneInfo(timezone_name)).date()


def run_manual_import_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import manual TradingView positions")
    parser.add_argument("--base-dir", default=".", help="Project base directory")
    parser.add_argument("--import-manual-positions", action="store_true")
    parser.add_argument("--manual-positions-file", default=None, help="Override manual positions CSV path")
    args, _ = parser.parse_known_args(argv)

    base_dir = Path(args.base_dir).resolve()
    config_values = _read_simple_yaml_values(base_dir / "config.yaml")
    database_path = base_dir / config_values.get("paths.database", "state/trading_lab.sqlite")
    if args.manual_positions_file:
        csv_path = Path(args.manual_positions_file)
        if not csv_path.is_absolute():
            csv_path = base_dir / csv_path
    else:
        csv_path = base_dir / config_values.get("paths.manual_positions", "data/manual_positions.csv")

    initial_capital = _float_config(config_values, "risk.initial_capital", 1000.0)
    fixed_commission = _float_config(config_values, "costs.fixed_commission", 1.0)
    timezone_name = config_values.get("project.timezone", "Europe/Rome")
    result = import_manual_positions_to_sqlite(
        database_path,
        csv_path,
        initial_capital,
        fixed_commission,
        _today(timezone_name),
    )

    print(f"Import posizioni manuali completato da: {csv_path}")
    if result["archived_database"]:
        print(f"Archivio database precedente: {result['archived_database']}")
    print(
        "Posizioni importate: "
        f"{result['imported']} | Notional: {result['notional']:.2f} EUR | "
        f"Commissioni ingresso stimate: {result['entry_commissions']:.2f} EUR | "
        f"Cash residuo: {result['cash']:.2f} EUR"
    )
    print(
        "Equity stimata con ultimi prezzi TradingView: "
        f"{result['equity']:.2f} EUR | P/L totale: {result['total_pnl']:.2f} EUR "
        f"({result['total_return_pct']:.2f}%)"
    )
    return 0
