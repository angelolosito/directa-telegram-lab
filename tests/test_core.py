from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.backtest import run_backtest
from src.paper_portfolio import PaperPortfolio
from src.strategy import Signal, score_signal


def sample_config() -> dict:
    return {
        "risk": {
            "initial_capital": 1000.0,
            "risk_per_trade": 25.0,
            "monthly_loss_limit": 100.0,
            "max_open_positions": 2,
            "max_trades_per_month": 6,
            "cooldown_after_stop_days": 5,
            "max_allocation_per_trade": 500.0,
            "min_reward_risk": 2.0,
            "max_holding_days": 45,
        },
        "costs": {
            "variable_rate": 0.0019,
            "min_commission": 1.50,
            "max_commission": 18.00,
        },
        "strategy": {
            "min_signal_score": 60.0,
            "score_base": 50.0,
            "trailing_atr_multiplier": 2.0,
            "enabled": {"trend_pullback": True, "controlled_breakout": True},
            "rsi_min_pullback": 40,
            "rsi_max_pullback": 68,
            "rsi_max_breakout": 75,
            "breakout_lookback_days": 20,
            "volume_breakout_multiplier": 1.10,
            "pullback_lookback_days": 8,
            "atr_stop_multiplier": 1.50,
            "breakout_atr_stop_multiplier": 2.00,
        },
        "backtest": {"max_new_positions_per_day": 1},
    }


class RiskControlTests(unittest.TestCase):
    def test_cooldown_blocks_reentry_after_stop(self) -> None:
        cfg = sample_config()
        with TemporaryDirectory() as tmp:
            portfolio = PaperPortfolio(Path(tmp) / "lab.sqlite", cfg)
            signal = Signal(
                symbol="TEST.MI",
                name="Test",
                instrument_type="stock",
                action="BUY",
                strategy="trend_pullback",
                date="2026-05-01",
                price=10.0,
                entry=10.0,
                stop=9.5,
                target=11.0,
                reward_risk=2.0,
            )
            portfolio.size_signal(signal)
            self.assertTrue(portfolio.open_position(signal))
            portfolio.close_position(1, 9.5, "stop_loss", date(2026, 5, 1))

            next_signal = Signal(
                symbol="TEST.MI",
                name="Test",
                instrument_type="stock",
                action="BUY",
                strategy="trend_pullback",
                date="2026-05-01",
                price=10.0,
                entry=10.0,
                stop=9.5,
                target=11.0,
                reward_risk=2.0,
            )
            allowed, reason = portfolio.can_open_new_position(next_signal, date(2026, 5, 1))
            portfolio.close()

        self.assertFalse(allowed)
        self.assertIn("Cooldown post-stop", reason)

    def test_signal_score_includes_cost_penalty(self) -> None:
        cfg = sample_config()
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-05-01",
            price=10.0,
            entry=10.0,
            stop=9.7,
            target=10.6,
            reward_risk=2.0,
            qty=1,
            notional=10.0,
            estimated_round_trip_cost=3.0,
            meta={
                "close": 10.0,
                "sma50": 9.5,
                "sma200": 9.0,
                "rsi14": 55.0,
                "volume": 1200,
                "vol20": 1000,
            },
        )
        scored = score_signal(signal, cfg["strategy"])

        self.assertIsNotNone(scored.score)
        self.assertLess(scored.score or 0, 100)
        self.assertIn("costi", scored.score_details)


class BacktestTests(unittest.TestCase):
    def test_backtest_opens_and_closes_mocked_signal(self) -> None:
        cfg = sample_config()
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        df = pd.DataFrame(
            {
                "Open": [10.0, 10.2, 11.0],
                "High": [10.1, 11.2, 11.4],
                "Low": [9.9, 10.0, 10.8],
                "Close": [10.0, 11.0, 11.2],
                "SMA20": [9.5, 9.6, 9.7],
                "SMA50": [9.0, 9.1, 9.2],
                "SMA200": [8.0, 8.1, 8.2],
                "RSI14": [55.0, 58.0, 60.0],
                "ATR14": [0.3, 0.3, 0.3],
                "Volume": [1000, 1100, 1200],
                "VOL20": [900, 950, 1000],
                "HIGH20_PREV": [10.0, 10.1, 11.2],
            },
            index=dates,
        )

        def fake_analyze(instrument, df_slice, strategy_cfg, today):  # noqa: ARG001
            if today == date(2026, 1, 1):
                return [
                    Signal(
                        symbol="TEST.MI",
                        name="Test",
                        instrument_type="stock",
                        action="BUY",
                        strategy="trend_pullback",
                        date=today.isoformat(),
                        price=10.0,
                        entry=10.0,
                        stop=9.5,
                        target=11.0,
                        reward_risk=2.0,
                        meta={
                            "close": 10.0,
                            "sma50": 9.0,
                            "sma200": 8.0,
                            "rsi14": 55.0,
                            "volume": 1000,
                            "vol20": 900,
                        },
                    )
                ]
            return []

        with patch("src.backtest.analyze_buy_signals", side_effect=fake_analyze):
            result = run_backtest(
                watchlist=[{"symbol": "TEST.MI", "name": "Test", "type": "stock"}],
                market_data={"TEST.MI": df},
                config=cfg,
            )

        self.assertEqual(result.closed_trades, 1)
        self.assertEqual(result.trades[0].exit_reason, "target_reached")
        self.assertGreater(result.trades[0].net_pnl, 0)


if __name__ == "__main__":
    unittest.main()
