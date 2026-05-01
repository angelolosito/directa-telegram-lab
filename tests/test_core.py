from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.backtest import run_backtest
from src.market_regime import evaluate_market_regime
from src.opportunity import review_opportunity
from src.paper_portfolio import PaperPortfolio
from src.signal_journal import append_signal_journal, update_signal_evaluations
from src.strategy import Signal, analyze_buy_signals, score_signal


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
            "near_breakout_pct": 1.5,
            "setup_watch_min_score": 50.0,
            "pullback_lookback_days": 8,
            "atr_stop_multiplier": 1.50,
            "breakout_atr_stop_multiplier": 2.00,
        },
        "opportunity": {
            "enabled": True,
            "min_decision_score": 62.0,
            "ideal_pullback_distance_pct": 2.5,
            "max_pullback_distance_pct": 4.5,
            "ideal_breakout_extension_atr": 0.8,
            "max_breakout_extension_atr": 1.4,
            "min_risk_pct": 0.8,
            "max_risk_pct": 8.0,
            "high_quality_cost_pct": 1.0,
            "max_cost_pct": 5.0,
        },
        "learning": {
            "enabled": True,
            "horizons_sessions": [2, 3],
            "primary_horizon_sessions": 2,
            "min_bucket_count": 1,
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

    def test_position_size_reserves_entry_commission(self) -> None:
        cfg = sample_config()
        cfg["risk"]["initial_capital"] = 101.0
        with TemporaryDirectory() as tmp:
            portfolio = PaperPortfolio(Path(tmp) / "lab.sqlite", cfg)
            signal = Signal(
                symbol="TEST.MI",
                name="Test",
                instrument_type="stock",
                action="BUY",
                strategy="trend_pullback",
                date="2026-05-01",
                price=100.0,
                entry=100.0,
                stop=90.0,
                target=120.0,
                reward_risk=2.0,
            )
            sized = portfolio.size_signal(signal)
            portfolio.close()

        self.assertEqual(sized.qty, 0)

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


class MarketRegimeTests(unittest.TestCase):
    def test_risk_off_regime_blocks_new_positions(self) -> None:
        cfg = sample_config()
        cfg["market_regime"] = {
            "enabled": True,
            "block_new_positions_when_risk_off": True,
            "benchmarks": [{"symbol": "BENCH.MI", "name": "Benchmark"}],
        }
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        benchmark = pd.DataFrame(
            {
                "Close": [90.0, 91.0, 92.0],
                "SMA50": [95.0, 95.0, 95.0],
                "SMA200": [100.0, 100.0, 100.0],
            },
            index=dates,
        )

        regime = evaluate_market_regime({"BENCH.MI": benchmark}, cfg, 60.0, date(2026, 1, 3))

        self.assertEqual(regime.state, "risk_off")
        self.assertFalse(regime.new_positions_allowed)
        self.assertEqual(regime.active_min_signal_score, 75.0)


class OpportunityReviewTests(unittest.TestCase):
    def test_overextended_breakout_becomes_watch(self) -> None:
        cfg = sample_config()
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="controlled_breakout",
            date="2026-05-01",
            price=12.0,
            entry=12.0,
            stop=11.0,
            target=14.0,
            reward_risk=2.0,
            qty=20,
            notional=240.0,
            estimated_round_trip_cost=3.0,
            score=82.0,
            score_details="base tecnico",
            meta={
                "close": 12.0,
                "sma50": 10.5,
                "sma200": 9.5,
                "rsi14": 58.0,
                "atr14": 1.0,
                "high20_prev": 10.0,
                "volume": 1300,
                "vol20": 1000,
            },
        )
        regime = {
            "state": "risk_on",
            "new_positions_allowed": True,
            "active_min_signal_score": 60.0,
        }

        reviewed = review_opportunity(signal, regime, cfg)

        self.assertEqual(reviewed.action, "WATCH")
        self.assertIn("breakout troppo esteso", reviewed.reason)
        self.assertEqual((reviewed.meta or {})["opportunity"]["decision"], "NO_GO")

    def test_clean_pullback_keeps_buy_decision(self) -> None:
        cfg = sample_config()
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-05-01",
            price=10.2,
            entry=10.2,
            stop=9.8,
            target=11.0,
            reward_risk=2.0,
            qty=30,
            notional=306.0,
            estimated_round_trip_cost=3.0,
            score=70.0,
            score_details="base tecnico",
            meta={
                "close": 10.2,
                "sma20": 10.0,
                "sma50": 9.6,
                "sma200": 9.0,
                "rsi14": 55.0,
                "atr14": 0.3,
                "volume": 1200,
                "vol20": 1000,
            },
        )
        regime = {
            "state": "risk_on",
            "new_positions_allowed": True,
            "active_min_signal_score": 60.0,
        }

        reviewed = review_opportunity(signal, regime, cfg)

        self.assertEqual(reviewed.action, "BUY")
        self.assertEqual((reviewed.meta or {})["opportunity"]["decision"], "GO")
        self.assertGreater(reviewed.score or 0, 70.0)


class SetupRadarTests(unittest.TestCase):
    def test_near_breakout_returns_watch_signal(self) -> None:
        cfg = sample_config()
        dates = pd.date_range("2026-01-01", periods=25, freq="D")
        df = pd.DataFrame(
            {
                "Open": [9.8] * 25,
                "High": [10.0] * 24 + [10.0],
                "Low": [9.5] * 25,
                "Close": [9.7] * 24 + [9.9],
                "SMA20": [9.4] * 25,
                "SMA50": [9.2] * 25,
                "SMA200": [8.8] * 25,
                "RSI14": [58.0] * 25,
                "ATR14": [0.3] * 25,
                "Volume": [1000] * 24 + [1200],
                "VOL20": [1000] * 25,
                "HIGH20_PREV": [10.0] * 25,
            },
            index=dates,
        )

        signals = analyze_buy_signals(
            {"symbol": "TEST.MI", "name": "Test", "type": "stock"},
            df,
            cfg["strategy"],
            date(2026, 1, 25),
        )

        self.assertEqual(signals[0].action, "WATCH")
        self.assertEqual(signals[0].strategy, "near_breakout_setup")
        self.assertGreaterEqual(signals[0].score or 0, 50.0)


class SignalJournalTests(unittest.TestCase):
    def test_signal_journal_deduplicates_and_evaluates_forward_returns(self) -> None:
        cfg = sample_config()
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-01-01",
            price=10.0,
            entry=10.0,
            stop=9.5,
            target=11.0,
            reward_risk=2.0,
            score=75.0,
            meta={"opportunity": {"decision": "GO", "grade": "B"}},
        )
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
        df = pd.DataFrame(
            {
                "Close": [10.0, 10.5, 11.2, 11.4],
                "High": [10.1, 10.7, 11.3, 11.5],
                "Low": [9.9, 10.2, 10.8, 11.1],
            },
            index=dates,
        )

        with TemporaryDirectory() as tmp:
            journal_path = Path(tmp) / "signal_journal.csv"
            evaluations_path = Path(tmp) / "signal_evaluations.csv"

            added = append_signal_journal(
                journal_path,
                [signal],
                {"state": "risk_on"},
                date(2026, 1, 1),
            )
            added_again = append_signal_journal(
                journal_path,
                [signal],
                {"state": "risk_on"},
                date(2026, 1, 1),
            )
            summary = update_signal_evaluations(
                journal_path,
                evaluations_path,
                {"TEST.MI": df},
                cfg,
                date(2026, 1, 4),
            )
            second_summary = update_signal_evaluations(
                journal_path,
                evaluations_path,
                {"TEST.MI": df},
                cfg,
                date(2026, 1, 5),
            )
            rows = pd.read_csv(evaluations_path)

        self.assertEqual(added, 1)
        self.assertEqual(added_again, 0)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(second_summary["new_or_updated"], 0)
        self.assertEqual(summary["positive_rate"], 100.0)
        self.assertIn("trend_pullback|GO|B", summary["best_bucket"])
        self.assertEqual(len(rows), 2)
        self.assertIn("target_hit", set(rows["outcome"]))


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

    def test_backtest_respects_risk_off_market_regime(self) -> None:
        cfg = sample_config()
        cfg["market_regime"] = {
            "enabled": True,
            "block_new_positions_when_risk_off": True,
            "benchmarks": [{"symbol": "BENCH.MI", "name": "Benchmark"}],
        }
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
        benchmark = pd.DataFrame(
            {
                "Close": [90.0, 91.0, 92.0],
                "SMA50": [95.0, 95.0, 95.0],
                "SMA200": [100.0, 100.0, 100.0],
            },
            index=dates,
        )

        def fake_analyze(instrument, df_slice, strategy_cfg, today):  # noqa: ARG001
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

        with patch("src.backtest.analyze_buy_signals", side_effect=fake_analyze):
            result = run_backtest(
                watchlist=[{"symbol": "TEST.MI", "name": "Test", "type": "stock"}],
                market_data={"TEST.MI": df},
                regime_data={"BENCH.MI": benchmark},
                config=cfg,
            )

        self.assertEqual(result.closed_trades, 0)
        self.assertEqual(len(result.open_positions), 0)
        self.assertEqual(result.regime_counts.get("risk_off"), 3)


if __name__ == "__main__":
    unittest.main()
