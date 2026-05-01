from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.backtest import run_backtest
from src.learning_feedback import apply_learning_feedback, load_learning_stats
from src.market_regime import evaluate_market_regime
from src.opportunity import review_opportunity
from src.paper_portfolio import PaperPortfolio
from src.relative_strength import (
    apply_relative_strength,
    benchmark_for_instrument,
    configured_relative_strength_benchmarks,
)
from src.signal_journal import append_signal_journal, build_learning_report, update_signal_evaluations
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
            "adaptive_feedback_enabled": True,
            "adaptive_min_samples": 2,
            "adaptive_positive_rate_floor": 45.0,
            "adaptive_avg_return_floor_pct": -0.5,
            "adaptive_positive_rate_good": 58.0,
            "adaptive_avg_return_good_pct": 0.5,
            "adaptive_penalty_points": 6.0,
            "adaptive_bonus_points": 3.0,
        },
        "relative_strength": {
            "enabled": True,
            "lookback_sessions": 3,
            "default_benchmark": "BENCH.MI",
            "benchmark_by_type": {"stock": "BENCH.MI", "etf": "BENCH.MI"},
            "weak_threshold_pct": -2.0,
            "strong_threshold_pct": 2.0,
            "very_strong_threshold_pct": 5.0,
            "penalty_points": 8.0,
            "bonus_points": 4.0,
            "strong_bonus_points": 7.0,
            "block_when_weak": False,
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

    def test_position_size_converts_foreign_notional_to_base_currency(self) -> None:
        cfg = sample_config()
        with TemporaryDirectory() as tmp:
            portfolio = PaperPortfolio(Path(tmp) / "lab.sqlite", cfg)
            signal = Signal(
                symbol="AAPL",
                name="Apple",
                instrument_type="stock",
                action="BUY",
                strategy="trend_pullback",
                date="2026-05-01",
                price=100.0,
                entry=100.0,
                stop=95.0,
                target=110.0,
                reward_risk=2.0,
                meta={"currency": "USD", "base_currency": "EUR", "fx_to_base": 0.9},
            )
            sized = portfolio.size_signal(signal)
            portfolio.close()

        self.assertEqual(sized.qty, 5)
        self.assertEqual(sized.notional, 450.0)

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

    def test_signal_score_includes_curated_universe_bonus(self) -> None:
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
            stop=9.6,
            target=10.8,
            reward_risk=2.0,
            qty=20,
            notional=200.0,
            estimated_round_trip_cost=3.0,
            meta={
                "close": 10.0,
                "sma50": 9.5,
                "sma200": 9.0,
                "rsi14": 55.0,
                "volume": 1200,
                "vol20": 1000,
                "universe_score": 5,
            },
        )
        scored = score_signal(signal, cfg["strategy"])

        self.assertEqual((scored.meta or {})["score_breakdown"]["universe"], 4.0)
        self.assertIn("universo +4.0", scored.score_details)


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


class RelativeStrengthTests(unittest.TestCase):
    def test_relative_strength_uses_explicit_and_regional_benchmarks(self) -> None:
        cfg = sample_config()
        cfg["relative_strength"]["benchmark_by_region"] = {"us_growth": "EQQQ.MI"}
        watchlist = [
            {"symbol": "AAPL", "type": "stock", "benchmark": "EQQQ.MI"},
            {"symbol": "JPM", "type": "stock", "region": "us", "benchmark": "CSSPX.MI"},
        ]

        benchmarks = configured_relative_strength_benchmarks(cfg, watchlist)
        symbols = [item["symbol"] for item in benchmarks]

        self.assertIn("EQQQ.MI", symbols)
        self.assertIn("CSSPX.MI", symbols)
        self.assertEqual(benchmark_for_instrument(watchlist[0], cfg), "EQQQ.MI")
        self.assertEqual(benchmark_for_instrument({"symbol": "NVDA", "region": "us_growth"}, cfg), "EQQQ.MI")

    def test_relative_strength_penalizes_lagging_signal(self) -> None:
        cfg = sample_config()
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        instrument_df = pd.DataFrame({"Close": [100.0, 100.0, 99.0, 99.0, 99.0]}, index=dates)
        benchmark_df = pd.DataFrame({"Close": [100.0, 101.0, 103.0, 105.0, 106.0]}, index=dates)
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-01-05",
            price=99.0,
            entry=99.0,
            stop=95.0,
            target=107.0,
            reward_risk=2.0,
            score=75.0,
            score_details="base tecnico",
        )

        reviewed = apply_relative_strength(
            signal,
            {"symbol": "TEST.MI", "name": "Test", "type": "stock"},
            instrument_df,
            {"BENCH.MI": benchmark_df},
            cfg,
        )

        self.assertEqual(reviewed.score, 67.0)
        self.assertEqual((reviewed.meta or {})["relative_strength"]["state"], "weak")
        self.assertIn("Forza relativa debole", reviewed.reason)

    def test_relative_strength_rewards_leader_signal(self) -> None:
        cfg = sample_config()
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        instrument_df = pd.DataFrame({"Close": [100.0, 103.0, 106.0, 109.0, 112.0]}, index=dates)
        benchmark_df = pd.DataFrame({"Close": [100.0, 100.5, 101.0, 101.5, 102.0]}, index=dates)
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-01-05",
            price=112.0,
            entry=112.0,
            stop=107.0,
            target=122.0,
            reward_risk=2.0,
            score=75.0,
            score_details="base tecnico",
        )

        reviewed = apply_relative_strength(
            signal,
            {"symbol": "TEST.MI", "name": "Test", "type": "stock"},
            instrument_df,
            {"BENCH.MI": benchmark_df},
            cfg,
        )

        self.assertEqual(reviewed.score, 82.0)
        self.assertEqual((reviewed.meta or {})["relative_strength"]["state"], "very_strong")
        self.assertIn("Forza relativa favorevole", reviewed.reason)


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
            report = build_learning_report(journal_path, evaluations_path, cfg)
            rows = pd.read_csv(evaluations_path)

        self.assertEqual(added, 1)
        self.assertEqual(added_again, 0)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(second_summary["new_or_updated"], 0)
        self.assertEqual(summary["positive_rate"], 100.0)
        self.assertIn("trend_pullback|GO|B", summary["best_bucket"])
        self.assertIn("Diario intelligente segnali", report)
        self.assertIn("Setup migliori", report)
        self.assertEqual(len(rows), 2)
        self.assertIn("target_hit", set(rows["outcome"]))

    def test_learning_feedback_penalizes_weak_signal_family(self) -> None:
        cfg = sample_config()
        evaluations = pd.DataFrame(
            [
                {
                    "evaluation_id": f"weak-{idx}",
                    "signal_id": f"sig-{idx}",
                    "date": "2026-01-01",
                    "symbol": "TEST.MI",
                    "instrument_type": "stock",
                    "strategy": "trend_pullback",
                    "action": "BUY",
                    "score": "72.0",
                    "opportunity_decision": "GO",
                    "opportunity_grade": "B",
                    "market_regime": "risk_on",
                    "horizon_sessions": "2",
                    "entry_price": "10.0000",
                    "end_date": "2026-01-03",
                    "close_return_pct": "-1.20",
                    "max_gain_pct": "0.30",
                    "max_drawdown_pct": "-2.00",
                    "hit_target": "false",
                    "hit_stop": "true",
                    "outcome": "stop_hit",
                    "updated_at": "2026-01-03",
                }
                for idx in range(2)
            ]
        )
        signal = Signal(
            symbol="TEST.MI",
            name="Test",
            instrument_type="stock",
            action="BUY",
            strategy="trend_pullback",
            date="2026-01-04",
            price=10.2,
            entry=10.2,
            stop=9.8,
            target=11.0,
            reward_risk=2.0,
            score=72.0,
            score_details="base tecnico",
            meta={"opportunity": {"decision": "GO", "grade": "B"}},
        )

        with TemporaryDirectory() as tmp:
            evaluations_path = Path(tmp) / "signal_evaluations.csv"
            evaluations.to_csv(evaluations_path, index=False)
            stats = load_learning_stats(evaluations_path, cfg)

        reviewed = apply_learning_feedback(signal, stats, {"state": "risk_on"}, cfg)

        self.assertEqual(reviewed.score, 66.0)
        self.assertEqual((reviewed.meta or {})["learning_feedback"]["verdict"], "weak")
        self.assertIn("feedback diario -6.0", reviewed.score_details)


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
