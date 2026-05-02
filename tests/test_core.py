from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.allocation import select_portfolio_candidates
from src.backtest import BacktestResult, BacktestTrade, run_backtest
from src.calibration import build_calibration_report
from src.fundamentals import FundamentalSnapshot, apply_fundamental_review, evaluate_fundamentals
from src.learning_feedback import apply_learning_feedback, load_learning_stats
from src.market_regime import evaluate_market_regime
from src.opportunity import review_opportunity
from src.paper_portfolio import PaperPortfolio
from src.relative_strength import (
    apply_relative_strength,
    benchmark_for_instrument,
    configured_relative_strength_benchmarks,
)
from src.report import build_daily_message
from src.scenario import apply_scenario, build_scenario_report, default_scenarios
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
        "allocation": {
            "enabled": True,
            "max_new_positions_per_run": 1,
            "max_same_sector_open": 1,
            "max_same_role_open": 1,
            "max_same_region_open": 2,
            "cautious_regime_states": ["neutral"],
            "cautious_etf_bonus": 4.0,
            "cautious_stock_penalty": 3.0,
            "leader_priority_bonus": 2.0,
            "strong_relative_bonus": 1.0,
            "very_strong_relative_bonus": 2.0,
            "weak_relative_penalty": 3.0,
            "max_cost_penalty_pct": 2.0,
        },
        "fundamentals": {
            "enabled": True,
            "provider": "yfinance",
            "apply_to_types": ["stock"],
            "cache_ttl_hours": 24,
            "request_timeout_seconds": 8,
            "process_timeout_seconds": 15,
            "report_errors": True,
            "block_when_weak": True,
            "block_when_missing": False,
            "block_below_score": 45.0,
            "weak_score": 45.0,
            "healthy_score": 60.0,
            "strong_score": 72.0,
            "score_adjustments": {"strong": 6.0, "healthy": 3.0, "mixed": 0.0, "weak": -8.0},
            "quality_gate": {
                "enabled": True,
                "block_on_critical_subscores": True,
                "max_critical_failures": 0,
                "min_subscores": {"profitability": 35.0, "balance_sheet": 35.0, "cashflow": 35.0},
            },
            "earnings_blackout": {
                "enabled": True,
                "days_before": 5,
                "days_after": 1,
                "action": "watch",
                "penalty_points": 5.0,
            },
            "weights": {
                "profitability": 0.25,
                "growth": 0.20,
                "balance_sheet": 0.18,
                "cashflow": 0.17,
                "valuation": 0.15,
                "shareholder": 0.03,
                "analyst": 0.02,
            },
        },
        "backtest": {"max_new_positions_per_day": 1},
    }


def sample_buy_signal(
    symbol: str,
    score: float,
    instrument_type: str = "stock",
    sector: str = "technology",
    region: str = "us_growth",
    role: str = "leader",
    priority: str = "leader",
) -> Signal:
    return Signal(
        symbol=symbol,
        name=symbol,
        instrument_type=instrument_type,
        action="BUY",
        strategy="trend_pullback",
        date="2026-05-01",
        price=100.0,
        entry=100.0,
        stop=95.0,
        target=110.0,
        reward_risk=2.0,
        qty=4,
        notional=400.0,
        estimated_round_trip_cost=3.0,
        score=score,
        score_details="base tecnico",
        meta={
            "sector": sector,
            "region": region,
            "role": role,
            "priority": priority,
            "currency": "EUR",
            "base_currency": "EUR",
        },
    )


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


class AllocationTests(unittest.TestCase):
    def test_allocation_blocks_duplicate_sector_against_open_positions(self) -> None:
        cfg = sample_config()
        candidates = [
            sample_buy_signal("NVDA", 90.0, sector="semiconductors", role="ai_leader"),
            sample_buy_signal("V", 82.0, sector="payments", region="us", role="compounder"),
        ]
        open_contexts = [{"symbol": "ASML.AS", "sector": "semiconductors", "region": "europe", "role": "european_leader"}]

        result = select_portfolio_candidates(
            candidates,
            open_contexts,
            {"state": "risk_on"},
            cfg,
            max_new_positions=1,
        )

        self.assertEqual([signal.symbol for signal in result.selected], ["V"])
        self.assertEqual(candidates[0].action, "WATCH")
        self.assertIn("settore gia coperto", candidates[0].reason)
        self.assertEqual((candidates[1].meta or {})["allocation"]["decision"], "SELECTED")

    def test_allocation_prefers_etf_when_market_is_neutral(self) -> None:
        cfg = sample_config()
        stock = sample_buy_signal("AAPL", 80.0, instrument_type="stock", sector="consumer_technology")
        etf = sample_buy_signal(
            "CSSPX.MI",
            78.0,
            instrument_type="etf",
            sector="broad_market",
            region="us",
            role="core_us",
            priority="core",
        )

        result = select_portfolio_candidates(
            [stock, etf],
            [],
            {"state": "neutral"},
            cfg,
            max_new_positions=1,
        )

        self.assertEqual([signal.symbol for signal in result.selected], ["CSSPX.MI"])
        self.assertEqual(stock.action, "WATCH")
        self.assertEqual((etf.meta or {})["allocation"]["decision"], "SELECTED")


class DecisionReportTests(unittest.TestCase):
    def test_daily_report_states_wait_when_only_candidates_exist(self) -> None:
        candidate = sample_buy_signal("TEST", 68.0)
        candidate.action = "WATCH"

        message = build_daily_message(
            run_date=date(2026, 5, 1),
            buy_signals=[],
            candidate_signals=[candidate],
            close_events=[],
            trail_events=[],
            summary={
                "cash": 1000.0,
                "open_market_value": 0.0,
                "equity": 1000.0,
                "open_positions": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "open_risk_to_stop": 0.0,
            },
            errors=[],
            market_regime={"enabled": True, "state": "risk_on", "new_positions_allowed": True},
        )

        self.assertIn("Decisione operativa", message)
        self.assertIn("Verdetto: WAIT selettivo", message)
        self.assertIn("Miglior candidato in osservazione: TEST", message)


class FundamentalAnalysisTests(unittest.TestCase):
    def test_fundamentals_reward_strong_company(self) -> None:
        cfg = sample_config()
        snapshot = FundamentalSnapshot(
            symbol="TEST",
            provider="unit",
            source_symbol="TEST",
            fetched_at="2026-05-01T00:00:00+00:00",
            metrics={
                "profit_margin": 0.25,
                "operating_margin": 0.22,
                "roe": 0.28,
                "revenue_growth": 0.18,
                "earnings_growth": 0.20,
                "debt_to_equity": 20.0,
                "current_ratio": 1.8,
                "operating_cashflow": 1_000_000.0,
                "free_cashflow": 700_000.0,
                "forward_pe": 22.0,
                "price_to_sales": 5.0,
                "ev_to_ebitda": 16.0,
                "payout_ratio": 0.25,
                "recommendation_mean": 1.8,
                "eps_revision_balance": 5.0,
            },
            events={"next_earnings_date": "2026-07-25"},
        )
        signal = sample_buy_signal("TEST", 70.0)

        review = evaluate_fundamentals(snapshot, cfg)
        reviewed = apply_fundamental_review(signal, {"symbol": "TEST", "type": "stock"}, snapshot, cfg)

        self.assertIn(review.state, {"strong", "healthy"})
        self.assertGreater(reviewed.score or 0, 70.0)
        self.assertEqual((reviewed.meta or {})["fundamentals"]["state"], review.state)
        self.assertIn("fondamentali", reviewed.score_details)

    def test_fundamentals_block_weak_company(self) -> None:
        cfg = sample_config()
        snapshot = FundamentalSnapshot(
            symbol="TEST",
            provider="unit",
            source_symbol="TEST",
            fetched_at="2026-05-01T00:00:00+00:00",
            metrics={
                "profit_margin": -0.05,
                "operating_margin": -0.08,
                "roe": -0.12,
                "revenue_growth": -0.20,
                "earnings_growth": -0.35,
                "debt_to_equity": 350.0,
                "current_ratio": 0.5,
                "operating_cashflow": -100_000.0,
                "free_cashflow": -120_000.0,
                "trailing_pe": 90.0,
                "price_to_sales": 20.0,
                "ev_to_ebitda": 40.0,
                "payout_ratio": 1.2,
                "recommendation_mean": 4.0,
                "eps_revision_balance": -5.0,
            },
            events={},
        )
        signal = sample_buy_signal("TEST", 70.0)

        reviewed = apply_fundamental_review(signal, {"symbol": "TEST", "type": "stock"}, snapshot, cfg)

        self.assertEqual(reviewed.action, "WATCH")
        self.assertLess((reviewed.meta or {})["fundamentals"]["score"], 45.0)
        self.assertIn("Fondamentali deboli", reviewed.reason)

    def test_unknown_fundamentals_do_not_block_by_default(self) -> None:
        cfg = sample_config()
        snapshot = FundamentalSnapshot(
            symbol="TEST",
            provider="unit",
            source_symbol="TEST",
            fetched_at="2026-05-01T00:00:00+00:00",
            metrics={},
            events={},
            error="provider temporaneamente non disponibile",
        )
        signal = sample_buy_signal("TEST", 70.0)

        reviewed = apply_fundamental_review(signal, {"symbol": "TEST", "type": "stock"}, snapshot, cfg)

        self.assertEqual(reviewed.action, "BUY")
        self.assertIsNone((reviewed.meta or {})["fundamentals"]["score"])
        self.assertEqual((reviewed.meta or {})["fundamentals"]["state"], "unknown")

    def test_quality_gate_blocks_critical_subscore(self) -> None:
        cfg = sample_config()
        snapshot = FundamentalSnapshot(
            symbol="TEST",
            provider="unit",
            source_symbol="TEST",
            fetched_at="2026-05-01T00:00:00+00:00",
            metrics={
                "profit_margin": 0.25,
                "operating_margin": 0.22,
                "roe": 0.28,
                "revenue_growth": 0.18,
                "earnings_growth": 0.20,
                "debt_to_equity": 20.0,
                "current_ratio": 1.8,
                "operating_cashflow": -100_000.0,
                "free_cashflow": -120_000.0,
                "forward_pe": 22.0,
                "price_to_sales": 5.0,
                "ev_to_ebitda": 16.0,
                "payout_ratio": 0.25,
                "recommendation_mean": 1.8,
            },
            events={},
        )
        signal = sample_buy_signal("TEST", 70.0)

        reviewed = apply_fundamental_review(signal, {"symbol": "TEST", "type": "stock"}, snapshot, cfg)

        self.assertEqual(reviewed.action, "WATCH")
        self.assertTrue((reviewed.meta or {})["fundamentals"]["quality_gate"]["blocked"])
        self.assertIn("Quality gate fondamentale", reviewed.reason)

    def test_earnings_blackout_blocks_near_quarterly_report(self) -> None:
        cfg = sample_config()
        snapshot = FundamentalSnapshot(
            symbol="TEST",
            provider="unit",
            source_symbol="TEST",
            fetched_at="2026-05-01T00:00:00+00:00",
            metrics={
                "profit_margin": 0.25,
                "operating_margin": 0.22,
                "roe": 0.28,
                "revenue_growth": 0.18,
                "earnings_growth": 0.20,
                "debt_to_equity": 20.0,
                "current_ratio": 1.8,
                "operating_cashflow": 1_000_000.0,
                "free_cashflow": 700_000.0,
                "forward_pe": 22.0,
                "price_to_sales": 5.0,
                "ev_to_ebitda": 16.0,
                "payout_ratio": 0.25,
                "recommendation_mean": 1.8,
            },
            events={"next_earnings_date": "2026-05-04"},
        )
        signal = sample_buy_signal("TEST", 70.0)
        signal.date = "2026-05-01"

        reviewed = apply_fundamental_review(signal, {"symbol": "TEST", "type": "stock"}, snapshot, cfg)

        self.assertEqual(reviewed.action, "WATCH")
        self.assertTrue((reviewed.meta or {})["fundamentals"]["earnings_blackout"]["active"])
        self.assertIn("Trimestrale troppo vicina", reviewed.reason)


class CalibrationTests(unittest.TestCase):
    def test_calibration_report_flags_small_sample_and_silent_watchlist(self) -> None:
        cfg = sample_config()
        result = BacktestResult(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 4, 1),
            initial_capital=1000.0,
            ending_equity=995.0,
            realized_pnl=-5.0,
            total_return_pct=-0.5,
            max_drawdown_pct=-2.0,
            trades=[
                BacktestTrade(
                    symbol="AAPL",
                    name="Apple",
                    instrument_type="stock",
                    strategy="trend_pullback",
                    entry_date=date(2026, 1, 10),
                    exit_date=date(2026, 1, 20),
                    entry_price=100.0,
                    exit_price=98.0,
                    qty=2,
                    exit_reason="stop_loss",
                    gross_pnl=-4.0,
                    net_pnl=-5.0,
                    meta={"region": "us_growth", "sector": "consumer_technology"},
                )
            ],
            open_positions=[],
            errors=[],
            equity_curve=[],
            regime_counts={"risk_on": 40, "neutral": 20},
        )

        report = build_calibration_report(
            result,
            [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
            cfg,
        )

        self.assertIn("Calibration Report", report)
        self.assertIn("Campione ancora piccolo", report)
        self.assertIn("Watchlist silenziosa", report)
        self.assertIn("MSFT", report)

    def test_calibration_report_groups_trades_by_sector_and_region(self) -> None:
        cfg = sample_config()
        trades = [
            BacktestTrade(
                symbol="NVDA",
                name="NVIDIA",
                instrument_type="stock",
                strategy="controlled_breakout",
                entry_date=date(2026, 1, 10),
                exit_date=date(2026, 1, 20),
                entry_price=100.0,
                exit_price=110.0,
                qty=2,
                exit_reason="target_reached",
                gross_pnl=20.0,
                net_pnl=17.0,
                meta={"region": "us_growth", "sector": "semiconductors"},
            ),
            BacktestTrade(
                symbol="ASML.AS",
                name="ASML",
                instrument_type="stock",
                strategy="controlled_breakout",
                entry_date=date(2026, 2, 10),
                exit_date=date(2026, 2, 20),
                entry_price=100.0,
                exit_price=108.0,
                qty=2,
                exit_reason="target_reached",
                gross_pnl=16.0,
                net_pnl=13.0,
                meta={"region": "europe", "sector": "semiconductors"},
            ),
        ]
        result = BacktestResult(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 4, 1),
            initial_capital=1000.0,
            ending_equity=1030.0,
            realized_pnl=30.0,
            total_return_pct=3.0,
            max_drawdown_pct=-1.0,
            trades=trades,
            open_positions=[],
            errors=[],
            equity_curve=[],
            regime_counts={},
        )

        report = build_calibration_report(
            result,
            [{"symbol": "NVDA"}, {"symbol": "ASML.AS"}],
            cfg,
        )

        self.assertIn("Settori", report)
        self.assertIn("semiconductors", report)
        self.assertIn("Tipi strumento", report)
        self.assertIn("stock", report)


class ScenarioReportTests(unittest.TestCase):
    def test_apply_scenario_does_not_mutate_base_config(self) -> None:
        cfg = sample_config()
        spec = default_scenarios()[1]

        scenario_cfg = apply_scenario(cfg, spec)

        self.assertEqual(cfg["strategy"]["min_signal_score"], 60.0)
        self.assertEqual(scenario_cfg["strategy"]["min_signal_score"], 65)

    def test_scenario_report_ranks_best_scenario_from_backtest_results(self) -> None:
        cfg = sample_config()
        cfg["scenario_report"] = {
            "max_scenarios": 3,
            "min_trades_for_ranking": 1,
            "max_monthly_trades": 4.0,
            "max_drawdown_warn_pct": 8.0,
            "drawdown_weight": 1.0,
            "profit_factor_weight": 6.0,
            "small_sample_penalty": 2.0,
            "overtrade_penalty": 2.0,
        }

        def fake_backtest(watchlist, market_data, config, regime_data=None, relative_strength_data=None):  # noqa: ARG001
            min_score = config["strategy"]["min_signal_score"]
            if min_score == 70:
                total_return = 4.0
                pnl = 40.0
                exit_price = 120.0
            elif min_score == 65:
                total_return = 8.0
                pnl = 80.0
                exit_price = 140.0
            else:
                total_return = -2.0
                pnl = -20.0
                exit_price = 90.0
            trade = BacktestTrade(
                symbol="TEST.MI",
                name="Test",
                instrument_type="stock",
                strategy="trend_pullback",
                entry_date=date(2026, 1, 1),
                exit_date=date(2026, 2, 1),
                entry_price=100.0,
                exit_price=exit_price,
                qty=1,
                exit_reason="target_reached" if pnl > 0 else "stop_loss",
                gross_pnl=pnl,
                net_pnl=pnl,
                meta={"region": "europe", "sector": "test"},
            )
            return BacktestResult(
                start_date=date(2026, 1, 1),
                end_date=date(2026, 4, 1),
                initial_capital=1000.0,
                ending_equity=1000.0 + pnl,
                realized_pnl=pnl,
                total_return_pct=total_return,
                max_drawdown_pct=-1.0,
                trades=[trade],
                open_positions=[],
                errors=[],
                equity_curve=[],
                regime_counts={},
            )

        with patch("src.scenario.run_backtest", side_effect=fake_backtest):
            report = build_scenario_report(
                [{"symbol": "TEST.MI"}],
                {"TEST.MI": pd.DataFrame({"Close": [1.0]})},
                cfg,
            )

        self.assertIn("Scenario Report", report)
        self.assertIn("Scenario: `quality_65`", report)
        self.assertIn("`strategy.min_signal_score` -> `65`", report)


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
