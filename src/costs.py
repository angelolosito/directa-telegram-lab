from __future__ import annotations


def estimate_commission(notional: float, costs_config: dict) -> float:
    """Estimate Directa order commission with a configurable variable model."""
    variable_rate = float(costs_config.get("variable_rate", 0.0019))
    min_commission = float(costs_config.get("min_commission", 1.50))
    max_commission = float(costs_config.get("max_commission", 18.00))
    commission = notional * variable_rate
    return round(max(min_commission, min(commission, max_commission)), 2)


def estimate_round_trip_cost(notional: float, costs_config: dict) -> float:
    return round(estimate_commission(notional, costs_config) * 2, 2)
