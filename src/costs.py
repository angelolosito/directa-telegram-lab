from __future__ import annotations

import math


def estimate_commission(notional: float, costs_config: dict) -> float:
    """Estimate order commission with a configurable fixed or variable model."""
    if notional <= 0:
        return 0.0
    fixed_commission = costs_config.get("fixed_commission")
    if fixed_commission is not None:
        return round(max(0.0, float(fixed_commission)), 2)

    variable_rate = float(costs_config.get("variable_rate", 0.0019))
    min_commission = float(costs_config.get("min_commission", 1.50))
    max_commission = float(costs_config.get("max_commission", 18.00))
    commission = notional * variable_rate
    return round(max(min_commission, min(commission, max_commission)), 2)


def estimate_round_trip_cost(notional: float, costs_config: dict) -> float:
    return round(estimate_commission(notional, costs_config) * 2, 2)


def max_affordable_quantity(
    entry_price: float,
    cash: float,
    max_allocation: float,
    costs_config: dict,
) -> float:
    if entry_price <= 0 or cash <= 0 or max_allocation <= 0:
        return 0.0

    if bool(costs_config.get("fractional_enabled", False)):
        precision = int(costs_config.get("quantity_precision", 6))
        min_notional = float(costs_config.get("min_order_notional", 1.0))
        notional_cap = min(cash, max_allocation)
        for _ in range(8):
            commission = estimate_commission(notional_cap, costs_config)
            next_cap = min(max_allocation, max(0.0, cash - commission))
            if abs(next_cap - notional_cap) < 0.01:
                notional_cap = next_cap
                break
            notional_cap = next_cap
        if notional_cap < min_notional:
            return 0.0
        scale = 10**precision
        qty = math.floor((notional_cap / entry_price) * scale) / scale
        return round(qty, precision) if qty * entry_price >= min_notional else 0.0

    qty = int(min(cash, max_allocation) // entry_price)
    while qty > 0:
        notional = qty * entry_price
        commission = estimate_commission(notional, costs_config)
        if notional <= max_allocation and notional + commission <= cash:
            return float(qty)
        qty -= 1
    return 0.0
