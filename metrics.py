"""
Metrics Computation
===================
KPI functions for measuring simulation outcomes.  All metrics describe the
simulation's internal state — they are not predictions about real markets.

Metrics computed:
  • Gini coefficient (wealth inequality)
  • Price index (average/weighted price level)
  • Employment rate
  • Per-firm revenue, profit, market share
  • Agent welfare (realized utility)
  • Aggregate tick-level summary
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from core import Agent, Firm


def compute_gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values.

    Uses the relative mean absolute difference formula:
        G = (Σ_i Σ_j |x_i - x_j|) / (2 n² μ)

    Returns 0 for perfect equality, approaches 1 for perfect inequality.
    """
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]  # filter negatives
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_v) - (n + 1) * np.sum(sorted_v)) / (n * np.sum(sorted_v)))


def compute_price_index(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Compute volume-weighted average price across goods.

    Falls back to simple mean if no volume data.
    """
    if volumes is not None and volumes.sum() > 0:
        return float(np.average(prices, weights=volumes))
    return float(np.mean(prices)) if len(prices) > 0 else 0.0


def compute_employment_rate(agents: List["Agent"]) -> float:
    """Fraction of agents currently employed."""
    if not agents:
        return 0.0
    employed = sum(1 for a in agents if a.employed)
    return employed / len(agents)


def compute_firm_metrics(firms: List["Firm"]) -> List[Dict]:
    """Per-firm metrics: revenue, profit, market share, inventory, employees."""
    total_revenue = sum(f.cumulative_revenue for f in firms) or 1.0
    return [
        {
            "firm_id": f.id,
            "good_produced": f.good_produced,
            "price": f.price,
            "inventory": f.inventory,
            "num_employees": len(f.employees),
            "cumulative_revenue": f.cumulative_revenue,
            "cumulative_cost": f.cumulative_cost,
            "profit": f.cumulative_revenue - f.cumulative_cost,
            "market_share": f.cumulative_revenue / total_revenue,
            "cash": f.cash,
        }
        for f in firms
    ]


def compute_total_welfare(agents: List["Agent"]) -> float:
    """Total agent welfare = sum of budget + savings (proxy for realized utility).

    A more sophisticated version would compute Cobb-Douglas utility from
    actual consumption bundles; this proxy tracks wealth accumulation.
    """
    return sum(a.budget + a.savings for a in agents)


def compute_total_output(market_data: Dict) -> float:
    """Total goods transacted (volume) this tick."""
    return float(np.sum(market_data.get("volumes", [0])))


def compute_average_wage(agents: List["Agent"]) -> float:
    """Average wage among employed agents."""
    wages = [a.wage for a in agents if a.employed and a.wage > 0]
    return float(np.mean(wages)) if wages else 0.0


def aggregate_tick_metrics(
    agents: List["Agent"],
    firms: List["Firm"],
    market_data: Dict,
    tick: int,
) -> Dict:
    """Compute all KPIs for a single tick and return as a flat dict.

    This is the main entry point called by the simulation loop.
    """
    budgets = np.array([a.budget + a.savings for a in agents])
    prices = market_data.get("prices", np.array([]))
    volumes = market_data.get("volumes", np.array([]))

    return {
        "tick": tick,
        "gini": compute_gini(budgets),
        "price_index": compute_price_index(prices, volumes),
        "employment_rate": compute_employment_rate(agents),
        "total_welfare": compute_total_welfare(agents),
        "total_output": compute_total_output(market_data),
        "average_wage": compute_average_wage(agents),
        "prices": prices.tolist() if hasattr(prices, "tolist") else list(prices),
        "volumes": volumes.tolist() if hasattr(volumes, "tolist") else list(volumes),
        "unmet_demand": market_data.get("unmet_demand", np.array([])).tolist(),
        "total_demand": market_data.get("total_demand", np.array([])).tolist(),
        "firm_metrics": compute_firm_metrics(firms),
        "num_agents": len(agents),
        "num_firms": len(firms),
    }
