"""
Simulation Configuration
========================
Central dataclass holding all tunable parameters for the agent economy simulation.
All results produced by this simulator are statements about the simulation's
internal dynamics — never predictions about real companies, markets, or events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """Immutable configuration for a single simulation run.

    Attributes are grouped by domain: population, firms, market mechanics,
    and timing. Override any field via keyword arguments or by subclassing.
    """

    # ── Population ──────────────────────────────────────────────────────
    num_agents: int = 200
    base_wage_min: float = 30.0
    base_wage_max: float = 80.0
    initial_budget_min: float = 100.0
    initial_budget_max: float = 500.0
    savings_rate_min: float = 0.05
    savings_rate_max: float = 0.30
    risk_aversion_min: float = 0.1
    risk_aversion_max: float = 0.9

    # ── Firms ───────────────────────────────────────────────────────────
    num_firms: int = 5
    num_goods: int = 4
    initial_firm_cash_min: float = 5_000.0
    initial_firm_cash_max: float = 20_000.0
    base_price_min: float = 5.0
    base_price_max: float = 25.0
    production_capacity_min: float = 80.0
    production_capacity_max: float = 200.0
    productivity_per_worker: float = 12.0
    input_cost_base: float = 3.0
    target_inventory_buffer: float = 1.3  # multiplier on expected demand

    # ── Market Mechanics ────────────────────────────────────────────────
    price_adjustment_rate: float = 0.03
    wage_adjustment_rate: float = 0.02
    demand_elasticity: float = 1.2
    awareness_threshold: float = 0.1  # min awareness to consider a good
    memory_window: int = 10  # ticks of price/purchase history kept

    # ── Timing ──────────────────────────────────────────────────────────
    num_ticks: int = 120

    # ── Experiment Defaults ─────────────────────────────────────────────
    default_num_seeds: int = 30
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000

    # ── Parallelism ─────────────────────────────────────────────────────
    max_workers: Optional[int] = None  # None → os.cpu_count()

    def copy(self, **overrides) -> "SimulationConfig":
        """Return a shallow copy with selected fields overridden."""
        from dataclasses import asdict
        d = asdict(self)
        d.update(overrides)
        return SimulationConfig(**d)
