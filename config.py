"""
Simulation Configuration
========================
Central dataclass holding all tunable parameters for the agent economy simulation.
All results produced by this simulator are statements about the simulation's
internal dynamics — never predictions about real companies, markets, or events.
"""

from __future__ import annotations

import jax
from flax import struct
from typing import Optional

@struct.dataclass
class SimulationConfig:
    """Immutable configuration for a single simulation run.

    Attributes are grouped by domain: population, firms, market mechanics,
    and timing. Override any field via keyword arguments or by subclassing.
    """

    # ── Simulation Modes ────────────────────────────────────────────────
    use_us_calibration: bool = struct.field(pytree_node=False, default=False)
    # We remove calibration_kwargs to keep the config purely hashable JAX leaves,
    # calibration parameters will be handled outside the core JAX config or passed explicitly.

    # ── Population ──────────────────────────────────────────────────────
    num_agents: int = struct.field(pytree_node=False, default=200)
    base_wage_min: float = 30.0
    base_wage_max: float = 80.0
    initial_budget_min: float = 100.0
    initial_budget_max: float = 500.0
    savings_rate_min: float = 0.05
    savings_rate_max: float = 0.30
    risk_aversion_min: float = 0.1
    risk_aversion_max: float = 0.9

    # ── Firms ───────────────────────────────────────────────────────────
    num_firms: int = struct.field(pytree_node=False, default=5)
    num_goods: int = struct.field(pytree_node=False, default=4)
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
    ces_elasticity: float = 0.8  # Sigma in CES demand function
    awareness_threshold: float = 0.1  # min awareness to consider a good
    memory_window: int = 10  # ticks of price/purchase history kept

    # ── Labor Market (DMP) ──────────────────────────────────────────────
    matching_efficiency: float = 0.6  # mu in M = mu * U^alpha * V^(1-alpha)
    matching_elasticity: float = 0.5  # alpha
    bargaining_power_agent: float = 0.5
    vacancy_cost: float = 5.0

    # ── Finance & Banking ───────────────────────────────────────────────
    central_bank_base_rate: float = 0.02
    commercial_bank_spread: float = 0.03
    reserve_requirement: float = 0.10
    
    # ── Government & Taxation ──────────────────────────────────────────
    corporate_tax_rate: float = 0.20
    income_tax_rate_base: float = 0.10
    income_tax_rate_top: float = 0.35
    income_tax_bracket_threshold: float = 100.0
    unemployment_benefit: float = 15.0
    minimum_wage: float = 12.0
    
    # ── Capital & Bankruptcy ───────────────────────────────────────────
    capital_cost: float = 50.0
    capital_depreciation: float = 0.05
    firm_entry_probability: float = 0.05
    bankruptcy_threshold: float = 0.0
    
    # ── Demographics & Skills ──────────────────────────────────────────
    agent_mortality_rate: float = 0.01
    skill_min: float = 0.5
    skill_max: float = 2.0
    
    # ── Geography & Housing (Module 3) ─────────────────────────────────
    num_regions: int = struct.field(pytree_node=False, default=3)
    housing_depreciation: float = 0.02
    housing_base_price: float = 200.0
    
    # ── Foreign Trade (Module 3) ───────────────────────────────────────
    foreign_demand_base: float = 50.0
    exchange_rate_volatility: float = 0.05
    
    # ── Menu Costs & Expectations (Module 3) ───────────────────────────
    menu_cost: float = 2.0
    expectation_alpha: float = 0.3
    
    # ── Insurance & Network (Module 3) ─────────────────────────────────
    insurance_premium: float = 5.0
    insurance_payout: float = 50.0
    network_influence: float = 0.1

    # ── Timing ──────────────────────────────────────────────────────────
    num_ticks: int = struct.field(pytree_node=False, default=120)

    # ── Baselines ───────────────────────────────────────────────────────
    firm_behavior_mode: int = struct.field(pytree_node=False, default=0) # 0: LMM, 1: ZI, 2: Heuristic

    # ── Experiment Defaults ─────────────────────────────────────────────
    default_num_seeds: int = 30
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000

    # ── Parallelism ─────────────────────────────────────────────────────
    max_workers: Optional[int] = None  # None → os.cpu_count()

    def copy(self, **overrides) -> "SimulationConfig":
        """Return a shallow copy with selected fields overridden."""
        return self.replace(**overrides)
