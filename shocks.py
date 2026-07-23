"""
Shocks Library
==============
Provides basic deterministic economic shocks (tariffs, rate hikes, demand shifts)
that can be injected into the simulation state at runtime.

This replaces the untrained LLM-shock layer, ensuring reproducible and 
economically valid shock dynamics for version 1.
"""

import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

def apply_interest_rate_hike(state: SimState, hike_amount: float = 0.02) -> SimState:
    """Increases the central bank base rate."""
    new_macro = state.macro._replace(base_rate=state.macro.base_rate + hike_amount)
    return state._replace(macro=new_macro)

def apply_demand_shock(state: SimState, savings_rate_increase: float = 0.05) -> SimState:
    """Increases agent savings rates, suppressing aggregate demand."""
    new_savings_rate = jnp.clip(state.agents.savings_rate + savings_rate_increase, 0.0, 0.9)
    new_agents = state.agents._replace(savings_rate=new_savings_rate)
    return state._replace(agents=new_agents)

def apply_supply_chain_disruption(state: SimState, cost_multiplier: float = 1.2) -> SimState:
    """Increases input costs for all firms."""
    new_input_cost = state.firms.input_cost_multiplier * cost_multiplier
    new_firms = state.firms._replace(input_cost_multiplier=new_input_cost)
    return state._replace(firms=new_firms)

def apply_technology_breakthrough(state: SimState, productivity_boost: float = 1.3) -> SimState:
    """Increases production capacity and quality for all active firms."""
    new_capacity = state.firms.production_capacity * productivity_boost
    new_quality = state.firms.quality * 1.1
    new_firms = state.firms._replace(
        production_capacity=new_capacity,
        quality=new_quality
    )
    return state._replace(firms=new_firms)
