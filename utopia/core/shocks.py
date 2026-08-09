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
from utopia.core.config import SimulationConfig
from utopia.core.state import SimState

def apply_interest_rate_hike(state: SimState, hike_amount: float = 0.02) -> SimState:
    """Increases the macro base rate."""
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

def apply_physical_telematics_shock(state: SimState, cost_multiplier: float) -> SimState:
    """
    Applies a physical telematics shock (e.g., weather, port congestion) 
    derived from the PhysicalShockCompiler.
    """
    # Physical shocks spike input costs and severely disrupt production capacity
    new_input_cost = state.firms.input_cost_multiplier * cost_multiplier
    # A 1.5x cost multiplier implies a 50% drop in capacity due to logistics failures
    capacity_penalty = 1.0 / cost_multiplier 
    new_capacity = state.firms.production_capacity * capacity_penalty
    
    new_firms = state.firms._replace(
        input_cost_multiplier=new_input_cost,
        production_capacity=new_capacity
    )
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

def apply_port_congestion_shock(state: SimState, severity: float = 0.5) -> SimState:
    """Reduces firm production_capacity proportionally."""
    new_capacity = state.firms.production_capacity * (1.0 - severity)
    new_firms = state.firms._replace(production_capacity=new_capacity)
    return state._replace(firms=new_firms)

def apply_freight_cost_shock(state: SimState, multiplier: float = 2.0) -> SimState:
    """Scales input_cost_multiplier on all firms."""
    new_input_cost = state.firms.input_cost_multiplier * multiplier
    new_firms = state.firms._replace(input_cost_multiplier=new_input_cost)
    return state._replace(firms=new_firms)

def apply_labor_shortage_shock(state: SimState, severity: float = 0.1) -> SimState:
    """Reduces available labor pool (by simulating early retirement/exit)."""
    # Reduce labor pool by marking a portion of agents as inactive/dead
    # Use the existing rng_key from state
    key, subkey = jax.random.split(state.rng_key)
    rand_vals = jax.random.uniform(subkey, state.agents.is_alive.shape)
    new_is_alive = jnp.where(rand_vals < severity, False, state.agents.is_alive)
    
    # Increase wage expectations to simulate a higher minimum wage floor
    new_wage = state.agents.wage * (1.0 + severity)
    
    new_agents = state.agents._replace(is_alive=new_is_alive, wage=new_wage)
    return state._replace(agents=new_agents, rng_key=key)
