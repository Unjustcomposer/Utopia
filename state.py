"""
JAX PyTree State Definitions
============================
Defines the unified, immutable state of the entire simulation.
By using NamedTuples, these automatically register as JAX PyTrees,
allowing them to be passed in and out of @jax.jit compiled functions.
"""
from typing import NamedTuple
import jax.numpy as jnp

class AgentState(NamedTuple):
    budget: jnp.ndarray           # (num_agents,)
    wage: jnp.ndarray             # (num_agents,)
    savings: jnp.ndarray          # (num_agents,)
    employed: jnp.ndarray         # (num_agents,) boolean
    employer_id: jnp.ndarray      # (num_agents,) int
    risk_aversion: jnp.ndarray    # (num_agents,)
    savings_rate: jnp.ndarray     # (num_agents,)
    preferences: jnp.ndarray      # (num_agents, num_goods)
    awareness: jnp.ndarray        # (num_agents, num_goods)
    past_avg_prices: jnp.ndarray  # (num_agents, num_goods)
    
    # Module 2 expansions
    is_alive: jnp.ndarray         # (num_agents,) boolean
    age: jnp.ndarray              # (num_agents,) int
    skill: jnp.ndarray            # (num_agents,) float
    
    # Module 3 expansions
    region_id: jnp.ndarray        # (num_agents,) int
    housing_status: jnp.ndarray   # (num_agents,) int
    housing_wealth: jnp.ndarray   # (num_agents,) float
    inflation_expectations: jnp.ndarray # (num_agents,) float
    neighbors: jnp.ndarray        # (num_agents, k) int (Social Network)
    insured: jnp.ndarray          # (num_agents,) boolean

class FirmState(NamedTuple):
    good_produced: jnp.ndarray    # (num_firms,) int
    cash: jnp.ndarray             # (num_firms,)
    inventory: jnp.ndarray        # (num_firms,)
    price: jnp.ndarray            # (num_firms,)
    quality: jnp.ndarray          # (num_firms,)
    production_capacity: jnp.ndarray # (num_firms,)
    num_employees: jnp.ndarray    # (num_firms,) int
    wage_offer: jnp.ndarray       # (num_firms,)
    debt: jnp.ndarray             # (num_firms,)
    cumulative_revenue: jnp.ndarray # (num_firms,)
    cumulative_cost: jnp.ndarray  # (num_firms,)
    
    # We use fixed-size ring buffers or shift registers for history
    demand_history: jnp.ndarray   # (num_firms, 3)
    revenue_history: jnp.ndarray  # (num_firms, 3)
    price_history: jnp.ndarray    # (num_firms, 3)
    profit_history: jnp.ndarray   # (num_firms, 3)
    
    input_cost_multiplier: jnp.ndarray # (num_firms,)
    
    # Module 1 & 3 expansions
    is_active: jnp.ndarray        # (num_firms,) boolean
    capital_goods: jnp.ndarray    # (num_firms,) float
    equity: jnp.ndarray           # (num_firms,) float (SFC accounting)
    region_id: jnp.ndarray        # (num_firms,) int
    menu_cost_paid: jnp.ndarray   # (num_firms,) boolean

class MacroState(NamedTuple):
    deposits: jnp.ndarray         # scalar
    loans: jnp.ndarray            # scalar
    base_rate: jnp.ndarray        # scalar
    price_index: jnp.ndarray      # scalar
    memory_count: jnp.ndarray     # scalar int
    bank_equity: jnp.ndarray      # scalar (SFC accounting)
    sfc_delta: jnp.ndarray        # scalar (Tracks leak)

class GovState(NamedTuple):
    tax_revenue: jnp.ndarray      # scalar
    transfers_paid: jnp.ndarray   # scalar
    bonds_outstanding: jnp.ndarray # scalar
    cash: jnp.ndarray             # scalar

class HousingState(NamedTuple):
    supply: jnp.ndarray           # (num_regions,)
    price: jnp.ndarray            # (num_regions,)

class ForeignState(NamedTuple):
    exchange_rate: jnp.ndarray    # scalar
    exports: jnp.ndarray          # scalar
    imports: jnp.ndarray          # scalar

class SimState(NamedTuple):
    agents: AgentState
    firms: FirmState
    macro: MacroState
    gov: GovState
    housing: HousingState
    foreign: ForeignState
    rng_key: jnp.ndarray          # PRNG key for jax.random
    lmm_params: dict              # Flax PyTree of LMM weights
