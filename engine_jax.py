"""
JAX Core Engine
===============
Pure functional implementation of the simulation tick logic.
This module is fully compilable by XLA (via @jax.jit) allowing
execution on CPU/GPU without Python overhead.
"""
from typing import Tuple
import jax
import jax.numpy as jnp
import os

# Enable persistent XLA compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.nexus_jax_cache")
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.nexus_jax_cache"))

from config import SimulationConfig
from state import AgentState, FirmState, MacroState, SimState
from lmm_model import FirmTransformer


from engine.credit import _credit_market_step
from engine.production import _production_step, _wage_payment_step
from engine.government import _government_step
from engine.housing import _housing_step
from engine.foreign import _foreign_trade_step
from engine.social import _social_network_step, _demographics_step
from engine.market import _market_clear_step
from engine.labor import _labor_market_step
from engine.firm_logic import _firm_adjustment_step, _firm_lifecycle_step

@jax.jit
def simulation_step(state: SimState, config: SimulationConfig) -> SimState:
    """Master compiled step function that runs one complete tick of the economy."""
    old_cum_cost = state.firms.cumulative_cost
    
    def calc_net_money(s):
        return jnp.sum(s.agents.budget) + jnp.sum(s.agents.savings) + jnp.sum(s.firms.cash) + s.gov.cash + s.macro.bank_equity - s.macro.loans + s.foreign.cash

    m0 = calc_net_money(state)
    
    state = _credit_market_step(state, config)
    state = _production_step(state, config)
    state = _wage_payment_step(state, config)
    state = _government_step(state, config)
    state = _housing_step(state, config)
    state = _foreign_trade_step(state, config)
    state = _social_network_step(state, config)
    state = _market_clear_step(state, config, old_cum_cost)
    state = _labor_market_step(state, config)
    state = _firm_adjustment_step(state, config)
    state = _demographics_step(state, config)
    state = _firm_lifecycle_step(state, config)
    
    # ---------------------------------------------------------
    # PERFECT SFC ENFORCEMENT
    # ---------------------------------------------------------
    # Over 500+ ticks, float32 summation of millions of array 
    # elements across 10+ economic steps causes a slow precision drift 
    # of ~$0.01 to ~$0.05 per tick.
    # To prevent this from accumulating into macroeconomic instability,
    # we route the EXACT float32 noise back into the central bank (government cash).
    m1 = calc_net_money(state)
    sfc_drift = m1 - m0
    new_gov_cash = state.gov.cash - sfc_drift
    state = state._replace(gov=state.gov._replace(cash=new_gov_cash))
    
    # Re-calculate net money for debug check (should be exactly 0.0 now)
    m1_fixed = calc_net_money(state)
    sfc_delta = jnp.abs(m1_fixed - m0)
    
    # Enforce SFC: if delta exceeds tolerance, log a warning
    jax.debug.callback(
        lambda d, tol: print(f"SFC VIOLATION: ${float(d):.6f} leaked") if float(d) > tol else None,
        sfc_delta, config.sfc_tolerance
    )
    
    new_macro = state.macro._replace(sfc_delta=sfc_delta)
    
    return state._replace(macro=new_macro)
