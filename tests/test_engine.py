import jax
import jax.numpy as jnp
import pytest

from config import SimulationConfig
from simulation_jax import init_sim_state
from engine_jax import simulation_step

def test_sfc_balance_closed_economy():
    """Verify that money is neither created nor destroyed in a closed economy."""
    config = SimulationConfig(
        num_agents=100,
        num_firms=10,
        num_goods=2,
        num_ticks=2,
        firm_behavior_mode=2,  # Heuristic mode
        foreign_demand_base=0.0 # Force a closed economy for exact SFC testing
    )
    
    # Initialize state
    state = init_sim_state(config, seed=42)
    
    # Calculate initial money supply
    def calc_net_money(s):
        total_agent_money = jnp.sum(s.agents.budget) + jnp.sum(s.agents.savings)
        total_firm_money = jnp.sum(s.firms.cash)
        total_gov_money = s.gov.cash
        total_bank_equity = s.macro.bank_equity
        total_loans = s.macro.loans
        return total_agent_money + total_firm_money + total_gov_money + total_bank_equity - total_loans
        
    initial_money = calc_net_money(state)
    
    # Run a tick
    new_state = simulation_step(state, config)
    
    final_money = calc_net_money(new_state)
    
    delta = final_money - initial_money
    assert jnp.abs(delta) < 1.0, f"SFC Leak Detected! Initial: {initial_money}, Final: {final_money}, Delta: {delta}"
    
def test_engine_no_nans():
    """Ensure the engine does not produce NaNs after a full run."""
    config = SimulationConfig(
        num_agents=50,
        num_firms=5,
        num_goods=2,
        num_ticks=5,
        firm_behavior_mode=2
    )
    
    state = init_sim_state(config, seed=42)
    
    # Run 5 ticks
    for _ in range(5):
        state = simulation_step(state, config)
        
    assert not jnp.any(jnp.isnan(state.agents.budget))
    assert not jnp.any(jnp.isnan(state.firms.cash))
    assert not jnp.any(jnp.isnan(state.macro.price_index))
