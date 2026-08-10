import jax
import jax.numpy as jnp
import pytest

from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state
from utopia.core.engine_jax import simulation_step

def test_sfc_balance_closed_economy():
    """Verify that money is neither created nor destroyed in a closed economy."""
    config = SimulationConfig(
        num_agents=100,
        num_firms=10,
        num_goods=2,
        num_ticks=500,
        firm_behavior_mode=2,  # Heuristic mode
        foreign_demand_base=0.0, # Force a closed economy for exact SFC testing
        bom_matrix=jnp.zeros((2, 2))
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
    
    # Run 500 ticks
    for _ in range(500):
        state = simulation_step(state, config)
    
    final_money = calc_net_money(state)
    
    delta = final_money - initial_money
    assert jnp.abs(delta) < 10.0, f"SFC Leak Detected! Initial: {initial_money}, Final: {final_money}, Delta: {delta}"
    
def test_engine_no_nans():
    """Ensure the engine does not produce NaNs after a full run."""
    config = SimulationConfig(
        num_agents=50,
        num_firms=5,
        num_goods=2,
        num_ticks=5,
        firm_behavior_mode=2,
        bom_matrix=jnp.zeros((2, 2))
    )
    
    state = init_sim_state(config, seed=42)
    
    # Run 5 ticks
    for _ in range(5):
        state = simulation_step(state, config)
        
    assert not jnp.any(jnp.isnan(state.agents.budget))
    assert not jnp.any(jnp.isnan(state.firms.cash))
    assert not jnp.any(jnp.isnan(state.macro.price_index))

def test_logistics_bottleneck():
    """Verify that port congestion correctly increases delivery delays."""
    config = SimulationConfig(
        num_agents=10,
        num_firms=2,
        num_goods=2,
        num_ticks=1,
        firm_behavior_mode=2,
        max_transit_delay=10,
        base_port_capacity=10.0,
        bom_matrix=jnp.zeros((2, 2))
    )
    
    state = init_sim_state(config, seed=42)
    
    # Inject massive backlog into in_transit_inventory
    # Firms have 100 goods in transit at index 1 (not arriving next tick)
    massive_transit = jnp.zeros((2, 2, 10), dtype=jnp.float32)
    massive_transit = massive_transit.at[:, :, 1].set(100.0)
    
    state = state._replace(
        firms=state.firms._replace(in_transit_inventory=massive_transit),
        logistics=state.logistics._replace(port_capacity=jnp.full((config.num_regions,), 10.0))
    )
    
    # Run 1 tick
    next_state = simulation_step(state, config)
    
    assert jnp.any(next_state.logistics.port_queue > 0.0)
    assert jnp.any(next_state.logistics.dwell_time > 1)
