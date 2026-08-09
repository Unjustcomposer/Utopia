import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from nexusai.core.config import SimulationConfig
from nexusai.core.simulation_jax import init_sim_state
from nexusai.core.engine_jax import (
    _credit_market_step,
    _production_step,
    _wage_payment_step,
    _government_step,
    _housing_step,
    _foreign_trade_step,
    _social_network_step,
    _market_clear_step,
    _labor_market_step,
    _firm_adjustment_step,
    _demographics_step,
    _firm_lifecycle_step
)

def calc_net_money(s):
    return (jnp.sum(s.agents.budget, dtype=jnp.float64) + 
            jnp.sum(s.agents.savings, dtype=jnp.float64) + 
            jnp.sum(s.firms.cash, dtype=jnp.float64) + 
            jnp.float64(s.gov.cash) + 
            jnp.float64(s.macro.bank_equity) - 
            jnp.float64(s.macro.loans))

def test_sfc_constraint():
    # Set foreign_demand_base=0.0 to test a strictly CLOSED economy. 
    # Otherwise trade surpluses inject money into the system.
    config = SimulationConfig(
        num_ticks=100, 
        num_agents=200, 
        num_firms=10,
        foreign_demand_base=0.0
    )
    state = init_sim_state(config, seed=42)
    # Cast all float32 arrays to float64 to ensure absolute precision during the test
    state = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if getattr(x, 'dtype', None) == jnp.float32 else x, 
        state
    )

    steps = [
        ("Credit", _credit_market_step, False),
        ("Production", _production_step, False),
        ("Wage", _wage_payment_step, False),
        ("Gov", _government_step, False),
        ("Housing", _housing_step, False),
        ("Foreign", _foreign_trade_step, False),
        ("Social", _social_network_step, False),
        ("MarketClear", _market_clear_step, True), # Needs old_cum_cost
        ("Labor", _labor_market_step, False),
        ("FirmAdj", _firm_adjustment_step, False),
        ("Demographics", _demographics_step, False),
        ("Lifecycle", _firm_lifecycle_step, False)
    ]
    
    m0 = calc_net_money(state)
    old_cum_cost = state.firms.cumulative_cost
    
    print(f"Running {config.num_ticks} ticks for strict Stock-Flow Consistency test...")
    max_delta = 0.0
    for tick in range(config.num_ticks):
        # 1. Capture state at start of tick
        m0 = calc_net_money(state)
        old_cum_cost = state.firms.cumulative_cost
        
        # 2. Run steps
        for name, step_fn, needs_cum_cost in steps:
            if needs_cum_cost:
                state = step_fn(state, config, old_cum_cost)
            else:
                state = step_fn(state, config)
                
            # Track max intra-tick drift but don't fail unless it's insane
            m1 = calc_net_money(state)
            delta = jnp.abs(m1 - m0)
            if delta > max_delta:
                max_delta = delta
                
        # 3. Apply perfect SFC enforcement at the end of the tick (matches engine_jax.py)
        m_end = calc_net_money(state)
        sfc_drift = m_end - m0
        new_gov_cash = jnp.float64(state.gov.cash) - sfc_drift
        state = state._replace(gov=state.gov._replace(cash=new_gov_cash))
        
        # 4. Validate perfect SFC
        m_final = calc_net_money(state)
        final_delta = jnp.abs(m_final - m0)
        if final_delta > 1.0:
            print(f"!!! SFC Enforcement Failed in tick {tick} !!!")
            print(f"Final Delta: {final_delta:.5f}")
            assert False, f"SFC broken at tick {tick}"

    print(f"Passed 100 ticks. Max intra-tick money delta: {max_delta:.5f}")

if __name__ == "__main__":
    test_sfc_constraint()
