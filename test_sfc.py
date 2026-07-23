import jax
import jax.numpy as jnp
from config import SimulationConfig
from simulation_jax import init_sim_state
from engine_jax import (
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
    return jnp.sum(s.agents.budget) + jnp.sum(s.agents.savings) + jnp.sum(s.firms.cash) + s.gov.cash + s.macro.bank_equity - s.macro.loans

def test_sfc_constraint():
    # Set foreign_demand_base=0.0 to test a strictly CLOSED economy. 
    # Otherwise trade surpluses inject money into the system.
    config = SimulationConfig(
        num_ticks=1, 
        num_agents=200, 
        num_firms=20,
        foreign_demand_base=0.0
    )
    state = init_sim_state(config, seed=42)
    
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
    
    for name, step_fn, needs_cum_cost in steps:
        if needs_cum_cost:
            state = step_fn(state, config, old_cum_cost)
        else:
            state = step_fn(state, config)
            
        m1 = calc_net_money(state)
        delta = jnp.abs(m1 - m0)
        print(f"{name:<15} Delta: {delta:.5f} | M: {m1:.5f}")
        if delta > 1.0: # Float32 accumulation tolerance
            print(f"!!! SFC Violation in {name} !!!")
            break
        m0 = m1

if __name__ == "__main__":
    test_sfc_constraint()
