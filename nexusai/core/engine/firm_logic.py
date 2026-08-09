import jax
import jax.numpy as jnp
from nexusai.core.config import SimulationConfig
from nexusai.core.state import SimState
from nexusai.core.lmm_model import FirmTransformer

def _firm_adjustment_step(state: SimState, config: SimulationConfig) -> SimState:
    """LMM-driven Firm Strategy Adjustment, with ZI and Heuristic Baselines."""
    firms, macro, key = state.firms, state.macro, state.rng_key
    
    # ── Mode 0: LMM ──
    macro_price = jnp.full_like(firms.demand_history, macro.price_index)
    macro_rate = jnp.full_like(firms.demand_history, macro.base_rate)
    
    # Roll the ring buffers into chronological order [t-2, t-1, t]
    shift_amt = -(macro.memory_count % 3 + 1)
    roll = lambda x: jnp.roll(x, shift=shift_amt, axis=1)
    
    lmm_inputs = jnp.stack([
        roll(firms.demand_history),
        roll(firms.profit_history),
        roll(firms.price_history),
        macro_price,
        macro_rate
    ], axis=-1)
    
    model = FirmTransformer()
    dp_lmm, dw_lmm, prod_lmm = model.apply({'params': state.lmm_params}, lmm_inputs)
    
    # ── Mode 1: Zero-Intelligence (Random) ──
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    dp_zi = jax.random.uniform(sk1, dp_lmm.shape, minval=-0.05, maxval=0.05)
    dw_zi = jax.random.uniform(sk2, dw_lmm.shape, minval=-0.02, maxval=0.02)
    prod_zi = firms.production_capacity * jax.random.uniform(sk3, prod_lmm.shape, minval=0.9, maxval=1.1)
    
    # ── Mode 2: Heuristic ──
    # Increase price if demand > production, decrease otherwise
    # The most recent value is at (macro.memory_count % 3)
    idx = macro.memory_count % 3
    recent_demand = firms.demand_history[:, idx]
    recent_profit = firms.profit_history[:, idx]
    
    demand_ratio = recent_demand / jnp.maximum(1.0, firms.production_capacity)
    dp_heur = jnp.where(demand_ratio > 1.05, 0.02, jnp.where(demand_ratio < 0.95, -0.02, 0.0))
    
    # Increase wages if profitable and capacity constrained, else freeze/cut
    dw_heur = jnp.where((recent_profit > 0) & (demand_ratio > 1.0), 0.01, jnp.where(recent_profit < 0, -0.01, 0.0))
    
    # Target production based on recent demand
    prod_heur = recent_demand * 1.1
    
    # ── Selection ──
    mode = config.firm_behavior_mode
    delta_price = jnp.where(mode == 0, dp_lmm, jnp.where(mode == 1, dp_zi, dp_heur))
    delta_wage = jnp.where(mode == 0, dw_lmm, jnp.where(mode == 1, dw_zi, dw_heur))
    target_production = jnp.where(mode == 0, prod_lmm, jnp.where(mode == 1, prod_zi, prod_heur))
    
    # Apply deltas
    new_price = firms.price * (1.0 + delta_price)
    min_price = config.input_cost_base * firms.input_cost_multiplier * 1.1
    final_price = jnp.maximum(new_price, min_price)
    
    new_wage_offer = firms.wage_offer * (1.0 + delta_wage)
    new_wage_offer = jnp.maximum(new_wage_offer, config.minimum_wage)
    
    new_production_capacity = target_production
    
    new_capital_goods = firms.capital_goods * (1.0 - config.capital_depreciation)
    
    new_firms = firms._replace(
        price=final_price,
        wage_offer=new_wage_offer,
        production_capacity=new_production_capacity,
        capital_goods=new_capital_goods
    )
    return state._replace(firms=new_firms, rng_key=key)


def _firm_lifecycle_step(state: SimState, config: SimulationConfig) -> SimState:
    """Firms enter and exit (bankruptcy)."""
    agents, firms, macro, key = state.agents, state.firms, state.macro, state.rng_key
    
    equity = firms.cash + firms.inventory * firms.price + firms.capital_goods * config.capital_cost - firms.debt
    
    # Differentiable bankruptcy gating
    temperature = 10.0
    survival_prob = jax.nn.sigmoid((equity - config.bankruptcy_threshold) * temperature)
    
    # Maintain continuous activity mask (1.0 = fully active, 0.0 = fully dead)
    new_is_active = firms.is_active * survival_prob
    bankrupt_mask = firms.is_active * (1.0 - survival_prob)
    
    bad_debt = bankrupt_mask * jnp.maximum(0.0, firms.debt - firms.cash)
    new_macro_loans = macro.loans - jnp.sum(bankrupt_mask * firms.debt)
    new_bank_equity = macro.bank_equity - jnp.sum(bad_debt)
    
    # Scale down states continuously
    new_cash = firms.cash * survival_prob
    new_debt = firms.debt * survival_prob
    
    # Float leak capture!
    total_firm_cash_loss = jnp.sum(firms.cash) - jnp.sum(new_cash)
    total_loans_loss = macro.loans - new_macro_loans
    total_equity_loss = macro.bank_equity - new_bank_equity
    
    # Gov absorbs all leftover cash and float mismatches to ensure PERFECT SFC balance
    new_gov_cash = state.gov.cash + total_firm_cash_loss - total_loans_loss + total_equity_loss
    new_inventory = firms.inventory * survival_prob
    new_employees = firms.num_employees * survival_prob
    new_capital = firms.capital_goods * survival_prob
    
    # Entry
    key, subkey = jax.random.split(key)
    # Using continuous logic: if inactive, probability of entry increases mask
    entry_prob = jax.random.uniform(subkey, firms.is_active.shape)
    enters_mask = jax.nn.sigmoid((config.firm_entry_probability - entry_prob) * 100.0) * (1.0 - new_is_active)
    
    new_is_active = new_is_active + enters_mask
    new_cash = new_cash + enters_mask * config.initial_firm_cash_min
    new_capital = new_capital + enters_mask * 10.0
    new_price = firms.price * (1.0 - enters_mask) + enters_mask * ((config.base_price_min + config.base_price_max)/2)
    new_debt = new_debt + enters_mask * config.initial_firm_cash_min
    new_macro_loans = new_macro_loans + jnp.sum(enters_mask * config.initial_firm_cash_min)
    
    # Layoff employees of bankrupt firms
    def update_agent(emp, emp_id):
        # Soft layoff: reduce effective employment probability based on firm bankruptcy
        firm_bankrupt_weight = jnp.where(emp_id >= 0, bankrupt_mask[jnp.maximum(0, emp_id)], 0.0)
        layoff_prob = firm_bankrupt_weight
        new_emp = emp * (1.0 - layoff_prob)
        return new_emp, emp_id
        
    new_employed, new_employer_id = jax.vmap(update_agent)(agents.employed.astype(jnp.float32), agents.employer_id)
    new_agents = agents._replace(employed=new_employed, employer_id=new_employer_id)
    
    new_firms = firms._replace(
        equity=equity,
        is_active=new_is_active,
        cash=new_cash,
        debt=new_debt,
        inventory=new_inventory,
        num_employees=new_employees,
        capital_goods=new_capital,
        price=new_price
    )
    
    new_macro = macro._replace(
        loans=new_macro_loans,
        bank_equity=new_bank_equity
    )
    
    return state._replace(
        agents=new_agents, 
        firms=new_firms, 
        macro=new_macro, 
        rng_key=key,
        gov=state.gov._replace(cash=new_gov_cash)
    )

