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

from config import SimulationConfig
from state import AgentState, FirmState, MacroState, SimState
from lmm_model import FirmTransformer

def _shift_history(hist: jnp.ndarray, new_val: jnp.ndarray) -> jnp.ndarray:
    """Shifts history array of shape (N, 3) left and appends new_val to the end."""
    shifted = jnp.roll(hist, shift=-1, axis=1)
    return shifted.at[:, -1].set(new_val)

def ste_boolean_mask(logits: jnp.ndarray) -> jnp.ndarray:
    """Straight-Through Estimator (STE) for boolean masks (x > 0)."""
    hard = (logits > 0).astype(jnp.float32)
    soft = jax.nn.sigmoid(logits)
    return jax.lax.stop_gradient(hard - soft) + soft

def _credit_market_step(state: SimState, config: SimulationConfig) -> SimState:
    """Endogenous credit and deposit step."""
    agents, firms, macro = state.agents, state.firms, state.macro
    
    deposit_rate = jnp.maximum(0.001, macro.base_rate - config.commercial_bank_spread / 2)
    lending_rate = macro.base_rate + config.commercial_bank_spread / 2
    
    # Agents earn interest on savings
    interest_earned = agents.savings * deposit_rate
    new_savings = agents.savings + interest_earned
    new_deposits = jnp.sum(new_savings)
    
    # Firms pay interest and request loans
    # Pay interest
    interest_payment = firms.debt * lending_rate
    firm_cash_after_interest = firms.cash - interest_payment
    
    # Try to pay down principal
    paydown = jnp.where(
        firm_cash_after_interest > firms.debt,
        firms.debt,
        jnp.where(firm_cash_after_interest > 0, firm_cash_after_interest * 0.5, 0.0)
    )
    firm_cash_after_paydown = firm_cash_after_interest - paydown
    firm_debt_after_paydown = firms.debt - paydown
    
    # Request loans
    expected_labor_cost = firms.num_employees * firms.wage_offer
    expected_input_cost = firms.production_capacity * config.input_cost_base * firms.input_cost_multiplier
    required_cash = expected_labor_cost + expected_input_cost
    
    shortfall = jnp.maximum(0.0, required_cash - firm_cash_after_paydown)
    max_new_lending = (new_deposits / config.reserve_requirement) - macro.loans
    
    # Distribute available lending proportionally or capped
    total_shortfall = jnp.sum(shortfall)
    # Avoid division by zero and NaN gradients
    safe_shortfall = jnp.where(total_shortfall > 0, total_shortfall, 1.0)
    ratio = jnp.where(total_shortfall > 0, jnp.minimum(1.0, max_new_lending / safe_shortfall), 0.0)
    ratio = jnp.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = jnp.maximum(0.0, ratio)
    
    granted_loans = shortfall * ratio
    
    new_cash = firm_cash_after_paydown + granted_loans
    new_debt = firm_debt_after_paydown + granted_loans
    new_macro_loans = macro.loans - jnp.sum(paydown) + jnp.sum(granted_loans)
    
    new_bank_equity = macro.bank_equity + jnp.sum(interest_payment) - jnp.sum(interest_earned)
    
    # Update Macro Base Rate (Taylor rule based on previous tick inflation)
    # We will do this at the end of the tick when we have new prices
    
    new_agents = agents._replace(savings=new_savings)
    new_firms = firms._replace(cash=new_cash, debt=new_debt)
    new_macro = macro._replace(deposits=new_deposits, loans=new_macro_loans, bank_equity=new_bank_equity)
    
    return state._replace(agents=new_agents, firms=new_firms, macro=new_macro)


def _production_step(state: SimState, config: SimulationConfig) -> SimState:
    """Firms produce goods."""
    agents, firms = state.agents, state.firms
    
    def total_firm_skill(firm_id):
        return jnp.sum(jnp.where(agents.employer_id == firm_id, agents.skill, 0.0))
        
    firm_ids = jnp.arange(firms.cash.shape[0])
    effective_labor = jax.vmap(total_firm_skill)(firm_ids)
    
    labor_output = effective_labor * config.productivity_per_worker
    
    # Capital determines physical capacity limit
    capacity = firms.capital_goods * 10.0
    raw_output = jnp.minimum(capacity, labor_output)
    
    # Only active firms produce
    raw_output = jnp.where(firms.is_active, raw_output, 0.0)
    
    effective_output = raw_output / jnp.maximum(firms.input_cost_multiplier, 0.01)
    new_inventory = firms.inventory + effective_output
    
    input_cost = effective_output * config.input_cost_base * firms.input_cost_multiplier
    new_cash = firms.cash - input_cost
    new_cumulative_cost = firms.cumulative_cost + input_cost
    
    new_gov_cash = state.gov.cash + jnp.sum(input_cost) # PREVENT MONEY LEAK
    
    new_firms = firms._replace(
        inventory=new_inventory,
        cash=new_cash,
        cumulative_cost=new_cumulative_cost,
        production_capacity=capacity
    )
    return state._replace(firms=new_firms, gov=state.gov._replace(cash=new_gov_cash))


def _wage_payment_step(state: SimState, config: SimulationConfig) -> SimState:
    """Firms pay wages to their employees."""
    agents, firms = state.agents, state.firms
    
    # Instead of iterating over firms, we vectorized agent wage receipt
    # Since agent employer_id maps to firm indices, we can gather wage_offer
    # But wait, agents already have `agent.wage` assigned during hiring!
    # So we just add agent.wage to their budget.
    earned_wages = jnp.where(agents.employed & agents.is_alive, agents.wage, 0.0)
    
    # Progressive Taxation
    tax = jnp.where(earned_wages > config.income_tax_bracket_threshold,
                    (earned_wages - config.income_tax_bracket_threshold) * config.income_tax_rate_top + config.income_tax_bracket_threshold * config.income_tax_rate_base,
                    earned_wages * config.income_tax_rate_base)
    tax = jnp.where(agents.employed & agents.is_alive, tax, 0.0)
    
    net_wages = earned_wages - tax
    new_budget = agents.budget + net_wages
    
    # Track taxes
    tick_tax_revenue = jnp.sum(tax)
    
    # Subtract from firm cash
    # We need to sum wages per firm.
    def sum_wages(firm_id):
        return jnp.sum(jnp.where(agents.employer_id == firm_id, agents.wage, 0.0))
        
    firm_ids = jnp.arange(firms.cash.shape[0])
    total_wages_per_firm = jax.vmap(sum_wages)(firm_ids)
    
    new_cash = firms.cash - total_wages_per_firm
    new_cumulative_cost = firms.cumulative_cost + total_wages_per_firm
    
    new_agents = agents._replace(budget=new_budget)
    new_firms = firms._replace(cash=new_cash, cumulative_cost=new_cumulative_cost)
    new_gov = state.gov._replace(tax_revenue=state.gov.tax_revenue + tick_tax_revenue, cash=state.gov.cash + tick_tax_revenue)
    
    return state._replace(agents=new_agents, firms=new_firms, gov=new_gov)

def _government_step(state: SimState, config: SimulationConfig) -> SimState:
    """Government collects corporate taxes and pays unemployment benefits."""
    agents, firms, gov = state.agents, state.firms, state.gov
    
    # Corporate Tax
    firm_profit = firms.cash - firms.debt - firms.cumulative_cost # Simplified proxy for taxable income
    corp_tax = jnp.where(firm_profit > 0, firm_profit * config.corporate_tax_rate, 0.0)
    corp_tax = jnp.where(firms.is_active, corp_tax, 0.0)
    
    new_firm_cash = firms.cash - corp_tax
    total_corp_tax = jnp.sum(corp_tax)
    
    # Unemployment Benefits
    unemployed_mask = ~agents.employed & agents.is_alive
    total_unemployed = jnp.sum(unemployed_mask)
    benefits_paid = unemployed_mask * config.unemployment_benefit
    
    new_agent_budget = agents.budget + benefits_paid
    total_benefits = jnp.sum(benefits_paid)
    
    new_gov_cash = gov.cash + total_corp_tax - total_benefits
    new_transfers_paid = gov.transfers_paid + total_benefits
    
    new_agents = agents._replace(budget=new_agent_budget)
    new_firms = firms._replace(cash=new_firm_cash)
    new_gov = gov._replace(cash=new_gov_cash, transfers_paid=new_transfers_paid)
    
    return state._replace(agents=new_agents, firms=new_firms, gov=new_gov)

def _housing_step(state: SimState, config: SimulationConfig) -> SimState:
    """Agents pay housing maintenance or rent, prices update based on demand."""
    agents, housing = state.agents, state.housing
    
    def get_regional_demand(r):
        mask = (agents.region_id == r) & agents.is_alive
        return jnp.sum(agents.budget * mask) / jnp.maximum(1, jnp.sum(mask))
        
    avg_regional_wealth = jax.vmap(get_regional_demand)(jnp.arange(config.num_regions))
    new_housing_price = config.housing_base_price + avg_regional_wealth * 0.1
    
    rent = new_housing_price[agents.region_id] * 0.05
    is_renter = (agents.housing_status == 0) & agents.is_alive
    
    maintenance = new_housing_price[agents.region_id] * config.housing_depreciation
    is_owner = (agents.housing_status == 1) & agents.is_alive
    
    housing_cost = jnp.where(is_renter, rent, jnp.where(is_owner, maintenance, 0.0))
    new_budget = agents.budget - housing_cost
    
    can_buy = (new_budget > new_housing_price[agents.region_id] * 0.2) & is_renter
    new_housing_status = jnp.where(can_buy, 1, agents.housing_status)
    down_payments = jnp.where(can_buy, new_housing_price[agents.region_id] * 0.2, 0.0)
    new_budget = new_budget - down_payments
    
    total_leaked_housing = jnp.sum(housing_cost) + jnp.sum(down_payments)
    new_gov_cash = state.gov.cash + total_leaked_housing # PREVENT MONEY LEAK
    
    new_agents = agents._replace(budget=new_budget, housing_status=new_housing_status, housing_wealth=jnp.where(new_housing_status == 1, new_housing_price[agents.region_id], 0.0))
    new_housing = housing._replace(price=new_housing_price)
    
    return state._replace(agents=new_agents, housing=new_housing, gov=state.gov._replace(cash=new_gov_cash))

def _foreign_trade_step(state: SimState, config: SimulationConfig) -> SimState:
    foreign, key = state.foreign, state.rng_key
    
    key, subkey = jax.random.split(key)
    shock = jax.random.normal(subkey) * config.exchange_rate_volatility
    new_exchange_rate = foreign.exchange_rate * jnp.exp(shock)
    
    exports = config.foreign_demand_base * new_exchange_rate
    imports = config.foreign_demand_base / new_exchange_rate
    
    new_foreign = foreign._replace(exchange_rate=new_exchange_rate, exports=exports, imports=imports)
    return state._replace(foreign=new_foreign, rng_key=key)

def _social_network_step(state: SimState, config: SimulationConfig) -> SimState:
    agents = state.agents
    
    def get_neighbor_awareness(agent_i):
        neighbor_indices = agents.neighbors[agent_i]
        return jnp.mean(agents.awareness[neighbor_indices], axis=0)
    
    avg_neighbor_awareness = jax.vmap(get_neighbor_awareness)(jnp.arange(agents.budget.shape[0]))
    new_awareness = (1 - config.network_influence) * agents.awareness + config.network_influence * avg_neighbor_awareness
    
    inflation_exp = jnp.mean(agents.past_avg_prices, axis=1) * config.expectation_alpha
    new_savings_rate = agents.savings_rate - inflation_exp * 0.01
    new_savings_rate = jnp.clip(new_savings_rate, config.savings_rate_min, config.savings_rate_max)
    
    new_agents = agents._replace(awareness=new_awareness, inflation_expectations=inflation_exp, savings_rate=new_savings_rate)
    return state._replace(agents=new_agents)

def _market_clear_step(state: SimState, config: SimulationConfig, old_cum_cost: jnp.ndarray) -> SimState:
    """Agents compute demand and market clears."""
    agents, firms, macro, key = state.agents, state.firms, state.macro, state.rng_key
    
    # 1. Gather prices and supply per good
    num_goods = config.num_goods
    
    # Average price per good
    def get_avg_price(g):
        mask = firms.good_produced == g
        return jnp.sum(firms.price * mask) / jnp.maximum(1, jnp.sum(mask))
        
    prices = jax.vmap(get_avg_price)(jnp.arange(num_goods))
    
    def get_total_supply(g):
        mask = firms.good_produced == g
        return jnp.sum(jnp.maximum(firms.inventory, 0.0) * mask)
        
    supply = jax.vmap(get_total_supply)(jnp.arange(num_goods))
    
    # 2. Agent CES Demand Calculation
    save_amount = agents.budget * agents.savings_rate * (1.0 + agents.risk_aversion * 0.5)
    save_amount = jnp.minimum(save_amount, agents.budget)
    new_savings = agents.savings + save_amount
    spending_budget = agents.budget - save_amount
    
    price_mask = prices > 0
    avail_mask = supply > 0
    # Differentiable smooth gate for awareness instead of boolean threshold
    awareness_gate = jax.nn.sigmoid((agents.awareness - config.awareness_threshold) * 10.0)
    visible_prefs = agents.preferences * awareness_gate * price_mask * avail_mask
    
    sigma = config.ces_elasticity
    safe_prices = jnp.where(prices > 0, prices, 1.0)
    
    numerator = visible_prefs * jnp.power(safe_prices, -sigma)
    denom_terms = visible_prefs * jnp.power(safe_prices, 1 - sigma)
    denominator = denom_terms.sum(axis=1, keepdims=True)
    
    valid_agents = (denominator > 0).flatten() & (spending_budget > 0)
    
    # quantities shape: (num_agents, num_goods)
    spend_per_good = jnp.where(
        valid_agents[:, None],
        (numerator / jnp.where(denominator > 0, denominator, 1.0)) * spending_budget[:, None],
        0.0
    )
    quantities = spend_per_good / safe_prices
    
    # Smooth foreign trade export demand
    export_demand_per_good = state.foreign.exports / num_goods
    total_demand_per_good = quantities.sum(axis=0) + export_demand_per_good
    
    # 3. Market Clearing (Proportional Rationing to avoid M log M sort)
    # Ratio of supply to demand per good
    safe_demand = jnp.where(total_demand_per_good > 0, total_demand_per_good, 1.0)
    ratio = jnp.where(
        total_demand_per_good > 0,
        jnp.minimum(1.0, supply / safe_demand),
        0.0
    )
    ratio = jnp.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Actual quantities agents get
    actual_purchased = quantities * ratio
    cost_per_agent = jnp.sum(actual_purchased * safe_prices, axis=1)
    
    # Geography transport cost penalty
    transport_cost = jnp.sum(actual_purchased, axis=1) * 0.1
    new_budget = agents.budget - save_amount - cost_per_agent - transport_cost
    
    # Send export revenue from foreign sector to firms and transport costs to government
    new_foreign_cash = state.foreign.cash - jnp.sum(export_demand_per_good * ratio * safe_prices) if hasattr(state.foreign, "cash") else 0.0 # Just a note, not strictly needed if foreign cash isn't tracked in total money.
    new_gov_cash = state.gov.cash + jnp.sum(transport_cost)
    
    # Allocate firm sales proportionally to their inventory share of the good
    def calc_firm_sales(i):
        g = firms.good_produced[i]
        firm_inv = jnp.maximum(firms.inventory[i], 0.0)
        good_supply = supply[g]
        safe_supply = jnp.where(good_supply > 0, good_supply, 1.0)
        share = jnp.where(good_supply > 0, firm_inv / safe_supply, 0.0)
        
        firm_volume = total_demand_per_good[g] * ratio[g] * share
        # FIX SFC: Firms receive the actual market clearing price the agents paid
        firm_revenue = firm_volume * safe_prices[g]
        
        # We also need to return demand observed by firm (proportion of total demand)
        firm_demand_obs = total_demand_per_good[g] * share
        return firm_volume, firm_revenue, firm_demand_obs
        
    firm_indices = jnp.arange(firms.cash.shape[0])
    firm_volumes, firm_revenues, firm_demands = jax.vmap(calc_firm_sales)(firm_indices)
    
    new_inventory = firms.inventory - firm_volumes
    new_cash = firms.cash + firm_revenues
    new_cum_revenue = firms.cumulative_revenue + firm_revenues
    
    # Update histories
    new_demand_history = _shift_history(firms.demand_history, firm_demands)
    new_revenue_history = _shift_history(firms.revenue_history, new_cum_revenue)
    new_price_history = _shift_history(firms.price_history, firms.price)
    
    tick_revenue = new_revenue_history[:, -1] - new_revenue_history[:, -2]
    tick_cost = firms.cumulative_cost - old_cum_cost
    tick_profit = tick_revenue - tick_cost
    new_profit_history = _shift_history(firms.profit_history, tick_profit) 
    
    new_firms = firms._replace(
        inventory=new_inventory,
        cash=new_cash,
        cumulative_revenue=new_cum_revenue,
        demand_history=new_demand_history,
        revenue_history=new_revenue_history,
        price_history=new_price_history,
        profit_history=new_profit_history
    )
    
    # Update agent memory
    mem_count = macro.memory_count
    alpha = 0.3
    new_past_prices = jnp.where(
        mem_count == 0,
        jnp.broadcast_to(prices, agents.past_avg_prices.shape),
        (1 - alpha) * agents.past_avg_prices + alpha * prices
    )
    
    new_agents = agents._replace(
        budget=new_budget,
        savings=new_savings,
        past_avg_prices=new_past_prices
    )
    
    # Update Macro Price Index (simplified average of prices)
    new_price_index = jnp.mean(prices)
    
    new_macro = macro._replace(
        memory_count=mem_count + 1,
        price_index=new_price_index
    )
    
    return state._replace(agents=new_agents, firms=new_firms, macro=new_macro, gov=state.gov._replace(cash=new_gov_cash))


def _labor_market_step(state: SimState, config: SimulationConfig) -> SimState:
    """DMP Labor Matching."""
    agents, firms, key = state.agents, state.firms, state.rng_key
    
    # Number of unemployed
    unemployed_mask = ~agents.employed
    U = jnp.sum(unemployed_mask)
    
    # Vacancies
    vacancies = jnp.maximum(0.0, (firms.production_capacity / config.productivity_per_worker) - firms.num_employees)
    V = jnp.sum(vacancies)
    
    # Matching Function
    mu = config.matching_efficiency
    alpha = config.matching_elasticity
    
    # Prevent NaN with safe power
    M = mu * jnp.power(jnp.maximum(1.0, U), alpha) * jnp.power(jnp.maximum(1.0, V), 1 - alpha)
    M = jnp.minimum(M, jnp.minimum(U, V))
    
    # In JAX, exact matching without loops is tricky. We'll use random allocation.
    # To keep it JIT-friendly, we'll give each unemployed agent a probability of being hired.
    safe_U = jnp.where(U > 0, U, 1.0)
    hire_prob = jnp.where(U > 0, M / safe_U, 0.0)
    key, subkey = jax.random.split(key)
    
    # Which agents get hired
    will_hire = jax.random.uniform(subkey, agents.employed.shape) < hire_prob
    newly_hired_mask = will_hire & unemployed_mask
    
    # Assign them to firms based on vacancy shares AND wage offers (differentiable attraction)
    key, subkey = jax.random.split(key)
    # Annealed temperature for Gumbel-Softmax like selection
    temp = jnp.maximum(0.1, 1.0 - (state.macro.memory_count * 0.01))
    
    # Use max-subtraction to prevent float32 overflow in jnp.exp
    scaled_wages = firms.wage_offer / temp
    max_scaled_wage = jnp.max(scaled_wages)
    
    attractiveness = jnp.maximum(0.0, vacancies) * jnp.exp(scaled_wages - max_scaled_wage)
    total_attr = jnp.sum(attractiveness)
    safe_total_attr = jnp.where(total_attr > 0, total_attr, 1.0)
    firm_probs = jnp.where(total_attr > 0, attractiveness / safe_total_attr, 0.0)
    firm_probs = jnp.nan_to_num(firm_probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # We sample a firm ID for ALL agents, but only apply it to newly_hired
    sampled_firm_ids = jax.random.choice(subkey, firms.cash.shape[0], shape=agents.employed.shape, p=firm_probs)
    
    new_employed = agents.employed | newly_hired_mask
    new_employer_id = jnp.where(newly_hired_mask, sampled_firm_ids, agents.employer_id)
    
    # Nash Bargaining (Vectorized across firms)
    firm_res_wages = firms.price * config.productivity_per_worker
    
    # For agents, just use base wage min + small premium
    agent_res_wage = config.minimum_wage
    beta = config.bargaining_power_agent
    negotiated_wages = beta * firm_res_wages + (1 - beta) * agent_res_wage
    negotiated_wages = jnp.maximum(negotiated_wages, config.minimum_wage)
    
    # Update firm wage_offer
    new_wage_offer = negotiated_wages
    
    # Set agent wages
    new_agent_wages = jnp.where(newly_hired_mask, new_wage_offer[sampled_firm_ids], agents.wage)
    
    # Update firm employee counts
    def count_employees(firm_id):
        return jnp.sum(new_employer_id == firm_id)
    
    new_num_employees = jax.vmap(count_employees)(jnp.arange(firms.cash.shape[0]))
    
    # Pay vacancy cost
    total_vacancy_cost = vacancies * config.vacancy_cost
    new_cash = firms.cash - total_vacancy_cost
    new_gov_cash = state.gov.cash + jnp.sum(total_vacancy_cost) # PREVENT MONEY LEAK
    
    new_agents = agents._replace(
        employed=new_employed,
        employer_id=new_employer_id,
        wage=new_agent_wages
    )
    new_firms = firms._replace(
        num_employees=new_num_employees,
        wage_offer=new_wage_offer,
        cash=new_cash
    )
    return state._replace(agents=new_agents, firms=new_firms, rng_key=key, gov=state.gov._replace(cash=new_gov_cash))


def _firm_adjustment_step(state: SimState, config: SimulationConfig) -> SimState:
    """LMM-driven Firm Strategy Adjustment, with ZI and Heuristic Baselines."""
    firms, macro, key = state.firms, state.macro, state.rng_key
    
    # ── Mode 0: LMM ──
    macro_price = jnp.full_like(firms.demand_history, macro.price_index)
    macro_rate = jnp.full_like(firms.demand_history, macro.base_rate)
    
    lmm_inputs = jnp.stack([
        firms.demand_history,
        firms.profit_history,
        firms.price_history,
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
    recent_demand = firms.demand_history[:, -1]
    recent_profit = firms.profit_history[:, -1]
    
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


def _demographics_step(state: SimState, config: SimulationConfig) -> SimState:
    """Agents age, die, and are replaced."""
    agents, key = state.agents, state.rng_key
    
    new_age = agents.age + 1
    
    key, subkey = jax.random.split(key)
    death_prob = config.agent_mortality_rate + (new_age > 100) * 0.05
    dies = jax.random.uniform(subkey, agents.is_alive.shape) < death_prob
    
    new_is_alive = jnp.ones_like(agents.is_alive)
    new_age = jnp.where(dies, 0, new_age)
    
    # SFC FIX: Estate goes to government, new agent gets government grant
    old_wealth = jnp.where(dies, agents.budget + agents.savings, 0.0)
    new_agent_wealth = jnp.where(dies, config.initial_budget_min, 0.0)
    wealth_delta = jnp.sum(old_wealth) - jnp.sum(new_agent_wealth)
    new_gov_cash = state.gov.cash + wealth_delta
    
    new_budget = jnp.where(dies, config.initial_budget_min, agents.budget)
    new_employed = jnp.where(dies, False, agents.employed)
    new_employer_id = jnp.where(dies, -1, agents.employer_id)
    new_savings = jnp.where(dies, 0.0, agents.savings)
    new_wage = jnp.where(dies, 0.0, agents.wage)
    
    new_agents = agents._replace(
        age=new_age,
        is_alive=new_is_alive,
        budget=new_budget,
        employed=new_employed,
        employer_id=new_employer_id,
        savings=new_savings,
        wage=new_wage
    )
    return state._replace(
        agents=new_agents, 
        rng_key=key, 
        gov=state.gov._replace(cash=new_gov_cash)
    )

def _firm_lifecycle_step(state: SimState, config: SimulationConfig) -> SimState:
    """Firms enter and exit (bankruptcy)."""
    agents, firms, macro, key = state.agents, state.firms, state.macro, state.rng_key
    
    equity = firms.cash + firms.inventory * firms.price + firms.capital_goods * config.capital_cost - firms.debt
    
    bankrupt = (equity < config.bankruptcy_threshold) & firms.is_active
    
    bad_debt = jnp.where(bankrupt, firms.debt - firms.cash, 0.0)
    bad_debt = jnp.maximum(0.0, bad_debt)
    new_macro_loans = macro.loans - jnp.sum(jnp.where(bankrupt, firms.debt, 0.0))
    new_bank_equity = macro.bank_equity - jnp.sum(bad_debt)
    
    new_is_active = jnp.where(bankrupt, False, firms.is_active)
    new_cash = jnp.where(bankrupt, 0.0, firms.cash)
    new_debt = jnp.where(bankrupt, 0.0, firms.debt)
    new_inventory = jnp.where(bankrupt, 0.0, firms.inventory)
    new_employees = jnp.where(bankrupt, 0, firms.num_employees)
    new_capital = jnp.where(bankrupt, 0.0, firms.capital_goods)
    
    # Entry
    key, subkey = jax.random.split(key)
    enters = (jax.random.uniform(subkey, firms.is_active.shape) < config.firm_entry_probability) & ~new_is_active
    
    new_is_active = jnp.where(enters, True, new_is_active)
    new_cash = jnp.where(enters, config.initial_firm_cash_min, new_cash)
    new_capital = jnp.where(enters, 10.0, new_capital)
    new_price = jnp.where(enters, (config.base_price_min + config.base_price_max)/2, firms.price)
    new_debt = jnp.where(enters, config.initial_firm_cash_min, new_debt)
    new_macro_loans = new_macro_loans + jnp.sum(jnp.where(enters, new_debt, 0.0))
    
    # Layoff employees of bankrupt firms
    def update_agent(emp, emp_id):
        is_bankrupt_firm = bankrupt[emp_id] & (emp_id >= 0)
        return jnp.where(is_bankrupt_firm, False, emp), jnp.where(is_bankrupt_firm, -1, emp_id)
        
    new_employed, new_employer_id = jax.vmap(update_agent)(agents.employed, agents.employer_id)
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
    
    return state._replace(agents=new_agents, firms=new_firms, macro=new_macro, rng_key=key)

@jax.jit
def simulation_step(state: SimState, config: SimulationConfig) -> SimState:
    """Master compiled step function that runs one complete tick of the economy."""
    old_cum_cost = state.firms.cumulative_cost
    
    def calc_net_money(s):
        return jnp.sum(s.agents.budget) + jnp.sum(s.agents.savings) + jnp.sum(s.firms.cash) + s.gov.cash + s.macro.bank_equity - s.macro.loans

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
    
    m1 = calc_net_money(state)
    sfc_delta = jnp.abs(m1 - m0)
    new_macro = state.macro._replace(sfc_delta=sfc_delta)
    
    return state._replace(macro=new_macro)
