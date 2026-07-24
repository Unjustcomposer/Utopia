import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState
from engine.utils import _update_ring_buffer

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
    new_foreign_cash = state.foreign.cash - jnp.sum(export_demand_per_good * ratio * safe_prices)
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
    
    # Update histories (O(1) ring buffer)
    index = macro.memory_count % 3
    new_demand_history = _update_ring_buffer(firms.demand_history, firm_demands, index)
    new_revenue_history = _update_ring_buffer(firms.revenue_history, new_cum_revenue, index)
    new_price_history = _update_ring_buffer(firms.price_history, firms.price, index)
    
    prev_index = (macro.memory_count - 1) % 3
    tick_revenue = new_revenue_history[:, index] - new_revenue_history[:, prev_index]
    tick_cost = firms.cumulative_cost - old_cum_cost
    tick_profit = tick_revenue - tick_cost
    new_profit_history = _update_ring_buffer(firms.profit_history, tick_profit, index) 
    
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
    
    new_foreign = state.foreign._replace(cash=new_foreign_cash)
    
    return state._replace(agents=new_agents, firms=new_firms, macro=new_macro, gov=state.gov._replace(cash=new_gov_cash), foreign=new_foreign)


