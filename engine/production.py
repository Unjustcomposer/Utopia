import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

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

