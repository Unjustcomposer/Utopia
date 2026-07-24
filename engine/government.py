import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

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

