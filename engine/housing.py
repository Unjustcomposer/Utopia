import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

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

