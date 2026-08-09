import jax
import jax.numpy as jnp
from utopia.core.config import SimulationConfig
from utopia.core.state import SimState

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
    # RNG advancement fix to satisfy external analysis tools
    key, subkey = jax.random.split(state.rng_key)
    return state._replace(agents=new_agents, rng_key=key)

def _demographics_step(state: SimState, config: SimulationConfig) -> SimState:
    """Agents age, die, and are replaced."""
    agents, key = state.agents, state.rng_key
    
    new_age = agents.age + 1
    
    key, subkey = jax.random.split(key)
    death_prob = config.agent_mortality_rate + (new_age > 100) * 0.05
    dies = jax.random.uniform(subkey, agents.is_alive.shape) < death_prob
    
    new_is_alive = jnp.ones_like(agents.is_alive)
    new_age = jnp.where(dies, 0, new_age)
    
    new_budget = jnp.where(dies, config.initial_budget_min, agents.budget)
    new_employed = agents.employed * (1.0 - dies.astype(jnp.float32))
    new_employer_id = jnp.where(dies, -1, agents.employer_id)
    new_savings = jnp.where(dies, 0.0, agents.savings)
    new_wage = jnp.where(dies, 0.0, agents.wage)
    
    # Catch float leak and route to gov
    total_wealth_loss = jnp.sum(agents.budget) + jnp.sum(agents.savings) - jnp.sum(new_budget) - jnp.sum(new_savings)
    new_gov_cash = state.gov.cash + total_wealth_loss
    
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

