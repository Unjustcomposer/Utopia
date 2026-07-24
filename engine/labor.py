import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

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
    
    # Prevent NaN with safe power (pre-mask zero conditions)
    safe_U_pow = jnp.where(U > 0, jnp.power(jnp.maximum(1.0, U), alpha), 0.0)
    safe_V_pow = jnp.where(V > 0, jnp.power(jnp.maximum(1.0, V), 1 - alpha), 0.0)
    M = mu * safe_U_pow * safe_V_pow
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


