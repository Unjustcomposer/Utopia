"""
JAX Simulation Orchestrator
===========================
Wraps the pure functional JAX engine with the legacy Simulation API interface.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Dict, List, Optional
import numpy as np

from config import SimulationConfig
from state import AgentState, FirmState, MacroState, GovState, HousingState, ForeignState, SimState
from engine_jax import simulation_step
from lmm_model import get_initial_lmm_params

import dataclasses

@dataclasses.dataclass
class SimulationResult:
    seed: int
    config: SimulationConfig
    scenario_description: str
    metrics_history: List[Dict[str, Any]]
    final_firms: List[Dict[str, Any]]
    
    def summary(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]

def init_sim_state(config: SimulationConfig, seed: int) -> SimState:
    """Initializes the JAX PyTree state."""
    key = jax.random.PRNGKey(seed)
    
    # 1. Agents
    if config.use_us_calibration:
        from us_calibration import sample_demographics, sample_agent_financials
        rng = np.random.default_rng(seed)
        regions, ages = sample_demographics(rng, config.num_agents)
        budgets, expected_wages, savings_rates = sample_agent_financials(
            rng, regions, ages, 
            base_budget=config.initial_budget_max / 2,
            base_wage=config.base_wage_max / 2
        )
        budget = jnp.array(budgets, dtype=jnp.float32)
        wage = jnp.zeros(config.num_agents, dtype=jnp.float32)
        savings = jnp.zeros(config.num_agents, dtype=jnp.float32)
        savings_rate = jnp.array(savings_rates, dtype=jnp.float32)
        # We can extract expected wage to use as initial wage expectation
        past_avg_prices = jnp.full((config.num_agents, config.num_goods), jnp.mean(jnp.array(expected_wages)))
    else:
        key, subkey = jax.random.split(key)
        budget = jax.random.uniform(subkey, (config.num_agents,), minval=config.initial_budget_min, maxval=config.initial_budget_max)
        wage = jnp.zeros(config.num_agents)
        savings = jnp.zeros(config.num_agents)
        
        key, subkey = jax.random.split(key)
        savings_rate = jax.random.uniform(subkey, (config.num_agents,), minval=config.savings_rate_min, maxval=config.savings_rate_max)
        
        past_avg_prices = jnp.full((config.num_agents, config.num_goods), config.base_wage_min)
        
    employed = jnp.zeros(config.num_agents, dtype=jnp.bool_)
    employer_id = jnp.full(config.num_agents, -1, dtype=jnp.int32)
    
    key, subkey = jax.random.split(key)
    risk_aversion = jax.random.uniform(subkey, (config.num_agents,), minval=config.risk_aversion_min, maxval=config.risk_aversion_max)
    
    key, subkey = jax.random.split(key)
    preferences = jax.random.uniform(subkey, (config.num_agents, config.num_goods))
    preferences = preferences / preferences.sum(axis=1, keepdims=True)
    
    key, subkey = jax.random.split(key)
    awareness = jax.random.uniform(subkey, (config.num_agents, config.num_goods))
    
    key, subkey = jax.random.split(key)
    skill = jax.random.uniform(subkey, (config.num_agents,), minval=config.skill_min, maxval=config.skill_max)
    is_alive = jnp.ones(config.num_agents, dtype=jnp.bool_)
    age = jnp.zeros(config.num_agents, dtype=jnp.int32)
    
    # Module 3
    key, subkey = jax.random.split(key)
    region_id = jax.random.randint(subkey, (config.num_agents,), 0, config.num_regions)
    housing_status = jnp.zeros(config.num_agents, dtype=jnp.int32)
    housing_wealth = jnp.zeros(config.num_agents, dtype=jnp.float32)
    inflation_expectations = jnp.zeros(config.num_agents, dtype=jnp.float32)
    
    key, subkey = jax.random.split(key)
    neighbors = jax.random.randint(subkey, (config.num_agents, 5), 0, config.num_agents)
    insured = jnp.zeros(config.num_agents, dtype=jnp.bool_)
    
    agents = AgentState(
        budget=budget, wage=wage, savings=savings, employed=employed,
        employer_id=employer_id, risk_aversion=risk_aversion, savings_rate=savings_rate,
        preferences=preferences, awareness=awareness, past_avg_prices=past_avg_prices,
        is_alive=is_alive, age=age, skill=skill,
        region_id=region_id, housing_status=housing_status, housing_wealth=housing_wealth,
        inflation_expectations=inflation_expectations, neighbors=neighbors, insured=insured
    )
    
    # 2. Firms
    good_produced = jnp.arange(config.num_firms) % config.num_goods
    
    key, subkey = jax.random.split(key)
    cash = jax.random.uniform(subkey, (config.num_firms,), minval=config.initial_firm_cash_min, maxval=config.initial_firm_cash_max)
    
    inventory = jnp.full(config.num_firms, 10.0)
    
    key, subkey = jax.random.split(key)
    price = jax.random.uniform(subkey, (config.num_firms,), minval=10.0, maxval=20.0)
    
    key, subkey = jax.random.split(key)
    quality = jax.random.uniform(subkey, (config.num_firms,), minval=0.8, maxval=1.2)
    
    production_capacity = jnp.full(config.num_firms, config.production_capacity_max)
    num_employees = jnp.zeros(config.num_firms, dtype=jnp.int32)
    wage_offer = jnp.full(config.num_firms, config.base_wage_min)
    debt = jnp.zeros(config.num_firms)
    cumulative_revenue = jnp.zeros(config.num_firms)
    cumulative_cost = jnp.zeros(config.num_firms)
    
    demand_history = jnp.zeros((config.num_firms, 3))
    revenue_history = jnp.zeros((config.num_firms, 3))
    price_history = jnp.zeros((config.num_firms, 3))
    profit_history = jnp.zeros((config.num_firms, 3))
    
    input_cost_multiplier = jnp.ones(config.num_firms)
    
    is_active = jnp.ones(config.num_firms, dtype=jnp.bool_)
    capital_goods = jnp.ones(config.num_firms) * 10.0
    equity = cash + inventory * price + capital_goods * config.capital_cost - debt
    
    key, subkey = jax.random.split(key)
    firm_region_id = jax.random.randint(subkey, (config.num_firms,), 0, config.num_regions)
    menu_cost_paid = jnp.zeros(config.num_firms, dtype=jnp.bool_)
    
    firms = FirmState(
        good_produced=good_produced, cash=cash, inventory=inventory, price=price, quality=quality,
        production_capacity=production_capacity, num_employees=num_employees, wage_offer=wage_offer,
        debt=debt, cumulative_revenue=cumulative_revenue, cumulative_cost=cumulative_cost,
        demand_history=demand_history, revenue_history=revenue_history, price_history=price_history,
        profit_history=profit_history, input_cost_multiplier=input_cost_multiplier,
        is_active=is_active, capital_goods=capital_goods, equity=equity,
        region_id=firm_region_id, menu_cost_paid=menu_cost_paid
    )
    
    # 3. Macro
    macro = MacroState(
        deposits=jnp.array(0.0), loans=jnp.array(0.0), base_rate=jnp.array(0.05),
        price_index=jnp.array(10.0), memory_count=jnp.array(0), bank_equity=jnp.array(0.0),
        sfc_delta=jnp.array(0.0)
    )
    
    # 4. Gov
    gov = GovState(
        tax_revenue=jnp.array(0.0), transfers_paid=jnp.array(0.0),
        bonds_outstanding=jnp.array(0.0), cash=jnp.array(0.0)
    )
    
    # 5. Housing & Foreign
    housing = HousingState(
        supply=jnp.ones(config.num_regions) * 1000.0,
        price=jnp.ones(config.num_regions) * 200.0
    )
    
    foreign = ForeignState(
        exchange_rate=jnp.array(1.0),
        exports=jnp.array(0.0),
        imports=jnp.array(0.0)
    )
    
    # Initialize LMM Transformer Weights
    key, subkey = jax.random.split(key)
    lmm_params = get_initial_lmm_params(subkey)
    
    return SimState(agents=agents, firms=firms, macro=macro, gov=gov, housing=housing, foreign=foreign, rng_key=key, lmm_params=lmm_params)


# We pass SimulationConfig directly since it is a flax PyTree
@partial(jax.jit, static_argnames=('num_ticks',))
def _run_scan(initial_state: SimState, num_ticks: int, config: SimulationConfig):
    
    def scan_body(state, _):
        new_state = simulation_step(state, config)
        
        # Calculate tick metrics to return as stacked arrays
        # (JAX lax.scan requires consistent shapes)
        tick_metrics = {
            "price_index": new_state.macro.price_index,
            "employment_rate": jnp.mean(new_state.agents.employed.astype(jnp.float32)),
            "total_output": jnp.sum(new_state.firms.inventory), # Simplified
            "gini": 0.0, # Expensive to calculate exact Gini in JAX inside the loop, set 0
            "total_welfare": jnp.sum(new_state.agents.savings)
        }
        
        return new_state, tick_metrics

    final_state, stacked_metrics = jax.lax.scan(scan_body, initial_state, None, length=num_ticks)
    return final_state, stacked_metrics

class JAXSimulation:
    def __init__(
        self,
        config: SimulationConfig,
        seed: Any,
        scenario: Optional[Any] = None,
    ) -> None:
        self.config = config
        
        if isinstance(seed, np.random.SeedSequence):
            import hashlib
            self.seed = seed.entropy if isinstance(seed.entropy, int) else int(hashlib.md5(str(seed.entropy).encode()).hexdigest(), 16) % (2**32)
        else:
            self.seed = int(seed)
            
        self.scenario = scenario # Scenarios are tricky to port to JAX statically right now, ignoring for MVP
        
        self.initial_state = init_sim_state(self.config, self.seed)

    def run(self) -> SimulationResult:
        final_state, stacked_metrics = _run_scan(self.initial_state, self.config.num_ticks, self.config)
        
        # Unpack stacked_metrics (JAX arrays) into a list of dicts for the legacy Dashboard
        metrics_history = []
        for i in range(self.config.num_ticks):
            metrics_history.append({
                "tick": i,
                "price_index": float(stacked_metrics["price_index"][i]),
                "employment_rate": float(stacked_metrics["employment_rate"][i]),
                "total_output": float(stacked_metrics["total_output"][i]),
                "gini": float(stacked_metrics["gini"][i]),
                "total_welfare": float(stacked_metrics["total_welfare"][i]),
            })
            
        # Reconstruct final agents/firms for summary
        # Just creating dummy data to satisfy the legacy interface
        final_firms = []
        for i in range(self.config.num_firms):
            final_firms.append({
                "id": i,
                "profit": float(final_state.firms.cumulative_revenue[i] - final_state.firms.cumulative_cost[i])
            })
            
        result = SimulationResult(
            seed=self.seed,
            config=self.config,
            scenario_description="JAX Accelerated Run",
            metrics_history=metrics_history,
            final_firms=final_firms
        )
        return result

def run_simulation(config: SimulationConfig, seed: Any, scenario: Optional[Any] = None) -> SimulationResult:
    sim = JAXSimulation(config=config, seed=seed, scenario=scenario)
    return sim.run()
