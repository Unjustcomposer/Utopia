import jax.numpy as jnp
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state
from utopia.core.engine_jax import simulation_step

def test():
    config = SimulationConfig(num_agents=50, num_firms=2, num_goods=1, num_ticks=3, firm_behavior_mode=2)
    state = init_sim_state(config, seed=42)
    
    print("Initial inventory:", jnp.sum(state.firms.inventory))
    for i in range(3):
        print(f"--- TICK {i} ---")
        state = simulation_step(state, config)
        print("End of tick inventory:", jnp.sum(state.firms.inventory))
        print("End of tick production_capacity:", state.firms.production_capacity)
        print("End of tick demand_history:", state.firms.demand_history)
        
test()
