import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state
from utopia.core.engine_jax import simulation_step

config = SimulationConfig(num_agents=500, num_firms=50, num_goods=5)
key = 42
state = init_sim_state(config, key)

print("Init inv sum:", jnp.sum(state.firms.inventory))
state = simulation_step(state, config)
print("Step 1 inv sum:", jnp.sum(state.firms.inventory))
