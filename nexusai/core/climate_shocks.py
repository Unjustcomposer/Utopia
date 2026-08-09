import jax
import jax.numpy as jnp
from nexusai.core.state import SimState

def apply_infrastructure_damage(state: SimState, severity: float) -> SimState:
    """Temporarily zeroes out production capacity for a subset of firms (representing physical damage)."""
    # Assuming severity is a scalar float between 0 and 1.
    # We'll use the state.rng_key to randomly select firms to damage based on severity.
    key, subkey = jax.random.split(state.rng_key)
    
    # Randomly select a subset of firms based on severity.
    # We zero out their capacity.
    rand_vals = jax.random.uniform(subkey, state.firms.production_capacity.shape)
    
    # If rand_vals < severity, production_capacity = 0, else keep it.
    new_capacity = jnp.where(rand_vals < severity, 0.0, state.firms.production_capacity)
    
    new_firms = state.firms._replace(production_capacity=new_capacity)
    return state._replace(firms=new_firms, rng_key=key)

def apply_route_closure(state: SimState, duration_penalty: float) -> SimState:
    """Increases input costs heavily and slashes trade velocity (representing canal/port closures)."""
    # Increase input cost multiplier.
    new_input_cost = state.firms.input_cost_multiplier * (1.0 + duration_penalty)
    
    # Slashes trade velocity: we can reduce production_capacity or simulate it by reducing inventory
    # The prompt says "slashes trade velocity". Reducing capacity directly or scaling it down.
    new_capacity = state.firms.production_capacity * jnp.clip(1.0 - duration_penalty, 0.1, 1.0)
    
    new_firms = state.firms._replace(
        input_cost_multiplier=new_input_cost,
        production_capacity=new_capacity
    )
    return state._replace(firms=new_firms)
