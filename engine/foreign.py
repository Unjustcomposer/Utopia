import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

def _foreign_trade_step(state: SimState, config: SimulationConfig) -> SimState:
    foreign, key = state.foreign, state.rng_key
    
    key, subkey = jax.random.split(key)
    shock = jax.random.normal(subkey) * config.exchange_rate_volatility
    new_exchange_rate = foreign.exchange_rate * jnp.exp(shock)
    
    exports = config.foreign_demand_base * new_exchange_rate
    imports = config.foreign_demand_base / new_exchange_rate
    
    new_foreign = foreign._replace(exchange_rate=new_exchange_rate, exports=exports, imports=imports)
    return state._replace(foreign=new_foreign, rng_key=key)

