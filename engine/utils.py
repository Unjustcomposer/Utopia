import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

def _update_ring_buffer(hist: jnp.ndarray, new_val: jnp.ndarray, index: jnp.ndarray) -> jnp.ndarray:
    """Updates the ring buffer at the given index in-place via XLA."""
    return hist.at[:, index].set(new_val)

def ste_boolean_mask(logits: jnp.ndarray) -> jnp.ndarray:
    """Straight-Through Estimator (STE) for boolean masks (x > 0)."""
    hard = (logits > 0).astype(jnp.float32)
    soft = jax.nn.sigmoid(logits)
    return jax.lax.stop_gradient(hard - soft) + soft

