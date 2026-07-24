import jax
import jax.numpy as jnp
import pytest

from lmm_model import FirmTransformer, get_initial_lmm_params

def test_lmm_shapes():
    """Verify that the LMM transformer produces the correct output shapes."""
    key = jax.random.PRNGKey(0)
    params = get_initial_lmm_params(key)
    
    # 20 firms, 5 economic state features
    # Input shape: (num_firms, seq_len, features) -> (20, 3, 5)
    dummy_input = jnp.ones((20, 3, 5))
    
    model = FirmTransformer()
    dp, dw, prod = model.apply({'params': params}, dummy_input)
    
    # Output should be (num_firms,) for the 3 firm actions:
    # 1. Price delta
    # 2. Production delta
    # 3. Wage delta
    assert dp.shape == (20,)
    assert dw.shape == (20,)
    assert prod.shape == (20,)

def test_lmm_gradients():
    """Verify that the LMM model is fully differentiable."""
    key = jax.random.PRNGKey(0)
    params = get_initial_lmm_params(key)
    
    dummy_input = jnp.ones((20, 3, 5))
    
    def loss_fn(p):
        model = FirmTransformer()
        dp, dw, prod = model.apply({'params': p}, dummy_input)
        # Dummy loss: maximize output
        return jnp.sum(dp) + jnp.sum(dw) + jnp.sum(prod)
        
    grad = jax.grad(loss_fn)(params)
    
    # Ensure gradients exist and are not NaN
    assert not jnp.any(jnp.isnan(grad["Dense_0"]["kernel"]))
