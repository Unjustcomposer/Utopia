import jax
import jax.numpy as jnp
import pytest

from utopia.core.lmm_model import FirmTransformer, get_initial_lmm_params

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

def test_checkpoint_round_trip(tmp_path):
    """Verify that training, saving, and loading LMM parameters works cleanly."""
    from utopia.core.checkpoint import save_lmm_checkpoint, load_lmm_checkpoint
    
    key = jax.random.PRNGKey(42)
    original_params = get_initial_lmm_params(key)
    
    checkpoint_file = tmp_path / "test_lmm_checkpoint.pkl"
    save_lmm_checkpoint(original_params, path=str(checkpoint_file))
    
    loaded_params = load_lmm_checkpoint(path=str(checkpoint_file))
    
    assert loaded_params is not None, "Failed to load checkpoint"
    
    # Check numerical equivalence for all leaf nodes
    original_leaves = jax.tree_util.tree_leaves(original_params)
    loaded_leaves = jax.tree_util.tree_leaves(loaded_params)
    
    for orig, loaded in zip(original_leaves, loaded_leaves):
        assert jnp.array_equal(orig, loaded)
        
    dummy_input = jnp.ones((20, 3, 5))
    model = FirmTransformer()
    
    dp_orig, dw_orig, prod_orig = model.apply({'params': original_params}, dummy_input)
    dp_load, dw_load, prod_load = model.apply({'params': loaded_params}, dummy_input)
    
    assert jnp.array_equal(dp_orig, dp_load)
    assert jnp.array_equal(dw_orig, dw_load)
    assert jnp.array_equal(prod_orig, prod_load)
