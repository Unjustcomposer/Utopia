import jax
import jax.numpy as jnp
import flax.linen as nn

class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x):
        # LayerNorm 1
        x_ln1 = nn.LayerNorm()(x)
        
        # Multi-Head Attention
        # Since seq_len is very small (e.g. 3 ticks of history), we just use standard attention
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x_ln1, x_ln1)
        x = x + attn
        
        # LayerNorm 2
        x_ln2 = nn.LayerNorm()(x)
        
        # MLP
        hidden_dim = int(self.dim * self.mlp_ratio)
        mlp = nn.Dense(hidden_dim)(x_ln2)
        mlp = nn.gelu(mlp)
        mlp = nn.Dense(self.dim)(mlp)
        
        x = x + mlp
        return x

class FirmTransformer(nn.Module):
    """
    A Mini-LMM (Large Macroeconomic Model) embedded within the JAX simulation.
    Processes historical firm state and macroeconomic state to output continuous policy actions.
    """
    dim: int = 64
    depth: int = 2
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        # x shape: (seq_len, feature_dim) where seq_len = 3 (history length)
        # Project features to hidden dimension
        x = nn.Dense(self.dim)(x)
        
        # Add positional embeddings
        seq_len = x.shape[-2]
        pos_emb = self.param('pos_emb', nn.initializers.normal(stddev=0.02), (1, seq_len, self.dim))
        x = x + pos_emb
        
        # Transformer blocks
        for _ in range(self.depth):
            x = TransformerBlock(dim=self.dim, num_heads=self.num_heads)(x)
            
        x = nn.LayerNorm()(x)
        
        # We only care about the last timestep representation to make a decision
        last_hidden = x[..., -1, :]
        
        # Policy Heads: Delta Price, Delta Wage, Target Production
        delta_price = nn.Dense(1, name='head_price')(last_hidden)
        delta_wage = nn.Dense(1, name='head_wage')(last_hidden)
        target_production = nn.Dense(1, name='head_prod')(last_hidden)
        
        # Apply activation functions for bounded actions
        # Price adjustment: tanh scaled to +/- 10%
        delta_price = nn.tanh(delta_price) * 0.1 
        
        # Wage adjustment: tanh scaled to +/- 5%
        delta_wage = nn.tanh(delta_wage) * 0.05
        
        # Target production is positive
        target_production = nn.softplus(target_production) * 100.0
        
        return jnp.squeeze(delta_price, axis=-1), jnp.squeeze(delta_wage, axis=-1), jnp.squeeze(target_production, axis=-1)

def get_initial_lmm_params(key):
    """Returns the PyTree of initialized weights for the FirmTransformer."""
    model = FirmTransformer()
    # Dummy input to initialize shapes: batch=1, seq_len=3, feature_dim=5
    dummy_input = jnp.zeros((1, 3, 5))
    variables = model.init(key, dummy_input)
    return variables['params']
