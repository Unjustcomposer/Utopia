import jax
import jax.numpy as jnp
import optax
from config import SimulationConfig
from simulation_jax import _run_scan, init_sim_state
from state import SimState
from scenarios import generate_shock_matrix

def macroeconomic_objective(lmm_params, initial_state: SimState, config: SimulationConfig):
    """
    Runs the simulation and computes the macroeconomic loss.
    This is the core of the Large Macroeconomic Model (LMM).
    """
    # Bind the LMM parameters into the initial state
    state = initial_state._replace(lmm_params=lmm_params)
    
    # Generate baseline shocks matrix
    shocks_matrix = jnp.array(generate_shock_matrix(config.num_ticks, "baseline"))
    
    # Run the simulation
    final_state, stacked_metrics = _run_scan(state, config.num_ticks, config, shocks_matrix)
    
    # Objective: Maximize GDP (Total Output) while penalizing high inflation and Gini
    # We negate it because we are using gradient descent (minimizing loss)
    
    # Extract metrics
    total_output = jnp.mean(stacked_metrics["total_output"])
    price_index = stacked_metrics["price_index"]
    
    # Compute inflation (approximate as final / initial)
    inflation = jnp.abs((price_index[-1] - price_index[0]) / price_index[0])
    
    # Combine into a single scalar loss
    loss = -1.0 * total_output + 1000.0 * inflation
    return loss, stacked_metrics

def train_lmm():
    """
    End-to-End Training Loop for the Large Macroeconomic Model.
    Backpropagates gradients from the macro-objective directly into the Firm Transformer weights.
    """
    print("Initializing LMM End-to-End Training...")
    
    config = SimulationConfig(num_agents=1000, num_firms=100, num_goods=10, num_ticks=50)
    seed = 42
    
    initial_state = init_sim_state(config, seed)
    lmm_params = initial_state.lmm_params
    
    # Optax optimizer
    tx = optax.adam(learning_rate=1e-3)
    opt_state = tx.init(lmm_params)
    
    # JIT compile the value_and_grad function
    loss_and_grad_fn = jax.value_and_grad(macroeconomic_objective, has_aux=True)
    loss_and_grad_fn_jit = jax.jit(loss_and_grad_fn, static_argnames=("config",))
    
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}: Running forward sim and backward trace...")
        
        # Calculate loss and gradients
        (loss, aux), grads = loss_and_grad_fn_jit(lmm_params, initial_state, config)
        
        # Apply gradients
        updates, opt_state = tx.update(grads, opt_state, lmm_params)
        lmm_params = optax.apply_updates(lmm_params, updates)
        
        print(f"  Loss: {loss:.4f} | Final Price Index: {aux['price_index'][-1]:.4f} | Mean Output: {jnp.mean(aux['total_output']):.4f}")
        
    print("LMM Training Complete. The Firm Transformer has learned a macroeconomic policy.")

if __name__ == "__main__":
    train_lmm()
