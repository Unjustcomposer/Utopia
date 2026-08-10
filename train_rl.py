import jax
import jax.numpy as jnp
import optax
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import _run_scan, init_sim_state
from utopia.core.state import SimState
from utopia.core.scenarios import generate_shock_matrix

import mlflow
import os

def macroeconomic_objective(lmm_params, initial_state: SimState, config: SimulationConfig, scenario: str = "baseline", lambda_inf: float = 1000.0, lambda_unemp: float = 1000.0, target_inf: float = 0.05, target_unemp: float = 0.08):
    """
    Runs the simulation and computes the macroeconomic loss.
    This is the core of the Large Macroeconomic Model (LMM).
    """
    # Bind the LMM parameters into the initial state
    state = initial_state._replace(lmm_params=lmm_params)
    
    # Generate shocks matrix
    shocks_matrix = jnp.array(generate_shock_matrix(config.num_ticks, scenario))
    
    # Run the simulation
    final_state, stacked_metrics = _run_scan(state, config.num_ticks, config, shocks_matrix)
    
    # Extract metrics
    total_output = jnp.mean(stacked_metrics["total_output"])
    price_index = stacked_metrics["price_index"]
    employment = stacked_metrics["employment_rate"]
    
    # Compute inflation (approximate as final / initial)
    inflation = jnp.abs((price_index[-1] - price_index[0]) / price_index[0])
    
    # Compute unemployment
    unemployment = jnp.mean(1.0 - employment)
    
    # Multi-Objective Lagrangian Relaxation
    # We want to maximize output, subject to inflation and unemployment constraints.
    # Negate because we are minimizing loss.
    inf_penalty = lambda_inf * jnp.square(jnp.maximum(0.0, inflation - target_inf))
    unemp_penalty = lambda_unemp * jnp.square(jnp.maximum(0.0, unemployment - target_unemp))
    
    loss = -1.0 * total_output + inf_penalty + unemp_penalty
    
    # Pack metrics for auxiliary output
    metrics = {
        "loss": loss,
        "total_output": total_output,
        "inflation": inflation,
        "unemployment": unemployment,
        "price_index": price_index
    }
    
    return loss, metrics

def train_lmm(seed: int = 42, epochs: int = 100, num_ticks: int = 50, learning_rate: float = 1e-3, lambda_inf: float = 1000.0, lambda_unemp: float = 1000.0):
    """
    End-to-End Training Loop for the Large Macroeconomic Model.
    Backpropagates gradients from the macro-objective directly into the Firm Transformer weights.
    """
    print(f"Initializing LMM End-to-End Training (LR: {learning_rate}, L_inf: {lambda_inf}, L_unemp: {lambda_unemp})...")
    
    config = SimulationConfig(num_agents=1000, num_firms=100, num_ticks=num_ticks)
    
    initial_state = init_sim_state(config, seed)
    lmm_params = initial_state.lmm_params
    
    from utopia.core.lmm_model import count_lmm_params
    param_count = count_lmm_params(lmm_params)
    print(f"LMM Policy Network instantiated with {param_count:,} parameters.")
    
    # Optax optimizer
    tx = optax.adam(learning_rate=learning_rate)
    opt_state = tx.init(lmm_params)
    
    # JIT compile the value_and_grad function
    # We need to wrap it to pass lambdas
    def objective_wrapper(params, state, conf, scen):
        return macroeconomic_objective(params, state, conf, scen, lambda_inf, lambda_unemp)
        
    loss_and_grad_fn = jax.value_and_grad(objective_wrapper, has_aux=True)
    loss_and_grad_fn_jit = jax.jit(loss_and_grad_fn, static_argnames=("scen",))
    
    training_scenarios = ["baseline", "recession", "oil_shock", "tariff_shock"]
    validation_scenarios = ["pandemic", "supply_chain_2021"]
    
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("utopia_lmm_training")
    
    best_val_loss = float('inf')
    best_params = lmm_params
    
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs": epochs,
            "lambda_inf": lambda_inf,
            "lambda_unemp": lambda_unemp,
            "num_ticks": num_ticks
        })
        
        for epoch in range(epochs):
            # Training Phase
            scenario = training_scenarios[epoch % len(training_scenarios)]
            
            # Vary seed per epoch for training diversity
            epoch_state = initial_state._replace(
                rng_key=jax.random.PRNGKey(seed + epoch),
                lmm_params=lmm_params
            )
            # Calculate loss and gradients
            (loss, aux), grads = loss_and_grad_fn_jit(lmm_params, epoch_state, config, scen=scenario)
            
            # Apply gradients
            updates, opt_state = tx.update(grads, opt_state, lmm_params)
            lmm_params = optax.apply_updates(lmm_params, updates)
            
            mlflow.log_metrics({
                f"train_loss": float(loss),
                f"train_output": float(aux["total_output"]),
                f"train_inflation": float(aux["inflation"]),
                f"train_unemployment": float(aux["unemployment"])
            }, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} [Train - {scenario}] | Loss: {loss:.4f} | Output: {aux['total_output']:.4f} | Inf: {aux['inflation']:.4f} | Unemp: {aux['unemployment']:.4f}")
            
            # Validation Phase (every 10 epochs and at the end)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                val_losses = []
                for val_scen in validation_scenarios:
                    (v_loss, v_aux), _ = loss_and_grad_fn_jit(lmm_params, epoch_state, config, scen=val_scen)
                    val_losses.append(v_loss)
                    mlflow.log_metrics({
                        f"val_loss_{val_scen}": float(v_loss),
                        f"val_output_{val_scen}": float(v_aux["total_output"]),
                        f"val_inflation_{val_scen}": float(v_aux["inflation"]),
                        f"val_unemployment_{val_scen}": float(v_aux["unemployment"])
                    }, step=epoch)
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                mlflow.log_metric("val_loss_mean", float(avg_val_loss), step=epoch)
                print(f"  --> [Validation] Mean Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = lmm_params
            
        print("LMM Training Complete. The Firm Transformer has learned a macroeconomic policy.")
        
        from utopia.core.checkpoint import save_lmm_checkpoint
        import hashlib
        config_hash = hashlib.md5(str(config).encode()).hexdigest()
        metadata = {"config_hash": config_hash, "epochs": epochs, "num_ticks": num_ticks, "val_loss": float(best_val_loss)}
        save_lmm_checkpoint(best_params, metadata=metadata)
        print(f"Saved LMM checkpoint to checkpoints/lmm_latest.pkl")
        
        mlflow.log_metric("best_val_loss", float(best_val_loss))
        
        # We can't log jax params directly via log_model easily without custom pyfunc, 
        # so logging as an artifact is fine for now, or just letting save_lmm_checkpoint handle it.
        mlflow.log_artifact("checkpoints/lmm_latest.pkl")
        
    return best_val_loss

if __name__ == "__main__":
    train_lmm()
