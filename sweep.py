import itertools
from train_rl import train_lmm
import mlflow

def run_sweep():
    # Define hyperparameter grid
    # Keeping it small for demo purposes
    learning_rates = [1e-3, 5e-4]
    epochs_list = [100]
    lambda_inf_list = [500.0, 1000.0]
    lambda_unemp_list = [1000.0]
    num_ticks = 50
    seed = 42

    grid = list(itertools.product(learning_rates, epochs_list, lambda_inf_list, lambda_unemp_list))
    
    print(f"Starting Hyperparameter Sweep with {len(grid)} configurations...")
    
    best_loss = float('inf')
    best_params_set = None
    
    for lr, ep, l_inf, l_unemp in grid:
        print(f"\n--- Running Sweep Config: LR={lr}, Epochs={ep}, L_inf={l_inf}, L_unemp={l_unemp} ---")
        try:
            val_loss = train_lmm(
                seed=seed, 
                epochs=ep, 
                num_ticks=num_ticks, 
                learning_rate=lr, 
                lambda_inf=l_inf, 
                lambda_unemp=l_unemp
            )
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_params_set = (lr, ep, l_inf, l_unemp)
                
        except Exception as e:
            print(f"Error running config LR={lr}, Epochs={ep}: {e}")
            
    print("\n=======================================================")
    print(" SWEEP COMPLETE ")
    print("=======================================================")
    print(f"Best Validation Loss: {best_loss:.4f}")
    if best_params_set:
        print(f"Best Configuration: LR={best_params_set[0]}, Epochs={best_params_set[1]}, L_inf={best_params_set[2]}, L_unemp={best_params_set[3]}")
    
    # Run a final full training on the best config to ensure the checkpoint is saved with it
    if best_params_set:
        print("\nRetraining on Best Configuration to save final checkpoint...")
        train_lmm(
            seed=seed, 
            epochs=200, # Use more epochs for the final run
            num_ticks=num_ticks, 
            learning_rate=best_params_set[0], 
            lambda_inf=best_params_set[2], 
            lambda_unemp=best_params_set[3]
        )

if __name__ == "__main__":
    run_sweep()
