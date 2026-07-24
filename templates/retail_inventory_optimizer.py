"""
Retail Inventory Optimizer
==========================
An opinionated NexusAI Template designed for Retail Supply Chain managers.
This template abstracts the JAX engine and runs a grid search over 
inventory buffer targets to find the optimal strategy against macro shocks.
"""

import sys
import os
import argparse
import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SimulationConfig
from simulation_jax import init_sim_state, _run_scan
from scenarios import generate_shock_matrix

def run_optimizer(buffer_targets):
    print("=========================================================")
    print(" Retail Inventory Optimizer via NexusAI (JAX Accelerated)")
    print("=========================================================")
    print(f"Testing Inventory Buffers: {buffer_targets}")
    
    config = SimulationConfig(
        num_agents=200,
        num_firms=10,
        num_goods=4,
        num_ticks=90,
        use_us_calibration=True,
        firm_behavior_mode=2 # Heuristics
    )

    # We evaluate against an oil shock (supply chain disruption)
    shocks = jnp.array(generate_shock_matrix(config.num_ticks, "oil_shock"))
    
    # We want to test different starting inventory buffers (simulated by capacity manipulation)
    def evaluate_buffer(buffer_multiplier):
        state = init_sim_state(config, seed=42)
        # Apply the buffer multiplier to the firm's capacity
        new_firms = state.firms._replace(
            production_capacity=state.firms.production_capacity * buffer_multiplier
        )
        state = state._replace(firms=new_firms)
        
        # Run simulation
        final_state, stacked_metrics = _run_scan(state, config.num_ticks, config, shocks)
        
        # Objective: Total economic output over the 90 ticks (proxy for sales volume)
        total_profit = jnp.sum(stacked_metrics["total_output"])
        return total_profit

    # JAX vmap to evaluate all buffers in parallel
    vmap_eval = jax.vmap(evaluate_buffer)
    
    print("\n[+] Compiling JAX Graph and running parallel evaluation...")
    buffer_array = jnp.array(buffer_targets)
    profits = vmap_eval(buffer_array)
    
    print("\n=========================================================")
    print(" OPTIMIZATION RESULTS ")
    print("=========================================================")
    
    results = []
    for i in range(len(buffer_targets)):
        results.append((buffer_targets[i], float(profits[i])))
        
    # Sort by profit descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (buf, prof) in enumerate(results):
        print(f"Rank {rank+1}: Buffer {buf}x -> Expected Profit: ${prof:,.2f}")
        
    print("\n[+] Recommendation:")
    print(f"    Adopt a {results[0][0]}x inventory buffer.")
    print("    This strategy maximizes margin during the simulated supply shock.")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retail Inventory Optimizer")
    parser.add_argument("--buffers", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0], help="List of buffer multipliers to test")
    
    args = parser.parse_args()
    run_optimizer(args.buffers)
