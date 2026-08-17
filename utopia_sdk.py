"""
Utopia Python SDK
==================
Enterprise client library for interacting with the Utopia Platform.
Abstracts away the complex JAX engine and exposes a clean, pythonic API.
"""

import jax
import jax.numpy as jnp
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state, _run_scan
from utopia.core.scenarios import generate_shock_matrix
from utopia.connectors.data_ingestion import GlobalBaselineCompiler

class UtopiaClient:
    def __init__(self, use_live_data=False):
        """
        Initializes the Utopia client.
        
        Args:
            use_live_data: If True, uses the FRED API to seed the engine with today's real-world state.
        """
        self.config = SimulationConfig(use_us_calibration=use_live_data)
        self.use_live_data = use_live_data
        print(f"[UtopiaClient] Initialized. Live Data Mode: {self.use_live_data}")
        
    def run_simulation(self, ticks=90, scenario="baseline", seed=42):
        """
        Runs a full macroeconomic simulation.
        
        Args:
            ticks: Number of months to simulate.
            scenario: The macroeconomic shock scenario to run.
            seed: Random seed for stochastic elements.
            
        Returns:
            Dictionary of resulting metrics over time.
        """
        print(f"[UtopiaClient] Booting digital twin (Ticks: {ticks}, Scenario: {scenario})...")
        
        # Compile Baseline
        overrides = None
        if self.use_live_data:
            compiler = GlobalBaselineCompiler(self.config)
            overrides, _ = compiler.compile_baseline()
            
        state = init_sim_state(self.config, seed=seed, baseline_state_overrides=overrides)
        shocks = jnp.array(generate_shock_matrix(ticks, scenario))
        
        print("[UtopiaClient] JAX Graph execution starting...")
        final_state, metrics = _run_scan(state, ticks, self.config, shocks)
        
        print("[UtopiaClient] Simulation complete.")
        return {
            "employment_rate": metrics["employment_rate"].tolist(),
            "gini": metrics["gini"].tolist(),
            "price_index": metrics["price_index"].tolist(),
            "total_output": metrics["total_output"].tolist(),
        }
