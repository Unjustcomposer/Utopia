import jax
import jax.numpy as jnp
import os
import pandas as pd
from config import SimulationConfig
from simulation_jax import init_sim_state
from engine_jax import simulation_step
from shocks import apply_demand_shock, apply_supply_chain_disruption

def run_2008_backtest():
    """
    Runs a historical validation backtest replicating the 2008 Financial Crisis.
    We inject a massive DemandShock (panic savings) and SupplyDisruption (credit freeze)
    at tick 10, then measure the maximum drawdown in GDP and peak unemployment.
    """
    print("Initializing 2008 Historical Backtest...")
    # Scale up agents/firms for better statistical significance
    config = SimulationConfig(
        num_ticks=50, 
        num_agents=1000, 
        num_firms=100, 
        use_us_calibration=True,
        firm_behavior_mode=2 # Use Heuristic for baseline stability
    )
    
    state = init_sim_state(config, seed=2008)
    
    metrics = []
    print("Running Simulation...")
    for tick in range(config.num_ticks):
        if tick == 10:
            print("  [Tick 10] INJECTING 2008 FINANCIAL CRISIS SHOCKS")
            # Severe panic savings (demand plummets)
            state = apply_demand_shock(state, savings_rate_increase=0.15)
            # Severe credit freeze (input costs spike)
            state = apply_supply_chain_disruption(state, cost_multiplier=1.8)
            
        state = simulation_step(state, config)
        
        # Calculate metrics
        gdp = jnp.sum(state.firms.price * state.firms.inventory) # Proxy for output value
        employed = jnp.sum(state.agents.employed)
        total_alive = jnp.sum(state.agents.is_alive)
        unemployment_rate = 1.0 - (employed / jnp.maximum(1.0, total_alive))
        
        metrics.append({
            "tick": tick,
            "gdp": float(gdp),
            "unemployment": float(unemployment_rate)
        })
    
    # ── Analysis ──
    print("\n=== 2008 Backtest Results ===")
    
    # Pre-crisis baseline (avg of ticks 0-9)
    baseline_gdp = sum(m["gdp"] for m in metrics[:10]) / 10
    baseline_unemp = sum(m["unemployment"] for m in metrics[:10]) / 10
    
    # Post-crisis extremes
    min_gdp = min(m["gdp"] for m in metrics[10:])
    max_unemp = max(m["unemployment"] for m in metrics[10:])
    
    gdp_drop_pct = ((baseline_gdp - min_gdp) / baseline_gdp) * 100
    unemp_spike_pct = (max_unemp - baseline_unemp) * 100 # percentage points
    
    print(f"Baseline GDP: {baseline_gdp:.2f} | Trough GDP: {min_gdp:.2f}")
    # ── Empirical Validation ──
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "fred_2008_macro.csv")
        df = pd.read_csv(data_path)
        
        # Calculate empirical baselines
        pre_crisis = df[df["Quarter"].str.contains("2007")]
        post_crisis = df[df["Quarter"].str.contains("2008|2009|2010")]
        
        actual_baseline_gdp = pre_crisis["RealGDP_Billions"].mean()
        actual_trough_gdp = post_crisis["RealGDP_Billions"].min()
        actual_gdp_drop = ((actual_baseline_gdp - actual_trough_gdp) / actual_baseline_gdp) * 100
        
        actual_baseline_unemp = pre_crisis["UnemploymentRate"].mean()
        actual_peak_unemp = post_crisis["UnemploymentRate"].max()
        actual_unemp_spike = (actual_peak_unemp - actual_baseline_unemp)
        
        print("\n=== Empirical Validation (FRED Data) ===")
        print(f"Simulated GDP Drop:        {gdp_drop_pct:.2f}%")
        print(f"Actual 2008 GDP Drop:      {actual_gdp_drop:.2f}%")
        print(f"-> Error:                  {abs(gdp_drop_pct - actual_gdp_drop):.2f}%\n")
        
        print(f"Simulated Unemp. Spike:    {unemp_spike_pct:.2f}% (points)")
        print(f"Actual 2008 Unemp. Spike:  {actual_unemp_spike:.2f}% (from {actual_baseline_unemp:.2f}% to {actual_peak_unemp:.2f}%)")
        print(f"-> Error:                  {abs(unemp_spike_pct - actual_unemp_spike):.2f}%")
    except Exception as e:
        print(f"\nWarning: Could not load empirical data for validation. Error: {e}")
        
if __name__ == "__main__":
    run_2008_backtest()
