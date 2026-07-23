import jax
import jax.numpy as jnp
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
    print(f"Simulated GDP Drop:        {gdp_drop_pct:.2f}%")
    print(f"Actual 2008 GDP Drop:      4.30%")
    print(f"-> Error:                  {abs(gdp_drop_pct - 4.30):.2f}%\n")
    
    print(f"Baseline Unemployment: {baseline_unemp*100:.2f}% | Peak Unemployment: {max_unemp*100:.2f}%")
    print(f"Simulated Unemp. Spike:    {unemp_spike_pct:.2f}% (points)")
    print(f"Actual 2008 Unemp. Spike:  5.00% (from ~5% to 10%)")
    print(f"-> Error:                  {abs(unemp_spike_pct - 5.0):.2f}%")

if __name__ == "__main__":
    run_2008_backtest()
