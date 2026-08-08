import jax
import jax.numpy as jnp
import os
import pandas as pd
from config import SimulationConfig
from simulation_jax import init_sim_state
from engine_jax import simulation_step
from shocks import apply_demand_shock, apply_supply_chain_disruption

def run_2008_scenario(firm_behavior_mode: int) -> dict:
    """
    Runs a single historical validation backtest replicating the 2008 Financial Crisis.
    """
    config = SimulationConfig(
        num_ticks=50, 
        num_agents=1000, 
        num_firms=100, 
        use_us_calibration=True,
        firm_behavior_mode=firm_behavior_mode
    )
    
    from checkpoint import load_lmm_checkpoint
    lmm_params = load_lmm_checkpoint() if firm_behavior_mode == 0 else None
    
    state = init_sim_state(config, seed=2008, lmm_params=lmm_params)
    
    metrics = []
    for tick in range(config.num_ticks):
        if tick == 10:
            # Severe panic savings (demand plummets)
            state = apply_demand_shock(state, savings_rate_increase=0.15)
            # Severe credit freeze (input costs spike)
            state = apply_supply_chain_disruption(state, cost_multiplier=1.8)
            
        state = simulation_step(state, config)
        
        # Calculate metrics
        def total_firm_skill(firm_id):
            return jnp.sum(jnp.where(state.agents.employer_id == firm_id, state.agents.skill, 0.0))
        firm_ids = jnp.arange(state.firms.cash.shape[0])
        effective_labor = jax.vmap(total_firm_skill)(firm_ids)
        labor_output = effective_labor * config.productivity_per_worker
        capacity = state.firms.capital_goods * 10.0
        raw_output = jnp.minimum(capacity, labor_output)
        raw_output = jnp.where(state.firms.is_active, raw_output, 0.0)
        production = raw_output / jnp.maximum(state.firms.input_cost_multiplier, 0.01)
        
        gdp = jnp.sum(production * state.firms.price)
        employed = jnp.sum(state.agents.employed)
        total_alive = jnp.sum(state.agents.is_alive)
        unemployment_rate = 1.0 - (employed / jnp.maximum(1.0, total_alive))
        
        metrics.append({
            "tick": tick,
            "gdp": float(gdp),
            "unemployment": float(unemployment_rate)
        })
    
    # ── Analysis ──
    baseline_gdp = sum(m["gdp"] for m in metrics[:10]) / 10
    baseline_unemp = sum(m["unemployment"] for m in metrics[:10]) / 10
    
    min_gdp = min(m["gdp"] for m in metrics[10:])
    max_unemp = max(m["unemployment"] for m in metrics[10:])
    
    gdp_drop_pct = ((baseline_gdp - min_gdp) / baseline_gdp) * 100
    unemp_spike_pct = (max_unemp - baseline_unemp) * 100
    
    return {
        "gdp_drop_pct": gdp_drop_pct,
        "unemp_spike_pct": unemp_spike_pct
    }

def run_comparative_backtest():
    print("\n=======================================================")
    print(" 2008 FINANCIAL CRISIS: LMM vs HEURISTIC COMPARISON")
    print("=======================================================")
    print("Simulating 2008 Macro Shocks under Heuristic Policy...")
    heu_res = run_2008_scenario(firm_behavior_mode=2)
    
    print("Simulating 2008 Macro Shocks under LMM AI Policy...")
    lmm_res = run_2008_scenario(firm_behavior_mode=0)
    
    # ── Empirical Validation ──
    actual_gdp_drop = 0.0
    actual_unemp_spike = 0.0
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "fred_2008_macro.csv")
        df = pd.read_csv(data_path)
        pre_crisis = df[df["Quarter"].str.contains("2007")]
        post_crisis = df[df["Quarter"].str.contains("2008|2009|2010")]
        
        actual_baseline_gdp = pre_crisis["RealGDP_Billions"].mean()
        actual_trough_gdp = post_crisis["RealGDP_Billions"].min()
        actual_gdp_drop = ((actual_baseline_gdp - actual_trough_gdp) / actual_baseline_gdp) * 100
        
        actual_baseline_unemp = pre_crisis["UnemploymentRate"].mean()
        actual_peak_unemp = post_crisis["UnemploymentRate"].max()
        actual_unemp_spike = (actual_peak_unemp - actual_baseline_unemp)
    except Exception as e:
        print(f"\nWarning: Could not load empirical data for validation. Error: {e}")
        
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'ACTUAL (FRED)':<15} | {'HEURISTIC (Z-I)':<18} | {'NEXUS LMM (AI)':<15}")
    print("-" * 80)
    
    heu_gdp_err = abs(heu_res['gdp_drop_pct'] - actual_gdp_drop)
    lmm_gdp_err = abs(lmm_res['gdp_drop_pct'] - actual_gdp_drop)
    print(f"{'GDP Drawdown (%)':<25} | {actual_gdp_drop:>14.2f}% | {heu_res['gdp_drop_pct']:>17.2f}% | {lmm_res['gdp_drop_pct']:>14.2f}%")
    print(f"{'  -> Tracking Error':<25} | {'-':>15} | {heu_gdp_err:>17.2f}% | {lmm_gdp_err:>14.2f}%")
    
    print("-" * 80)
    heu_unemp_err = abs(heu_res['unemp_spike_pct'] - actual_unemp_spike)
    lmm_unemp_err = abs(lmm_res['unemp_spike_pct'] - actual_unemp_spike)
    print(f"{'Unemployment Spike (pts)':<25} | {actual_unemp_spike:>14.2f}% | {heu_res['unemp_spike_pct']:>17.2f}% | {lmm_res['unemp_spike_pct']:>14.2f}%")
    print(f"{'  -> Tracking Error':<25} | {'-':>15} | {heu_unemp_err:>17.2f}% | {lmm_unemp_err:>14.2f}%")
    print("=" * 80)
    
    print("\n[CONCLUSION]")
    if lmm_gdp_err < heu_gdp_err and lmm_unemp_err < heu_unemp_err:
        print("LMM outperforms the Heuristic policy, tracking real-world 2008 macro dynamics significantly better.")
    else:
        print("LMM and Heuristic performance compared. (Train LMM longer to improve tracking error).")

if __name__ == "__main__":
    run_comparative_backtest()
