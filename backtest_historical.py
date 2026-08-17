import jax
import jax.numpy as jnp
import os
import pandas as pd
import requests
import argparse
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state
from utopia.core.engine_jax import simulation_step
from utopia.core.shocks import apply_demand_shock, apply_supply_chain_disruption

FRED_API_KEY = os.getenv("FRED_API_KEY", "8328c79609e22d49d6e9ba4c02295911")

def get_fred_data(series_id, start_date, end_date):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()["observations"]
    # return as pandas dataframe
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors='coerce')
    return df

def get_actual_metrics(era):
    if era == "2008":
        start, peak, end = "2007-01-01", "2008-01-01", "2010-01-01"
    elif era == "2020":
        start, peak, end = "2019-01-01", "2020-04-01", "2021-01-01"
    elif era == "2021":
        start, peak, end = "2020-01-01", "2021-01-01", "2022-01-01"
    else:
        raise ValueError("Unknown era")
        
    gdp_df = get_fred_data("GDPC1", start, end)
    unemp_df = get_fred_data("UNRATE", start, end)
    
    pre_gdp = gdp_df[gdp_df["date"] < peak]["value"].mean()
    trough_gdp = gdp_df[gdp_df["date"] >= peak]["value"].min()
    actual_gdp_drop = ((pre_gdp - trough_gdp) / pre_gdp) * 100
    
    pre_unemp = unemp_df[unemp_df["date"] < peak]["value"].mean()
    peak_unemp = unemp_df[unemp_df["date"] >= peak]["value"].max()
    actual_unemp_spike = peak_unemp - pre_unemp
    
    return actual_gdp_drop, actual_unemp_spike

def run_scenario(era: str, firm_behavior_mode: int) -> dict:
    config = SimulationConfig(
        num_ticks=50, 
        num_agents=1000, 
        num_firms=100, 
        use_us_calibration=True,
        firm_behavior_mode=firm_behavior_mode
    )
    
    from utopia.core.checkpoint import load_lmm_checkpoint
    lmm_params = load_lmm_checkpoint() if firm_behavior_mode == 0 else None
    
    # Use seed based on era
    seed = int(era)
    state = init_sim_state(config, seed=seed, lmm_params=lmm_params)
    
    metrics = []
    for tick in range(config.num_ticks):
        if tick == 10:
            if era == "2008":
                state = apply_demand_shock(state, savings_rate_increase=0.01)
                state = apply_supply_chain_disruption(state, cost_multiplier=1.03)
            elif era == "2020":
                state = apply_demand_shock(state, savings_rate_increase=0.02)
                state = apply_supply_chain_disruption(state, cost_multiplier=1.05)
            elif era == "2021":
                state = apply_supply_chain_disruption(state, cost_multiplier=1.04)
                # strong demand recovery
                state = apply_demand_shock(state, savings_rate_increase=-0.01)
            
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
        production = raw_output
        
        # Calculate Real GDP (using a constant base price of 15.0) to match FRED GDPC1
        gdp = jnp.sum(production * 15.0)
        employed = jnp.sum(state.agents.employed)
        total_alive = jnp.sum(state.agents.is_alive)
        unemployment_rate = 1.0 - (employed / jnp.maximum(1.0, total_alive))
        
        metrics.append({
            "tick": tick,
            "gdp": float(gdp),
            "unemployment": float(unemployment_rate) * 100 # convert to percentage
        })
    
    # Analysis
    baseline_gdp = sum(m["gdp"] for m in metrics[:10]) / 10
    baseline_unemp = sum(m["unemployment"] for m in metrics[:10]) / 10
    
    min_gdp = min(m["gdp"] for m in metrics[10:])
    max_unemp = max(m["unemployment"] for m in metrics[10:])
    
    gdp_drop_pct = ((baseline_gdp - min_gdp) / baseline_gdp) * 100
    unemp_spike_pct = (max_unemp - baseline_unemp)
    
    return {
        "gdp_drop_pct": gdp_drop_pct,
        "unemp_spike_pct": unemp_spike_pct
    }

def run_backtest(era: str):
    print(f"\n=======================================================")
    print(f" {era} MACRO SHOCK: LMM vs HEURISTIC COMPARISON")
    print(f"=======================================================")
    
    print(f"Fetching empirical FRED data for {era}...")
    try:
        actual_gdp_drop, actual_unemp_spike = get_actual_metrics(era)
    except Exception as e:
        print(f"Failed to fetch FRED data: {e}")
        return
        
    print(f"Simulating {era} Macro Shocks under Heuristic Policy...")
    heu_res = run_scenario(era, firm_behavior_mode=2)
    
    print(f"Simulating {era} Macro Shocks under LMM AI Policy...")
    lmm_res = run_scenario(era, firm_behavior_mode=0)
    
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'ACTUAL (FRED)':<15} | {'HEURISTIC (Z-I)':<18} | {'UTOPIA LMM (AI)':<15}")
    print("-" * 80)
    
    heu_gdp_err = abs(heu_res['gdp_drop_pct'] - actual_gdp_drop)
    lmm_gdp_err = abs(lmm_res['gdp_drop_pct'] - actual_gdp_drop)
    print(f"{'GDP Drawdown (%)':<25} | {actual_gdp_drop:>14.2f}% | {heu_res['gdp_drop_pct']:>17.2f}% | {lmm_res['gdp_drop_pct']:>14.2f}%")
    print(f"{'  -> Tracking Error':<25} | {'-':>15} | {heu_gdp_err:>17.2f} pts | {lmm_gdp_err:>14.2f} pts")
    
    print("-" * 80)
    heu_unemp_err = abs(heu_res['unemp_spike_pct'] - actual_unemp_spike)
    lmm_unemp_err = abs(lmm_res['unemp_spike_pct'] - actual_unemp_spike)
    print(f"{'Unemployment Spike (pts)':<25} | {actual_unemp_spike:>14.2f} | {heu_res['unemp_spike_pct']:>17.2f} | {lmm_res['unemp_spike_pct']:>14.2f}")
    print(f"{'  -> Tracking Error':<25} | {'-':>15} | {heu_unemp_err:>17.2f} pts | {lmm_unemp_err:>14.2f} pts")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", type=str, choices=["2008", "2020", "2021"], required=True)
    args = parser.parse_args()
    run_backtest(args.era)

if __name__ == "__main__":
    main()
