"""
2021 Supply Chain Crunch — Historical Backtest
===============================================
Replicates the 2021 supply chain crisis and compares simulation output 
against actual FRED/BLS macro data for directional validation.

Run: python backtest_2021_supply_chain.py
"""
import jax
import jax.numpy as jnp
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import SimulationConfig
from simulation_jax import init_sim_state
from engine_jax import simulation_step


# ───────────────────────────────────────────────────────────────────────
# Warm-up: let the economy reach steady state before applying shocks
# ───────────────────────────────────────────────────────────────────────
WARMUP_TICKS = 60
SHOCK_TICKS  = 120
TOTAL_TICKS  = WARMUP_TICKS + SHOCK_TICKS


def run_2021_backtest():
    print("=" * 60)
    print("  UTOPIA — 2021 Supply Chain Crunch Backtest")
    print("=" * 60)

    config = SimulationConfig(
        num_ticks=TOTAL_TICKS,
        num_agents=500,
        num_firms=50,
        use_us_calibration=False,
        firm_behavior_mode=2,      # Heuristic for stability
    )

    # --- Load calibration profile if available ---
    try:
        from us_calibration import load_calibration_profile
        profile = load_calibration_profile("us_retail_2021")
        print(f"✓ Loaded calibration profile: us_retail_2021")
        if isinstance(profile, dict):
            macro = profile.get("macro_baselines", {})
            if macro:
                print(f"  FRED baseline rate: {macro.get('federal_funds_rate', 'N/A')}")
                print(f"  FRED savings rate:  {macro.get('personal_savings_rate', 'N/A')}")
    except Exception as e:
        print(f"ℹ  Calibration profile not loaded ({e}), using inline defaults.")

    # --- Initialize state ---
    state = init_sim_state(config, seed=2021)
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 0: WARM-UP — reach economic steady state (no shocks)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n◆ Warm-up phase: {WARMUP_TICKS} ticks (no shocks)...")
    for tick in range(WARMUP_TICKS):
        state = simulation_step(state, config)
    
    # Record steady-state baseline
    ss_price = float(state.macro.price_index)
    ss_employed = float(jnp.sum(state.agents.employed.astype(jnp.float32)))
    ss_unemp = 1.0 - (ss_employed / config.num_agents)
    ss_inventory = float(jnp.sum(state.firms.inventory))
    ss_demand = float(jnp.sum(state.firms.demand_history[:, int(state.macro.memory_count) % 3]))
    ss_inv_sales = ss_inventory / max(0.01, ss_demand)
    ss_production_capacity = state.firms.production_capacity  # Save for absolute shock application
    
    print(f"  Steady state reached:")
    print(f"    Price index:      {ss_price:.2f}")
    print(f"    Unemployment:     {ss_unemp*100:.1f}%")
    print(f"    Inventory/Sales:  {ss_inv_sales:.2f}")
    print(f"    Total employment: {int(ss_employed)}/{config.num_agents}")
    
    # ═══════════════════════════════════════════════════════════════════
    # PHASES 1-4: SUPPLY CHAIN CRISIS
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n◆ Shock phase: {SHOCK_TICKS} ticks with 2021 supply chain disruption...")
    
    metrics = []
    peak_price = ss_price
    min_inv_sales = ss_inv_sales
    
    for tick in range(SHOCK_TICKS):
        # ── Phase 1 (ticks 0-30): Port Congestion Onset ──
        # Freight costs ramp 1.0→2.0x; production starts getting constrained
        if tick < 30:
            progress = tick / 30.0
            target_cost = 1.0 + 1.0 * progress          # 1.0 → 2.0
            rate_delta = 0.0
            savings_delta = 0.0
            # Also reduce production capacity to simulate port bottleneck
            cap_penalty = 1.0 - 0.15 * progress          # up to 15% capacity loss
            
        # ── Phase 2 (ticks 30-60): Full Crisis ──
        # Container rates peak at 3x; consumers still spending stimulus
        elif tick < 60:
            progress = (tick - 30) / 30.0
            target_cost = 2.0 + 1.0 * progress           # 2.0 → 3.0
            rate_delta = 0.0
            savings_delta = -0.001                        # spending stimulus
            cap_penalty = 0.85 - 0.05 * progress          # 85% → 80% capacity
            
        # ── Phase 3 (ticks 60-90): Bullwhip / Whiplash ──
        # Costs declining; consumers pulling back; Fed starts signaling
        elif tick < 90:
            progress = (tick - 60) / 30.0
            target_cost = 3.0 - 1.5 * progress           # 3.0 → 1.5
            rate_delta = 0.0005
            savings_delta = 0.002
            cap_penalty = 0.80 + 0.15 * progress          # recovering to 95%
            
        # ── Phase 4 (ticks 90-120): Normalization ──
        # Costs near normal; rate hiking cycle begins
        else:
            progress = min(1.0, (tick - 90) / 30.0)
            target_cost = 1.5 - 0.3 * progress           # 1.5 → 1.2
            rate_delta = 0.001
            savings_delta = 0.0
            cap_penalty = 0.95 + 0.05 * progress          # recovering to 100%
        
        # Apply shocks — SET absolute values, don't compound
        state = state._replace(
            macro=state.macro._replace(
                base_rate=state.macro.base_rate + rate_delta
            ),
            agents=state.agents._replace(
                savings_rate=jnp.clip(state.agents.savings_rate + savings_delta, 0.0, 0.9)
            ),
            firms=state.firms._replace(
                input_cost_multiplier=jnp.full(config.num_firms, target_cost),
                production_capacity=ss_production_capacity * cap_penalty
            )
        )
        
        # Run simulation step
        state = simulation_step(state, config)
        
        # ── Collect metrics ──
        price_idx = float(state.macro.price_index)
        employed = float(jnp.sum(state.agents.employed.astype(jnp.float32)))
        unemp_rate = 1.0 - (employed / config.num_agents)
        
        total_inventory = float(jnp.sum(state.firms.inventory))
        mem_idx = int(state.macro.memory_count) % 3
        recent_demand = float(jnp.sum(state.firms.demand_history[:, mem_idx]))
        inv_to_sales = total_inventory / max(0.01, recent_demand) if recent_demand > 0.01 else total_inventory / max(0.01, 1.0)
        
        peak_price = max(peak_price, price_idx)
        min_inv_sales = min(min_inv_sales, inv_to_sales) if np.isfinite(inv_to_sales) else min_inv_sales
        
        metrics.append({
            "tick": tick,
            "price_index": price_idx,
            "unemployment": unemp_rate,
            "inv_to_sales": inv_to_sales,
            "cost_mult": target_cost,
            "employed": int(employed),
        })
        
        if tick % 30 == 29:
            phase = tick // 30 + 1
            print(f"  Phase {phase} (tick {tick+1}): "
                  f"price={price_idx:.2f}, unemp={unemp_rate*100:.1f}%, "
                  f"inv/sales={inv_to_sales:.2f}, employed={int(employed)}")

    # ═══════════════════════════════════════════════════════════════════
    # RESULTS COMPARISON
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS: Model vs. Actual 2021 FRED/BLS Data")
    print("=" * 80)
    
    end_price = metrics[-1]["price_index"]
    
    # Inflation: compare peak price vs steady-state baseline
    peak_inflation = ((peak_price - ss_price) / ss_price) * 100
    end_inflation = ((end_price - ss_price) / ss_price) * 100
    
    # Unemployment
    end_unemp = metrics[-1]["unemployment"] * 100
    ss_unemp_pct = ss_unemp * 100
    
    # Inventory-to-sales
    end_inv_sales = metrics[-1]["inv_to_sales"]
    
    print(f"\n  Steady-state baseline: price={ss_price:.2f}, unemp={ss_unemp_pct:.1f}%, inv/sales={ss_inv_sales:.2f}")
    print()
    print(f"  {'Metric':<26} | {'Actual 2021':<20} | {'Model Output':<22} | {'Dir.'}")
    print("  " + "-" * 82)
    
    # 1. Inflation
    inf_match = "✓" if peak_inflation > 0 else "✗"
    print(f"  {'Peak Inflation':<26} | {'+7.0% CPI-U YoY':<20} | {f'+{peak_inflation:.1f}% (peak vs SS)':<22} | {inf_match}")
    
    inf_end_match = "✓" if end_inflation > 0 else "~"
    print(f"  {'End Inflation':<26} | {'Elevated, sticky':<20} | {f'+{end_inflation:.1f}% vs SS':<22} | {inf_end_match}")
    
    # 2. Inventory-to-Sales
    inv_dropped = min_inv_sales < ss_inv_sales * 0.95
    inv_rebounded = end_inv_sales > min_inv_sales * 1.05
    if inv_dropped and inv_rebounded:
        inv_match = "✓"
    elif inv_dropped or inv_rebounded:
        inv_match = "~"
    else:
        inv_match = "✗"
    print(f"  {'Inv-to-Sales Drop':<26} | {'1.25 → 1.08 (trough)':<20} | {f'{ss_inv_sales:.2f} → {min_inv_sales:.2f}':<22} | {'✓' if inv_dropped else '✗'}")
    print(f"  {'Inv-to-Sales Rebound':<26} | {'1.08 → 1.26':<20} | {f'{min_inv_sales:.2f} → {end_inv_sales:.2f}':<22} | {'✓' if inv_rebounded else '✗'}")
    
    # 3. Unemployment (should decrease as firms compete for labor)
    unemp_decreased = end_unemp < ss_unemp_pct
    unemp_match = "✓" if unemp_decreased else "✗"
    print(f"  {'Unemployment Trend':<26} | {'5.8% → 4.2% (down)':<20} | {f'{ss_unemp_pct:.1f}% → {end_unemp:.1f}%':<22} | {unemp_match}")
    
    # Score
    checks = [peak_inflation > 0, inv_dropped, inv_rebounded, unemp_decreased, end_inflation > 0]
    score = sum(checks)
    print(f"\n  Directional accuracy: {score}/{len(checks)} indicators match real-world direction")
    
    # ═══════════════════════════════════════════════════════════════════
    # CHARTS
    # ═══════════════════════════════════════════════════════════════════
    os.makedirs("data/backtest_2021_results", exist_ok=True)
    
    ticks = [m["tick"] for m in metrics]
    prices = [m["price_index"] for m in metrics]
    unemps = [m["unemployment"] * 100 for m in metrics]
    invs = [m["inv_to_sales"] for m in metrics]
    costs = [m["cost_mult"] for m in metrics]
    employed_counts = [m["employed"] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Utopia — 2021 Supply Chain Crunch Backtest\n"
                 f"(Warm-up: {WARMUP_TICKS} ticks, Shock: {SHOCK_TICKS} ticks, "
                 f"Agents: {config.num_agents}, Firms: {config.num_firms})",
                 fontsize=13, fontweight='bold')
    
    phase_labels = [(0, 'Congestion'), (30, 'Full Crisis'), (60, 'Bullwhip'), (90, 'Normalize')]
    
    for ax in axes.flat:
        for x, label in phase_labels:
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.4)
        ax.grid(alpha=0.2)
    
    # Price Index
    ax1 = axes[0, 0]
    ax1.plot(ticks, prices, color='#f85149', linewidth=2)
    ax1.axhline(y=ss_price, color='#f85149', linestyle=':', alpha=0.5, label=f'SS baseline ({ss_price:.1f})')
    ax1.set_title("Price Index (Inflation Proxy)")
    ax1.set_xlabel("Tick (shock phase)")
    ax1.set_ylabel("Price Level")
    ax1.legend(fontsize=8)
    
    # Inventory-to-Sales
    ax2 = axes[0, 1]
    ax2.plot(ticks, invs, color='#58a6ff', linewidth=2)
    ax2.axhline(y=ss_inv_sales, color='#58a6ff', linestyle=':', alpha=0.5, label=f'SS baseline ({ss_inv_sales:.2f})')
    ax2.set_title("Inventory-to-Sales Ratio")
    ax2.set_xlabel("Tick (shock phase)")
    ax2.set_ylabel("Ratio")
    ax2.legend(fontsize=8)
    
    # Employment
    ax3 = axes[1, 0]
    ax3.plot(ticks, employed_counts, color='#2ea043', linewidth=2)
    ax3.axhline(y=ss_employed, color='#2ea043', linestyle=':', alpha=0.5, label=f'SS baseline ({int(ss_employed)})')
    ax3.set_title(f"Employment (of {config.num_agents} agents)")
    ax3.set_xlabel("Tick (shock phase)")
    ax3.set_ylabel("Employed agents")
    ax3.legend(fontsize=8)
    
    # Cost multiplier
    ax4 = axes[1, 1]
    ax4.fill_between(ticks, costs, alpha=0.3, color='#f0883e')
    ax4.plot(ticks, costs, color='#f0883e', linewidth=2)
    ax4.set_title("Input Cost Multiplier (Shock Intensity)")
    ax4.set_xlabel("Tick (shock phase)")
    ax4.set_ylabel("Multiplier")
    
    plt.tight_layout()
    chart_path = "data/backtest_2021_results/backtest_2021_charts.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\n  ✓ Charts saved to {chart_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_2021_backtest()
