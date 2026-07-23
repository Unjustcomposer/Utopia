"""Debug scenario injection to verify treatments diverge from control."""
from simulation import Simulation
from config import SimulationConfig
from scenario import MarketingCampaign, SupplyDisruption

cfg = SimulationConfig(num_ticks=40, num_agents=50)

# Marketing Campaign with high reach and boost
scenario = MarketingCampaign(
    start_tick=5, duration=20, target_good=0,
    spend=5000, reach=0.8, awareness_boost=0.5
)
r1 = Simulation(config=cfg.copy(), seed=42, scenario=None).run()
r2 = Simulation(config=cfg.copy(), seed=42, scenario=scenario).run()

print("Marketing Campaign: tick-by-tick comparison")
for t in [0, 5, 10, 15, 20, 25, 30, 35]:
    m1 = r1.metrics_history[t]
    m2 = r2.metrics_history[t]
    d_out = m2["total_output"] - m1["total_output"]
    d_price = m2["price_index"] - m1["price_index"]
    print(f"  t={t:3d}: output_delta={d_out:+.4f}, price_delta={d_price:+.6f}")

# Supply Disruption with severe reduction
scenario2 = SupplyDisruption(
    start_tick=5, duration=15, target_firm=0,
    capacity_reduction=0.8, cost_increase=3.0
)
r3 = Simulation(config=cfg.copy(), seed=42, scenario=scenario2).run()

print("\nSupply Disruption: tick-by-tick comparison")
for t in [0, 5, 10, 15, 20, 25, 30, 35]:
    m1 = r1.metrics_history[t]
    m3 = r3.metrics_history[t]
    d_out = m3["total_output"] - m1["total_output"]
    d_price = m3["price_index"] - m1["price_index"]
    print(f"  t={t:3d}: output_delta={d_out:+.4f}, price_delta={d_price:+.6f}")

# Check final summaries
s1 = r1.summary()
s2 = r2.summary()
s3 = r3.summary()
print("\nFinal summary deltas vs control:")
for k in ["final_gini", "final_employment", "final_price_index", "final_welfare", "total_output_sum"]:
    print(f"  {k}: marketing={s2[k]-s1[k]:+.4f}, supply_shock={s3[k]-s1[k]:+.4f}")
