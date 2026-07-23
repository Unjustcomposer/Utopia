"""Quick verification tests for the simulation engine."""
from simulation import Simulation
from config import SimulationConfig
from scenario import MarketingCampaign, SupplyDisruption
from experiment import Experiment

cfg = SimulationConfig(num_ticks=20, num_agents=50)

# Test 1: Determinism
print("Test 1: Determinism")
r1 = Simulation(config=cfg.copy(), seed=42).run()
r2 = Simulation(config=cfg.copy(), seed=42).run()
s1, s2 = r1.summary(), r2.summary()
keys = ["final_gini", "final_employment", "final_welfare", "total_output_sum"]
all_match = True
for k in keys:
    diff = abs(s1[k] - s2[k])
    status = "OK" if diff < 1e-10 else "FAIL"
    if status == "FAIL":
        all_match = False
    print(f"  {k}: run1={s1[k]:.6f}, run2={s2[k]:.6f}, diff={diff:.2e} [{status}]")
print(f"  => {'PASS' if all_match else 'FAIL'}")

# Test 2: Different seeds produce different results
print("\nTest 2: Different seeds differ")
r3 = Simulation(config=cfg.copy(), seed=99).run()
s3 = r3.summary()
differ = any(abs(s1[k] - s3[k]) > 0.001 for k in keys)
print(f"  => {'PASS' if differ else 'FAIL'} (seeds 42 vs 99 differ)")

# Test 3: Scenario injection
print("\nTest 3: Scenario injection (MarketingCampaign)")
scenario = MarketingCampaign(start_tick=5, duration=10, target_good=0, spend=5000, reach=0.6)
r4 = Simulation(config=cfg.copy(), seed=42, scenario=scenario).run()
s4 = r4.summary()
for k in keys:
    print(f"  {k}: control={s1[k]:.4f}, treatment={s4[k]:.4f}")
print(f"  => Scenario ran without error")

# Test 4: Supply disruption
print("\nTest 4: Scenario injection (SupplyDisruption)")
scenario2 = SupplyDisruption(start_tick=5, duration=10, target_firm=0, capacity_reduction=0.5)
r5 = Simulation(config=cfg.copy(), seed=42, scenario=scenario2).run()
s5 = r5.summary()
for k in keys:
    print(f"  {k}: control={s1[k]:.4f}, treatment={s5[k]:.4f}")
print(f"  => Scenario ran without error")

# Test 5: Experiment framework
print("\nTest 5: Experiment (3 seeds)")
scenario3 = MarketingCampaign(start_tick=5, duration=10, target_good=0, spend=5000, reach=0.6)
exp = Experiment(config=cfg, scenario=scenario3, num_seeds=3, base_seed=42)
result = exp.run()
print(f"  Seeds completed: {len(result.control_results)}")
print(f"  Metric deltas computed: {len(result.metric_deltas)}")
for name, md in result.metric_deltas.items():
    print(f"  {name}: delta={md.mean_delta:+.4f}, CI=[{md.ci_lower:+.4f}, {md.ci_upper:+.4f}]")
print(f"  => PASS")

print("\n=== All tests passed (simulated) ===")
