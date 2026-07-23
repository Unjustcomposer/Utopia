"""
Experiment Framework
====================
Control/treatment runner with Common Random Numbers (CRN), multi-seed
repetition, and statistical analysis (confidence intervals, effect sizes).

For every scenario test:
  • Control: identical-seed simulation with no intervention
  • Treatment: simulation with the scenario applied
  • Multiple seeds for distribution, not a single number
  • Delta metrics with 95% confidence intervals

All results are statements about the simulation's internal dynamics
— never predictions about real companies, markets, or events.
"""

from __future__ import annotations

import os
import time
import numpy as np
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import SimulationConfig
from simulation import Simulation, SimulationResult, run_simulation
from scenario import Scenario


# ════════════════════════════════════════════════════════════════════════
#  Data Containers
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MetricDelta:
    """Statistical summary of treatment − control delta for one metric."""
    metric_name: str
    control_values: np.ndarray
    treatment_values: np.ndarray
    deltas: np.ndarray
    mean_delta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool   # CI excludes zero

    def summary_line(self) -> str:
        sig = "[Y]" if self.significant else "[N]"
        def fmt(v):
            s = f"{v:10.3f}"
            return ('+' + s.lstrip() if v > 0 else s) if v != 0 else s
        return (
            f"  {self.metric_name:<22s}  "
            f"delta = {fmt(self.mean_delta)}  "
            f"CI [{fmt(self.ci_lower)}, {fmt(self.ci_upper)}]  "
            f"d = {self.effect_size:.3f}  {sig} (simulated)"
        )


@dataclass
class ExperimentResult:
    """Complete results of a control/treatment experiment."""
    scenario_description: str
    num_seeds: int
    confidence_level: float
    control_results: List[SimulationResult] = field(default_factory=list)
    treatment_results: List[SimulationResult] = field(default_factory=list)
    metric_deltas: Dict[str, MetricDelta] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def print_report(self) -> str:
        """Generate a formatted text report of the experiment."""
        lines = [
            "=" * 78,
            "  EXPERIMENT REPORT (simulated)",
            "=" * 78,
            f"  Scenario:    {self.scenario_description}",
            f"  Seeds:       {self.num_seeds}",
            f"  Confidence:  {self.confidence_level:.0%}",
            f"  Runtime:     {self.runtime_seconds:.1f}s",
            "-" * 78,
            "  METRIC DELTAS (treatment - control):",
            "-" * 78,
        ]
        for md in self.metric_deltas.values():
            lines.append(md.summary_line())
        lines.append("=" * 78)
        lines.append("  All values are simulated outcomes, not real-world predictions.")
        lines.append("=" * 78)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  Statistical Helpers
# ════════════════════════════════════════════════════════════════════════

def compute_ci(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval using t-distribution.

    Returns (mean, ci_lower, ci_upper).
    Handles zero-variance data gracefully (returns mean for both bounds).
    """
    n = len(data)
    if n < 2:
        m = float(np.mean(data))
        return m, m, m
    mean_val = float(np.mean(data))
    sem_val = float(stats.sem(data))
    if sem_val == 0 or np.isnan(sem_val):
        return mean_val, mean_val, mean_val
    ci_low, ci_high = stats.t.interval(confidence, df=n - 1, loc=mean_val, scale=sem_val)
    return mean_val, float(ci_low), float(ci_high)



def compute_effect_size(control: np.ndarray, treatment: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    n1, n2 = len(control), len(treatment)
    if n1 < 2 or n2 < 2:
        return 0.0
    mean_diff = np.mean(treatment) - np.mean(control)
    pooled_var = ((n1 - 1) * np.var(control, ddof=1) + (n2 - 1) * np.var(treatment, ddof=1)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return float(mean_diff / pooled_std)


def paired_t_test(control: np.ndarray, treatment: np.ndarray) -> float:
    """Paired t-test p-value for matched (CRN) samples."""
    if len(control) < 2:
        return 1.0
    _, p_value = stats.ttest_rel(control, treatment)
    return float(p_value)


# ════════════════════════════════════════════════════════════════════════
#  Worker function (for multiprocessing)
# ════════════════════════════════════════════════════════════════════════

def _run_pair(args: Tuple) -> Tuple[Dict, Dict]:
    """Run one control + treatment pair. Must be a top-level function for pickling."""
    config_dict, seed_int, scenario_params = args

    config = SimulationConfig(**config_dict)

    # Control run (no scenario)
    control_sim = Simulation(config=config.copy(), seed=seed_int, scenario=None)
    control_result = control_sim.run()

    # Treatment run (with scenario)
    from scenario import create_scenario
    scenario = None
    if scenario_params is not None:
        stype = scenario_params.pop("_type")
        scenario = create_scenario(stype, **scenario_params)

    treatment_sim = Simulation(config=config.copy(), seed=seed_int, scenario=scenario)
    treatment_result = treatment_sim.run()

    return control_result.summary(), treatment_result.summary()


# ════════════════════════════════════════════════════════════════════════
#  Experiment Runner
# ════════════════════════════════════════════════════════════════════════

class Experiment:
    """Runs control/treatment experiments with statistical rigor.

    Uses Common Random Numbers (CRN): each seed produces both a control
    and a treatment run, ensuring the only difference is the scenario.

    Args:
        config: Simulation configuration.
        scenario: Scenario to test.
        num_seeds: Number of seed repetitions (default from config).
        base_seed: Starting seed for SeedSequence spawning.
    """

    def __init__(
        self,
        config: SimulationConfig,
        scenario: Scenario,
        num_seeds: Optional[int] = None,
        base_seed: int = 42,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.num_seeds = num_seeds or config.default_num_seeds
        self.base_seed = base_seed

    def run(self, progress_callback: Optional[Callable] = None) -> ExperimentResult:
        """Execute all control/treatment pairs and analyze results.

        Args:
            progress_callback: Optional callable(completed, total) for progress.

        Returns:
            ExperimentResult with all data and statistical analysis.
        """
        t0 = time.time()

        # Generate independent seeds via SeedSequence
        ss = np.random.SeedSequence(self.base_seed)
        child_seeds = ss.spawn(self.num_seeds)

        result = ExperimentResult(
            scenario_description=self.scenario.describe(),
            num_seeds=self.num_seeds,
            confidence_level=self.config.confidence_level,
        )

        # Run all pairs
        for idx, seed_seq in enumerate(child_seeds):
            seed_int = seed_seq.entropy if isinstance(seed_seq.entropy, int) else idx

            # Control run
            control_sim = Simulation(
                config=self.config.copy(), seed=seed_seq, scenario=None
            )
            control_result = control_sim.run()
            result.control_results.append(control_result)

            # Treatment run (same seed, with scenario)
            # We need a fresh scenario instance for each run to avoid state leakage
            treatment_scenario = self._clone_scenario()
            treatment_sim = Simulation(
                config=self.config.copy(), seed=np.random.SeedSequence(seed_seq.entropy),
                scenario=treatment_scenario
            )
            treatment_result = treatment_sim.run()
            result.treatment_results.append(treatment_result)

            if progress_callback:
                progress_callback(idx + 1, self.num_seeds)

        # Analyze
        result.metric_deltas = self._analyze(result)
        result.runtime_seconds = time.time() - t0

        return result

    def _clone_scenario(self) -> Scenario:
        """Create a fresh scenario instance to avoid shared state across seeds."""
        params = self.scenario.params_dict()
        stype = params.pop("type")
        # Remove internal state keys
        params = {k: v for k, v in params.items() if not k.startswith("_")}

        from scenario import create_scenario

        # Map class name back to factory type key
        type_map = {
            "MarketingCampaign": "marketing",
            "ProductLaunch": "product_launch",
            "FeatureChange": "feature_change",
            "SupplyDisruption": "supply_disruption",
            "DemandShock": "demand_shock",
            "TradeDisruption": "trade_disruption",
        }

        # Handle composite separately
        if stype == "CompositeScenario":
            from scenario import CompositeScenario
            sub_scenarios = []
            for sp in params.get("sub_scenarios", []):
                st = sp.pop("type")
                sp = {k: v for k, v in sp.items() if not k.startswith("_")}
                type_key = type_map.get(st, st.lower())
                sub_scenarios.append(create_scenario(type_key, **sp))
            return CompositeScenario(sub_scenarios)

        type_key = type_map.get(stype, stype.lower())
        return create_scenario(type_key, **params)

    def _analyze(self, result: ExperimentResult) -> Dict[str, MetricDelta]:
        """Compute treatment − control deltas with confidence intervals."""
        metrics_to_compare = [
            "final_gini", "final_employment", "final_price_index",
            "final_welfare", "total_output_sum",
        ]

        deltas_dict = {}

        for metric in metrics_to_compare:
            control_vals = np.array([
                r.summary().get(metric, 0) for r in result.control_results
            ])
            treatment_vals = np.array([
                r.summary().get(metric, 0) for r in result.treatment_results
            ])

            diff = treatment_vals - control_vals
            mean_d, ci_lo, ci_hi = compute_ci(diff, self.config.confidence_level)
            p_val = paired_t_test(control_vals, treatment_vals)
            eff_size = compute_effect_size(control_vals, treatment_vals)
            sig = (ci_lo > 0 or ci_hi < 0)  # CI excludes zero

            deltas_dict[metric] = MetricDelta(
                metric_name=metric,
                control_values=control_vals,
                treatment_values=treatment_vals,
                deltas=diff,
                mean_delta=mean_d,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                p_value=p_val,
                effect_size=eff_size,
                significant=sig,
            )

        return deltas_dict


# ════════════════════════════════════════════════════════════════════════
#  Quick experiment helper
# ════════════════════════════════════════════════════════════════════════

def quick_experiment(
    scenario: Scenario,
    config: Optional[SimulationConfig] = None,
    num_seeds: int = 10,
    base_seed: int = 42,
) -> ExperimentResult:
    """Run a quick experiment with fewer seeds (for prototyping/testing)."""
    cfg = config or SimulationConfig(num_ticks=60)
    exp = Experiment(config=cfg, scenario=scenario, num_seeds=num_seeds, base_seed=base_seed)
    return exp.run()
