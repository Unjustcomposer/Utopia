"""
Strategy Search Module
======================
Parameter-search optimization over scenario variants.  This is a defined
parameter sweep, not a black-box magic strategy finder.

Search methods:
  1. Grid search    — exhaustive over discrete parameter grid
  2. Random search  — sample N random points from parameter space
  3. (optional) Bayesian optimization — surrogate-model-based

Seed-robustness validation:
  • Top candidates re-evaluated with FRESH seeds (not used during search)
  • Reports which strategies hold up vs. which overfit to search seeds
  • Catches and reports seed-overfitting explicitly

All results are statements about the simulation's internal dynamics
— never predictions about real companies, markets, or events.
This is a decision-support and portfolio-demonstration tool,
NOT a trading system.  No claim of "finding alpha" for real markets.
"""

from __future__ import annotations

import itertools
import time
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import SimulationConfig
from simulation import Simulation, SimulationResult
from scenario import Scenario
from experiment import Experiment, ExperimentResult, compute_ci


# ════════════════════════════════════════════════════════════════════════
#  Data Containers
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateResult:
    """Result of evaluating one parameter combination."""
    params: Dict[str, float]
    objective_values: np.ndarray       # one per seed
    objective_mean: float
    objective_ci_lower: float
    objective_ci_upper: float
    rank: int = 0

    def summary_line(self) -> str:
        params_str = ", ".join(f"{k}={v:.3g}" for k, v in self.params.items())
        return (
            f"  #{self.rank:<3d}  "
            f"obj = {self.objective_mean:12.2f}  "
            f"CI [{self.objective_ci_lower:12.2f}, {self.objective_ci_upper:12.2f}]  "
            f"| {params_str} (simulated)"
        )


@dataclass
class RobustnessCheck:
    """Results of re-evaluating a candidate with fresh seeds."""
    params: Dict[str, float]
    search_mean: float
    search_ci: Tuple[float, float]
    validation_mean: float
    validation_ci: Tuple[float, float]
    degradation_ratio: float
    ks_statistic: float
    ks_p_value: float
    is_overfit: bool
    robustness_score: float  # fraction of fresh seeds where candidate beats control

    def summary_line(self) -> str:
        flag = "[!OVERFIT]" if self.is_overfit else "[ROBUST]  "
        params_str = ", ".join(f"{k}={v:.3g}" for k, v in self.params.items())
        return (
            f"  {flag}  "
            f"search={self.search_mean:.2f}  "
            f"validation={self.validation_mean:.2f}  "
            f"degradation={self.degradation_ratio:.2f}x  "
            f"robustness={self.robustness_score:.0%}  "
            f"KS p={self.ks_p_value:.4f}  "
            f"| {params_str} (simulated)"
        )


@dataclass
class SearchResult:
    """Complete results of a strategy search."""
    objective_name: str
    method: str
    param_space: Dict[str, Any]
    candidates: List[CandidateResult] = field(default_factory=list)
    robustness_checks: List[RobustnessCheck] = field(default_factory=list)
    best_experiment: Optional[ExperimentResult] = None
    runtime_seconds: float = 0.0

    def print_report(self) -> str:
        lines = [
            "=" * 90,
            "  STRATEGY SEARCH REPORT (simulated)",
            "=" * 90,
            f"  Objective:     {self.objective_name}",
            f"  Method:        {self.method}",
            f"  Candidates:    {len(self.candidates)}",
            f"  Runtime:       {self.runtime_seconds:.1f}s",
            "-" * 90,
            "  RANKED CANDIDATES:",
            "-" * 90,
        ]
        for c in self.candidates[:20]:  # top 20
            lines.append(c.summary_line())

        if self.robustness_checks:
            lines.append("-" * 90)
            lines.append("  SEED-ROBUSTNESS VALIDATION (top candidates on fresh seeds):")
            lines.append("-" * 90)
            for rc in self.robustness_checks:
                lines.append(rc.summary_line())

        lines.append("=" * 90)
        lines.append("  All values are simulated outcomes, not real-world predictions.")
        lines.append("  This is a parameter search, not a claim of 'finding alpha.'")
        lines.append("=" * 90)
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  Objective Functions
# ════════════════════════════════════════════════════════════════════════

def firm_profit_objective(firm_id: int = 0) -> Callable:
    """Objective: maximize a firm's cumulative profit (simulated)."""
    def objective(result: SimulationResult) -> float:
        return result.firm_profit(firm_id)
    objective.__name__ = f"firm_{firm_id}_profit"
    return objective


def total_welfare_objective() -> Callable:
    """Objective: maximize total agent welfare (simulated)."""
    def objective(result: SimulationResult) -> float:
        return result.summary().get("final_welfare", 0)
    objective.__name__ = "total_welfare"
    return objective


def price_stability_objective() -> Callable:
    """Objective: minimize price volatility (simulated)."""
    def objective(result: SimulationResult) -> float:
        prices = result.metric_series("price_index")
        if len(prices) < 2:
            return 0.0
        return -float(np.std(prices))  # negate because we maximize
    objective.__name__ = "price_stability"
    return objective


def total_output_objective() -> Callable:
    """Objective: maximize total output (simulated)."""
    def objective(result: SimulationResult) -> float:
        return result.summary().get("total_output_sum", 0)
    objective.__name__ = "total_output"
    return objective


OBJECTIVES = {
    "firm_profit": firm_profit_objective,
    "welfare": total_welfare_objective,
    "price_stability": price_stability_objective,
    "total_output": total_output_objective,
}


# ════════════════════════════════════════════════════════════════════════
#  Strategy Search
# ════════════════════════════════════════════════════════════════════════

class StrategySearch:
    """Parameter search over scenario variants against a stated objective.

    This is an explicit, documented parameter sweep — not a black-box
    magic strategy finder.

    Args:
        config: Simulation configuration.
        scenario_factory: callable(params_dict) → Scenario instance.
        param_space: Dict of param_name → list of values (grid) or (min, max) tuple (random).
        objective: callable(SimulationResult) → float (higher is better).
        method: "grid" or "random".
        num_seeds_per_eval: Seeds per candidate during search.
        num_random_samples: Number of random samples (for "random" method).
        search_base_seed: Base seed for search phase.
        validation_base_seed: Base seed for validation phase (must differ from search).
        validation_num_seeds: Seeds for final validation of top candidates.
        top_k_validate: Number of top candidates to validate.
    """

    def __init__(
        self,
        config: SimulationConfig,
        scenario_factory: Callable[[Dict[str, float]], Scenario],
        param_space: Dict[str, Any],
        objective: Callable[[SimulationResult], float],
        method: str = "grid",
        num_seeds_per_eval: int = 10,
        num_random_samples: int = 50,
        search_base_seed: int = 100,
        validation_base_seed: int = 99999,
        validation_num_seeds: int = 30,
        top_k_validate: int = 5,
    ) -> None:
        self.config = config
        self.scenario_factory = scenario_factory
        self.param_space = param_space
        self.objective = objective
        self.method = method
        self.num_seeds_per_eval = num_seeds_per_eval
        self.num_random_samples = num_random_samples
        self.search_base_seed = search_base_seed
        self.validation_base_seed = validation_base_seed
        self.validation_num_seeds = validation_num_seeds
        self.top_k_validate = top_k_validate

    def run(self, progress_callback: Optional[Callable] = None) -> SearchResult:
        """Execute the search and return ranked results with robustness checks."""
        t0 = time.time()

        # Generate parameter combinations
        if self.method == "grid":
            param_combos = self._grid_combinations()
        elif self.method == "random":
            param_combos = self._random_combinations()
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'grid' or 'random'.")

        # Evaluate each combination
        candidates: List[CandidateResult] = []
        total = len(param_combos)

        for idx, params in enumerate(param_combos):
            obj_values = self._evaluate(params, self.search_base_seed, self.num_seeds_per_eval)
            mean_v, ci_lo, ci_hi = compute_ci(obj_values, self.config.confidence_level)
            candidates.append(CandidateResult(
                params=params,
                objective_values=obj_values,
                objective_mean=mean_v,
                objective_ci_lower=ci_lo,
                objective_ci_upper=ci_hi,
            ))
            if progress_callback:
                progress_callback(idx + 1, total)

        # Rank by objective mean (higher is better)
        candidates.sort(key=lambda c: c.objective_mean, reverse=True)
        for rank, c in enumerate(candidates, 1):
            c.rank = rank

        # Seed-robustness validation of top K
        top_k = candidates[:self.top_k_validate]
        robustness_checks = self._validate_robustness(top_k)

        # Full experiment for the #1 candidate
        best_scenario = self.scenario_factory(candidates[0].params)
        best_exp = Experiment(
            config=self.config,
            scenario=best_scenario,
            num_seeds=self.validation_num_seeds,
            base_seed=self.validation_base_seed + 1000,  # fresh seeds
        ).run()

        result = SearchResult(
            objective_name=getattr(self.objective, '__name__', 'custom'),
            method=self.method,
            param_space=self.param_space,
            candidates=candidates,
            robustness_checks=robustness_checks,
            best_experiment=best_exp,
            runtime_seconds=time.time() - t0,
        )

        return result

    def _evaluate(
        self, params: Dict[str, float], base_seed: int, num_seeds: int,
    ) -> np.ndarray:
        """Evaluate a parameter combination across multiple seeds."""
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(num_seeds)

        obj_values = []
        for seed_seq in child_seeds:
            scenario = self.scenario_factory(params)
            sim = Simulation(
                config=self.config.copy(), seed=seed_seq, scenario=scenario
            )
            result = sim.run()
            obj_values.append(self.objective(result))

        return np.array(obj_values)

    def _grid_combinations(self) -> List[Dict[str, float]]:
        """Generate all combinations from grid-defined param space.

        Interpretation rules:
          - tuple of 3 numbers: (min, max, step) → generates a range via np.arange
          - list or tuple of other lengths: discrete values used as-is
          - single scalar: wrapped in a list
        """
        keys = list(self.param_space.keys())
        value_lists = []
        for key in keys:
            spec = self.param_space[key]
            if isinstance(spec, tuple) and len(spec) == 3 and all(isinstance(s, (int, float)) for s in spec):
                # Tuple of 3 numbers → treat as (min, max, step)
                value_lists.append(
                    np.arange(spec[0], spec[1] + spec[2] * 0.5, spec[2]).tolist()
                )
            elif isinstance(spec, (list, tuple, np.ndarray)):
                # List or other iterable → discrete values
                value_lists.append(list(spec))
            else:
                value_lists.append([spec])

        combos = []
        for values in itertools.product(*value_lists):
            combos.append(dict(zip(keys, values)))
        return combos

    def _random_combinations(self) -> List[Dict[str, float]]:
        """Generate random samples from the parameter space."""
        rng = np.random.default_rng(self.search_base_seed)
        combos = []
        keys = list(self.param_space.keys())

        for _ in range(self.num_random_samples):
            params = {}
            for key in keys:
                spec = self.param_space[key]
                if isinstance(spec, (list, np.ndarray)):
                    params[key] = float(rng.choice(spec))
                elif isinstance(spec, tuple) and len(spec) == 2:
                    params[key] = float(rng.uniform(spec[0], spec[1]))
                elif isinstance(spec, tuple) and len(spec) == 3:
                    params[key] = float(rng.uniform(spec[0], spec[1]))
                else:
                    params[key] = float(spec)
            combos.append(params)
        return combos

    def _validate_robustness(
        self, top_candidates: List[CandidateResult],
    ) -> List[RobustnessCheck]:
        """Re-evaluate top candidates with fresh seeds and check for overfitting.

        This is the critical step that catches seed-specific overfitting.
        """
        checks = []

        for candidate in top_candidates:
            # Evaluate with completely fresh seeds
            val_values = self._evaluate(
                candidate.params,
                base_seed=self.validation_base_seed,
                num_seeds=self.validation_num_seeds,
            )

            val_mean, val_ci_lo, val_ci_hi = compute_ci(
                val_values, self.config.confidence_level
            )

            # Degradation ratio
            search_mean = candidate.objective_mean
            if abs(search_mean) > 1e-10:
                degradation = val_mean / search_mean
            else:
                degradation = 1.0 if abs(val_mean) < 1e-10 else float('inf')

            # KS test: do search and validation distributions differ?
            ks_stat, ks_p = stats.ks_2samp(candidate.objective_values, val_values)

            # Robustness score: fraction of validation seeds where objective > 0
            # (i.e., the strategy improves on what a zero-intervention scenario would do)
            # More precisely: compare to control
            control_values = self._evaluate_control(self.validation_base_seed, self.validation_num_seeds)
            robustness_score = float(np.mean(val_values > control_values))

            is_overfit = degradation < 0.5 or ks_p < 0.05

            checks.append(RobustnessCheck(
                params=candidate.params,
                search_mean=search_mean,
                search_ci=(candidate.objective_ci_lower, candidate.objective_ci_upper),
                validation_mean=val_mean,
                validation_ci=(val_ci_lo, val_ci_hi),
                degradation_ratio=degradation,
                ks_statistic=float(ks_stat),
                ks_p_value=float(ks_p),
                is_overfit=is_overfit,
                robustness_score=robustness_score,
            ))

        return checks

    def _evaluate_control(self, base_seed: int, num_seeds: int) -> np.ndarray:
        """Evaluate the control (no scenario) across seeds for comparison."""
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(num_seeds)
        values = []
        for seed_seq in child_seeds:
            sim = Simulation(config=self.config.copy(), seed=seed_seq, scenario=None)
            result = sim.run()
            values.append(self.objective(result))
        return np.array(values)


# ════════════════════════════════════════════════════════════════════════
#  Convenience Functions
# ════════════════════════════════════════════════════════════════════════

def quick_search(
    scenario_factory: Callable[[Dict[str, float]], Scenario],
    param_space: Dict[str, Any],
    objective: Callable[[SimulationResult], float],
    config: Optional[SimulationConfig] = None,
    method: str = "grid",
    num_seeds: int = 5,
) -> SearchResult:
    """Run a quick strategy search with minimal seeds (for prototyping)."""
    cfg = config or SimulationConfig(num_ticks=60)
    search = StrategySearch(
        config=cfg,
        scenario_factory=scenario_factory,
        param_space=param_space,
        objective=objective,
        method=method,
        num_seeds_per_eval=num_seeds,
        validation_num_seeds=num_seeds * 2,
        top_k_validate=3,
    )
    return search.run()
