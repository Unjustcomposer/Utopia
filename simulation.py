"""
Simulation Engine
=================
Tick-based simulation loop that orchestrates agents, firms, and market clearing.
Supports optional scenario injection via a hook called each tick.

All results are statements about the simulation's internal dynamics (simulated).

Architecture:
    1. Firms produce goods
    2. Firms pay wages to employees
    3. Scenario hook (if any) modifies state
    4. Agents decide purchases
    5. Market clears
    6. Firms adjust prices and hire/fire
    7. Agents update memory
    8. Metrics computed and stored
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from config import SimulationConfig
from core import Agent, Firm, Market
from metrics import aggregate_tick_metrics

if TYPE_CHECKING:
    pass


@dataclass
class SimulationResult:
    """Container for a completed simulation run.

    Attributes:
        seed: The random seed used.
        config: The simulation config.
        metrics_history: List of per-tick metric dicts.
        final_agents: Snapshot of all agents at the last tick.
        final_firms: Snapshot of all firms at the last tick.
        scenario_description: Human-readable description of the scenario (if any).
    """
    seed: int
    config: SimulationConfig
    metrics_history: List[Dict] = field(default_factory=list)
    final_agents: List[Dict] = field(default_factory=list)
    final_firms: List[Dict] = field(default_factory=list)
    scenario_description: str = "No scenario (control)"

    def metric_series(self, key: str) -> np.ndarray:
        """Extract a single metric as a time series."""
        return np.array([m[key] for m in self.metrics_history if key in m])

    def firm_profit(self, firm_id: int) -> float:
        """Cumulative profit for a specific firm (simulated)."""
        for fm in self.final_firms:
            if fm["id"] == firm_id:
                return fm["profit"]
        return 0.0

    def summary(self) -> Dict:
        """High-level summary of the run (simulated)."""
        if not self.metrics_history:
            return {}
        last = self.metrics_history[-1]
        return {
            "seed": self.seed,
            "scenario": self.scenario_description,
            "final_gini": last.get("gini", 0),
            "final_employment": last.get("employment_rate", 0),
            "final_price_index": last.get("price_index", 0),
            "final_welfare": last.get("total_welfare", 0),
            "total_output_sum": float(sum(
                m.get("total_output", 0) for m in self.metrics_history
            )),
        }


class Simulation:
    """Tick-based economic simulation.

    Args:
        config: Simulation configuration.
        seed: Random seed for reproducibility (int or SeedSequence).
        scenario: Optional scenario object with an ``apply(tick, agents, firms, market_data, rng)`` method.
    """

    def __init__(
        self,
        config: SimulationConfig,
        seed: Any,
        scenario: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.scenario = scenario

        # Create reproducible random generator
        if isinstance(seed, np.random.SeedSequence):
            self._seed_seq = seed
            self.seed = seed.entropy if isinstance(seed.entropy, int) else hash(str(seed.entropy))
        else:
            self._seed_seq = np.random.SeedSequence(seed)
            self.seed = int(seed)
        self.rng = np.random.default_rng(self._seed_seq)

        # Initialize entities
        self.agents: List[Agent] = []
        self.firms: List[Firm] = []
        self._agent_dict: Dict[int, Agent] = {}
        self._init_firms()
        self._init_agents()

    def _init_firms(self) -> None:
        """Create firms, distributing goods across them."""
        # Optional: distribute firms roughly evenly across US regions if US calibration is enabled
        from us_calibration import REGIONS
        
        for i in range(self.config.num_firms):
            good = i % self.config.num_goods
            region = REGIONS[i % len(REGIONS)] if self.config.use_us_calibration else "Midwest"
            firm = Firm(firm_id=i, good_produced=good, config=self.config, rng=self.rng, region=region)
            self.firms.append(firm)

    def _init_agents(self) -> None:
        """Create agents and assign initial employment."""
        from us_calibration import sample_demographics, sample_agent_financials
        
        demographics = []
        if self.config.use_us_calibration:
            demographics = sample_demographics(self.rng, self.config.num_agents)

        for i in range(self.config.num_agents):
            if self.config.use_us_calibration:
                region, age = demographics[i]
                agent = Agent(agent_id=i, config=self.config, rng=self.rng, age=age, region=region)
                budget, wage, savings_rate = sample_agent_financials(
                    self.rng, region, age, 
                    (self.config.initial_budget_min + self.config.initial_budget_max) / 2,
                    (self.config.base_wage_min + self.config.base_wage_max) / 2
                )
                agent.budget = budget
                agent.savings_rate = savings_rate
            else:
                agent = Agent(agent_id=i, config=self.config, rng=self.rng)
                
            self.agents.append(agent)
            self._agent_dict[i] = agent

        # Assign ~60% (US calibrated: 95%) of agents to jobs initially
        shuffled = list(range(self.config.num_agents))
        self.rng.shuffle(shuffled)
        emp_rate = 0.95 if self.config.use_us_calibration else 0.60
        target_employed = int(self.config.num_agents * emp_rate)
        for idx in range(target_employed):
            agent = self._agent_dict[shuffled[idx]]
            firm = self.firms[idx % len(self.firms)]
            agent.employed = True
            agent.employer_id = firm.id
            agent.wage = firm.wage_offer
            firm.employees.append(agent.id)

    def run(self) -> SimulationResult:
        """Execute the full simulation loop.

        Returns a SimulationResult with the complete metrics history.
        """
        result = SimulationResult(
            seed=self.seed,
            config=self.config,
            scenario_description=(
                self.scenario.describe() if self.scenario else "No scenario (control)"
            ),
        )

        for tick in range(self.config.num_ticks):
            market_data = self._step(tick)
            tick_metrics = aggregate_tick_metrics(
                self.agents, self.firms, market_data, tick
            )
            result.metrics_history.append(tick_metrics)

        # Final snapshots
        result.final_agents = [a.snapshot() for a in self.agents]
        result.final_firms = [f.snapshot() for f in self.firms]
        # Add profit to firm snapshots
        for fs in result.final_firms:
            fs["profit"] = fs["cumulative_revenue"] - fs["cumulative_cost"]

        return result

    def _step(self, tick: int) -> Dict:
        """Execute a single simulation tick."""

        # Phase 1: Firms produce goods
        for firm in self.firms:
            firm.produce()

        # Phase 2: Firms pay wages
        for firm in self.firms:
            firm.pay_wages(self._agent_dict)

        # Phase 3: Scenario hook
        if self.scenario is not None:
            self.scenario.apply(tick, self.agents, self.firms, self.config, self.rng)

        # Phase 4: Agents decide purchases
        prices = self._get_prices()
        available = self._get_supply()
        agent_demands: Dict[int, np.ndarray] = {}
        for agent in self.agents:
            demand = agent.decide_purchases(prices, available, self.rng)
            agent_demands[agent.id] = demand

        # Phase 5: Market clears
        market_data = Market.clear(
            self.agents, self.firms, agent_demands, self.config, self.rng
        )

        # Phase 6: Firms adjust prices and hire/fire
        unemployed = [a for a in self.agents if not a.employed]
        for firm in self.firms:
            firm.adjust_price()
            hired, fired = firm.hire_fire(unemployed, self.rng)
            # Update fired agents
            for aid in fired:
                agent = self._agent_dict.get(aid)
                if agent:
                    agent.employed = False
                    agent.employer_id = None
                    agent.wage = 0.0
            # Refresh unemployed list after each firm's action
            unemployed = [a for a in self.agents if not a.employed]

        return market_data

    def _get_prices(self) -> np.ndarray:
        """Get current average price per good across firms."""
        prices = np.zeros(self.config.num_goods)
        counts = np.zeros(self.config.num_goods)
        for firm in self.firms:
            prices[firm.good_produced] += firm.price
            counts[firm.good_produced] += 1
        safe_counts = np.where(counts > 0, counts, 1)
        return prices / safe_counts

    def _get_supply(self) -> np.ndarray:
        """Get current total supply (inventory) per good."""
        supply = np.zeros(self.config.num_goods)
        for firm in self.firms:
            supply[firm.good_produced] += max(firm.inventory, 0)
        return supply


def run_simulation(
    config: SimulationConfig,
    seed: Any,
    scenario: Optional[Any] = None,
) -> SimulationResult:
    """Convenience function: create and run a simulation in one call.

    Useful for parallel execution via ProcessPoolExecutor.
    """
    sim = Simulation(config=config, seed=seed, scenario=scenario)
    return sim.run()
