"""
Scenario Injection Framework
=============================
Defines interventions and shocks that can be injected into a running simulation.
Each scenario is a pure parameter mutation applied through the simulation's
existing agent/firm decision functions — no new subsystems required.

All scenarios model numeric parameter changes (spend, capacity, cost
multipliers, awareness, savings rates). Disasters and disruptions are
implemented purely as supply/demand/cost shocks — no narrative content.

Every result is a statement about the simulation's internal dynamics
— never a prediction about a real company, market, or event.

Scenario Types:
    1. MarketingCampaign  — shifts agent awareness for a good
    2. ProductLaunch      — adds a new good to the market
    3. FeatureChange      — modifies price/quality/availability mid-run
    4. SupplyDisruption   — reduces firm capacity / raises input costs
    5. DemandShock        — shifts agent risk-aversion / savings rate
    6. TradeDisruption    — raises cost / removes availability of goods
    7. CompositeScenario  — combines multiple scenarios
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core import Agent, Firm
    from config import SimulationConfig


class Scenario(ABC):
    """Abstract base for all scenario interventions.

    Subclasses implement ``_apply_intervention`` which is called only
    during the active window [start_tick, start_tick + duration).
    """

    def __init__(
        self, 
        start_tick: int, 
        duration: Optional[int] = None,
        target_region: str = "All",
        target_age_group: str = "All"
    ) -> None:
        self.start_tick = start_tick
        self.duration = duration  # None = permanent once started
        self.target_region = target_region
        self.target_age_group = target_age_group
        self._applied_ticks: List[int] = []

    def is_active(self, tick: int) -> bool:
        """Check if the scenario is active at the given tick."""
        if tick < self.start_tick:
            return False
        if self.duration is not None and tick >= self.start_tick + self.duration:
            return False
        return True
        
    def _is_agent_targeted(self, agent: "Agent") -> bool:
        """Check if an agent matches the demographic targeting filters."""
        if self.target_region != "All" and getattr(agent, "region", "All") != self.target_region:
            return False
        if self.target_age_group != "All" and getattr(agent, "age", "All") != self.target_age_group:
            return False
        return True

    def _is_firm_targeted(self, firm: "Firm") -> bool:
        """Check if a firm matches the regional targeting filters."""
        if self.target_region != "All" and getattr(firm, "region", "All") != self.target_region:
            return False
        return True

    def ticks_elapsed(self, tick: int) -> int:
        """Number of ticks since the scenario started."""
        return max(0, tick - self.start_tick)

    def apply(
        self,
        tick: int,
        agents: List["Agent"],
        firms: List["Firm"],
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> None:
        """Apply the scenario if it is active at this tick."""
        if self.is_active(tick):
            self._apply_intervention(tick, agents, firms, config, rng)
            self._applied_ticks.append(tick)

    @abstractmethod
    def _apply_intervention(
        self,
        tick: int,
        agents: List["Agent"],
        firms: List["Firm"],
        config: "SimulationConfig",
        rng: np.random.Generator,
    ) -> None:
        """Override in subclasses to implement the specific intervention."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this scenario."""
        ...

    def params_dict(self) -> Dict[str, Any]:
        """Return scenario parameters as a dict (for logging/search)."""
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["type"] = self.__class__.__name__
        return d


# ════════════════════════════════════════════════════════════════════════
#  1. Marketing Campaign
# ════════════════════════════════════════════════════════════════════════

class MarketingCampaign(Scenario):
    """Temporarily boosts agent awareness for a target good.

    Models an advertising campaign: a fraction of agents (``reach``) receive
    an awareness boost for ``target_good``, which decays each tick.

    Parameters:
        target_good: Index of the good being marketed.
        spend: Marketing budget (used for parameterization, not deducted).
        reach: Fraction of agents exposed (0.0 to 1.0).
        awareness_boost: How much awareness increases for exposed agents.
        decay_rate: Per-tick decay of the awareness boost.
    """

    def __init__(
        self,
        start_tick: int,
        duration: Optional[int],
        target_good: int,
        spend: float = 5000.0,
        reach: float = 0.5,
        awareness_boost: float = 0.4,
        decay_rate: float = 0.05,
        target_region: str = "All",
        target_age_group: str = "All",
    ) -> None:
        super().__init__(start_tick, duration, target_region, target_age_group)
        self.target_good = target_good
        self.spend = spend
        self.reach = reach
        self.awareness_boost = awareness_boost
        self.decay_rate = decay_rate
        self._exposed_agents: Optional[set] = None

    def _apply_intervention(self, tick, agents, firms, config, rng):
        # First tick: select exposed agents deterministically (not via RNG,
        # to avoid corrupting the CRN stream used for control/treatment pairing)
        if self._exposed_agents is None:
            n_exposed = max(1, int(len(agents) * self.reach))
            # Deterministic selection: pick first N agents by sorted ID
            all_ids = sorted(a.id for a in agents)
            self._exposed_agents = set(all_ids[:n_exposed])

        elapsed = self.ticks_elapsed(tick)
        # Decaying boost: boost * (1 - decay_rate)^elapsed
        current_boost = self.awareness_boost * ((1.0 - self.decay_rate) ** elapsed)

        for agent in agents:
            if not self._is_agent_targeted(agent):
                continue
            if agent.id in self._exposed_agents:
                # Boost awareness toward 1.0
                agent.awareness[self.target_good] = min(
                    1.0,
                    agent.awareness[self.target_good] + current_boost * 0.05
                )
                # Also boost preference weight for the target good (the real effect)
                # This shifts demand toward the marketed product
                pref_boost = current_boost * 0.02
                agent.preferences[self.target_good] += pref_boost
                # Re-normalize preferences to sum to 1
                pref_sum = agent.preferences.sum()
                if pref_sum > 0:
                    agent.preferences /= pref_sum

    def describe(self) -> str:
        return (
            f"MarketingCampaign(good={self.target_good}, spend=${self.spend:,.0f}, "
            f"reach={self.reach:.0%}, boost={self.awareness_boost}, "
            f"decay={self.decay_rate}, start={self.start_tick}, dur={self.duration})"
        )


# ════════════════════════════════════════════════════════════════════════
#  2. Product Launch
# ════════════════════════════════════════════════════════════════════════

class ProductLaunch(Scenario):
    """Adds a new good or modifies an existing good's attributes.

    When the scenario activates, a designated firm begins producing a new good.
    Agents receive a new preference weight and start with low awareness.

    Parameters:
        producing_firm: ID of the firm that produces the new good.
        price: Initial price of the new good.
        quality: Quality attribute (affects preference).
        initial_awareness: Starting awareness for agents (typically low).
    """

    def __init__(
        self,
        start_tick: int,
        producing_firm: int,
        price: float = 15.0,
        quality: float = 1.0,
        initial_awareness: float = 0.15,
    ) -> None:
        super().__init__(start_tick, duration=None)  # permanent
        self.producing_firm = producing_firm
        self.price = price
        self.quality = quality
        self.initial_awareness = initial_awareness
        self._launched = False

    def _apply_intervention(self, tick, agents, firms, config, rng):
        if self._launched:
            return  # one-time event

        new_good_id = config.num_goods  # next index
        config.num_goods += 1

        # Update the producing firm
        for firm in firms:
            if firm.id == self.producing_firm:
                firm.good_produced = new_good_id
                firm.price = self.price
                firm.quality = self.quality
                firm.inventory = 30.0  # initial stock
                break

        # Expand agent preference and awareness arrays
        for agent in agents:
            new_pref_weight = rng.uniform(0.05, 0.2)
            agent.preferences = np.append(agent.preferences, new_pref_weight)
            # Re-normalize
            agent.preferences /= agent.preferences.sum()
            agent.awareness = np.append(agent.awareness, self.initial_awareness)

        self._launched = True

    def describe(self) -> str:
        return (
            f"ProductLaunch(firm={self.producing_firm}, price=${self.price:.2f}, "
            f"quality={self.quality}, awareness={self.initial_awareness}, "
            f"start={self.start_tick})"
        )


# ════════════════════════════════════════════════════════════════════════
#  3. Feature Change
# ════════════════════════════════════════════════════════════════════════

class FeatureChange(Scenario):
    """Modifies price, quality, or availability of an existing good mid-run.

    Parameters:
        target_good: Index of the good to modify.
        new_price: New price (None = no change).
        new_quality: New quality multiplier (None = no change).
        availability_multiplier: Scale supply (0.0 = removed, 1.0 = unchanged).
    """

    def __init__(
        self,
        start_tick: int,
        duration: Optional[int],
        target_good: int,
        new_price: Optional[float] = None,
        new_quality: Optional[float] = None,
        availability_multiplier: float = 1.0,
        target_region: str = "All",
        target_age_group: str = "All",
    ) -> None:
        super().__init__(start_tick, duration, target_region, target_age_group)
        self.target_good = target_good
        self.new_price = new_price
        self.new_quality = new_quality
        self.availability_multiplier = availability_multiplier
        self._original_prices: Dict[int, float] = {}
        self._original_qualities: Dict[int, float] = {}

    def _apply_intervention(self, tick, agents, firms, config, rng):
        for firm in firms:
            if not self._is_firm_targeted(firm):
                continue
            if firm.good_produced != self.target_good:
                continue

            # Store originals on first application
            if firm.id not in self._original_prices:
                self._original_prices[firm.id] = firm.price
                self._original_qualities[firm.id] = firm.quality

            if self.new_price is not None:
                firm.price = self.new_price
            if self.new_quality is not None:
                firm.quality = self.new_quality
            firm.inventory *= self.availability_multiplier

    def describe(self) -> str:
        parts = [f"FeatureChange(good={self.target_good}"]
        if self.new_price is not None:
            parts.append(f"price=${self.new_price:.2f}")
        if self.new_quality is not None:
            parts.append(f"quality={self.new_quality}")
        parts.append(f"availability={self.availability_multiplier:.0%}")
        parts.append(f"start={self.start_tick}, dur={self.duration})")
        return ", ".join(parts)


# ════════════════════════════════════════════════════════════════════════
#  4. Supply Disruption (macro shock)
# ════════════════════════════════════════════════════════════════════════

class SupplyDisruption(Scenario):
    """Reduces a firm's production capacity and/or raises input costs.

    Models a supply-side shock as numeric parameter changes — no narrative
    or event-depicting content.

    Parameters:
        target_firm: ID of the affected firm.
        capacity_reduction: Fraction of capacity lost (0.0 to 1.0).
        cost_increase: Multiplier on input costs (1.0 = no change).
    """

    def __init__(
        self,
        start_tick: int,
        duration: int,
        target_firm: int,
        capacity_reduction: float = 0.5,
        cost_increase: float = 1.5,
        target_region: str = "All",
        target_age_group: str = "All",
    ) -> None:
        super().__init__(start_tick, duration, target_region, target_age_group)
        self.target_firm = target_firm
        self.capacity_reduction = capacity_reduction
        self.cost_increase = cost_increase
        self._original_capacity: Optional[float] = None
        self._original_cost_mult: Optional[float] = None

    def _apply_intervention(self, tick, agents, firms, config, rng):
        for firm in firms:
            if not self._is_firm_targeted(firm):
                continue
            if firm.id != self.target_firm and self.target_firm != -1:
                continue

            # Store originals on first tick
            if self._original_capacity is None:
                self._original_capacity = firm.production_capacity
                self._original_cost_mult = firm.input_cost_multiplier

            firm.production_capacity = self._original_capacity * (1.0 - self.capacity_reduction)
            firm.input_cost_multiplier = self._original_cost_mult * self.cost_increase

    def apply(self, tick, agents, firms, config, rng):
        """Override to handle restoration when duration expires."""
        if self.is_active(tick):
            self._apply_intervention(tick, agents, firms, config, rng)
            self._applied_ticks.append(tick)
        elif (
            self._original_capacity is not None
            and tick == self.start_tick + self.duration
        ):
            # Restore original values
            for firm in firms:
                if firm.id == self.target_firm:
                    firm.production_capacity = self._original_capacity
                    firm.input_cost_multiplier = self._original_cost_mult

    def describe(self) -> str:
        return (
            f"SupplyDisruption(firm={self.target_firm}, "
            f"capacity_loss={self.capacity_reduction:.0%}, "
            f"cost_increase={self.cost_increase:.1f}x, "
            f"start={self.start_tick}, dur={self.duration})"
        )


# ════════════════════════════════════════════════════════════════════════
#  5. Demand Shock (macro shock)
# ════════════════════════════════════════════════════════════════════════

class DemandShock(Scenario):
    """Shifts aggregate demand via agent risk-aversion and savings rate changes.

    Models recession fear, geopolitical uncertainty, or booms as parameter
    changes to agent behavior — not narrative content.

    Parameters:
        risk_aversion_delta: Added to each agent's risk_aversion.
        savings_rate_delta: Added to each agent's savings_rate.
    """

    def __init__(
        self,
        start_tick: int,
        duration: int,
        risk_aversion_delta: float = 0.2,
        savings_rate_delta: float = 0.1,
        target_region: str = "All",
        target_age_group: str = "All",
    ) -> None:
        super().__init__(start_tick, duration, target_region, target_age_group)
        self.risk_aversion_delta = risk_aversion_delta
        self.savings_rate_delta = savings_rate_delta
        self._originals: Dict[int, tuple] = {}  # agent_id → (risk_av, savings_rate)

    def _apply_intervention(self, tick, agents, firms, config, rng):
        for agent in agents:
            if not self._is_agent_targeted(agent):
                continue
            # Store originals on first application
            if agent.id not in self._originals:
                self._originals[agent.id] = (agent.risk_aversion, agent.savings_rate)

            orig_ra, orig_sr = self._originals[agent.id]
            agent.risk_aversion = np.clip(orig_ra + self.risk_aversion_delta, 0, 1)
            agent.savings_rate = np.clip(orig_sr + self.savings_rate_delta, 0, 0.9)

    def apply(self, tick, agents, firms, config, rng):
        """Override to handle restoration when duration expires."""
        if self.is_active(tick):
            self._apply_intervention(tick, agents, firms, config, rng)
            self._applied_ticks.append(tick)
        elif (
            self._originals
            and tick == self.start_tick + self.duration
        ):
            # Restore original values
            for agent in agents:
                if agent.id in self._originals:
                    agent.risk_aversion, agent.savings_rate = self._originals[agent.id]

    def describe(self) -> str:
        return (
            f"DemandShock(risk_aversion_delta={self.risk_aversion_delta:+.2f}, "
            f"savings_rate_delta={self.savings_rate_delta:+.2f}, "
            f"start={self.start_tick}, dur={self.duration})"
        )


# ════════════════════════════════════════════════════════════════════════
#  6. Trade Disruption (macro shock)
# ════════════════════════════════════════════════════════════════════════

class TradeDisruption(Scenario):
    """Raises cost or removes availability of a subset of goods.

    Models an embargo or supply-chain break as a market mechanism
    — purely numeric cost and availability changes.

    Parameters:
        affected_goods: List of good indices affected.
        cost_increase: Multiplier on prices for affected goods.
        availability_reduction: Fraction of supply removed (0.0 to 1.0).
    """

    def __init__(
        self,
        start_tick: int,
        duration: int,
        affected_goods: List[int],
        cost_increase: float = 1.8,
        availability_reduction: float = 0.4,
        target_region: str = "All",
        target_age_group: str = "All",
    ) -> None:
        super().__init__(start_tick, duration, target_region, target_age_group)
        self.affected_goods = affected_goods
        self.cost_increase = cost_increase
        self.availability_reduction = availability_reduction
        self._original_costs: Dict[int, float] = {}

    def _apply_intervention(self, tick, agents, firms, config, rng):
        for firm in firms:
            if not self._is_firm_targeted(firm):
                continue
            if firm.good_produced not in self.affected_goods:
                continue

            if firm.id not in self._original_costs:
                self._original_costs[firm.id] = firm.input_cost_multiplier

            firm.input_cost_multiplier = self._original_costs[firm.id] * self.cost_increase
            # Reduce available inventory each tick
            firm.inventory *= (1.0 - self.availability_reduction * 0.1)

    def apply(self, tick, agents, firms, config, rng):
        """Override to handle restoration when duration expires."""
        if self.is_active(tick):
            self._apply_intervention(tick, agents, firms, config, rng)
            self._applied_ticks.append(tick)
        elif (
            self._original_costs
            and tick == self.start_tick + self.duration
        ):
            for firm in firms:
                if firm.id in self._original_costs:
                    firm.input_cost_multiplier = self._original_costs[firm.id]

    def describe(self) -> str:
        return (
            f"TradeDisruption(goods={self.affected_goods}, "
            f"cost={self.cost_increase:.1f}x, "
            f"avail_reduction={self.availability_reduction:.0%}, "
            f"start={self.start_tick}, dur={self.duration})"
        )


# ════════════════════════════════════════════════════════════════════════
#  7. Composite Scenario
# ════════════════════════════════════════════════════════════════════════

class CompositeScenario(Scenario):
    """Combines multiple scenarios, applying each one's logic per tick.

    Useful for complex experiments (e.g., marketing campaign + price change).
    """

    def __init__(self, scenarios: List[Scenario]) -> None:
        if not scenarios:
            raise ValueError("CompositeScenario requires at least one sub-scenario.")
        earliest = min(s.start_tick for s in scenarios)
        super().__init__(start_tick=earliest, duration=None)
        self.scenarios = scenarios

    def is_active(self, tick: int) -> bool:
        return any(s.is_active(tick) for s in self.scenarios)

    def apply(self, tick, agents, firms, config, rng):
        for scenario in self.scenarios:
            scenario.apply(tick, agents, firms, config, rng)

    def _apply_intervention(self, tick, agents, firms, config, rng):
        pass  # handled by individual scenarios

    def describe(self) -> str:
        descs = [s.describe() for s in self.scenarios]
        return f"CompositeScenario([{', '.join(descs)}])"

    def params_dict(self):
        return {
            "type": "CompositeScenario",
            "sub_scenarios": [s.params_dict() for s in self.scenarios],
        }


# ════════════════════════════════════════════════════════════════════════
#  Factory helper
# ════════════════════════════════════════════════════════════════════════

SCENARIO_TYPES = {
    "marketing": MarketingCampaign,
    "product_launch": ProductLaunch,
    "feature_change": FeatureChange,
    "supply_disruption": SupplyDisruption,
    "demand_shock": DemandShock,
    "trade_disruption": TradeDisruption,
}


def create_scenario(scenario_type: str, **kwargs) -> Scenario:
    """Factory function for creating scenarios by name.

    Example:
        >>> s = create_scenario("marketing", start_tick=20, duration=30,
        ...                     target_good=0, spend=5000)
    """
    cls = SCENARIO_TYPES.get(scenario_type)
    if cls is None:
        raise ValueError(
            f"Unknown scenario type '{scenario_type}'. "
            f"Available: {list(SCENARIO_TYPES.keys())}"
        )
    return cls(**kwargs)
