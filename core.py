"""
Core Simulation Entities
========================
Agent, Firm, and Market classes forming the base economic simulation.
Uses NumPy Generator API (not global seeds) for reproducible, independent
random streams. All market dynamics are internal to the simulation.

Every result produced is a statement about the simulation's internal dynamics
— never a prediction about a real company, market, or geopolitical event.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import SimulationConfig


# ════════════════════════════════════════════════════════════════════════
#  Agent
# ════════════════════════════════════════════════════════════════════════

class Agent:
    """A consumer/worker with preferences, awareness, memory, and budget.

    Decision model: Cobb-Douglas utility maximization gated by awareness.
    Agents only consider goods where ``awareness[g] > threshold``.
    """

    __slots__ = (
        "id", "wage", "budget", "savings", "employed", "employer_id",
        "risk_aversion", "savings_rate", "preferences", "awareness",
        "memory", "_cfg",
    )

    def __init__(
        self,
        agent_id: int,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.id = agent_id
        self._cfg = config
        self.wage = 0.0
        self.budget = rng.uniform(config.initial_budget_min, config.initial_budget_max)
        self.savings = 0.0
        self.employed = False
        self.employer_id: Optional[int] = None
        self.risk_aversion = rng.uniform(config.risk_aversion_min, config.risk_aversion_max)
        self.savings_rate = rng.uniform(config.savings_rate_min, config.savings_rate_max)

        # Preference weights per good (Cobb-Douglas exponents, sum to 1)
        raw_prefs = rng.dirichlet(np.ones(config.num_goods))
        self.preferences: np.ndarray = raw_prefs  # shape (num_goods,)

        # Awareness per good: starts uniform in [0.3, 0.9]
        self.awareness: np.ndarray = rng.uniform(0.3, 0.9, size=config.num_goods)

        # Memory: list of dicts, each {prices: ndarray, purchased: ndarray}
        self.memory: List[Dict] = []

    def receive_wage(self, amount: float) -> None:
        """Add wage income to budget."""
        self.budget += amount

    def decide_purchases(
        self,
        prices: np.ndarray,
        available: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Allocate spending across goods via awareness-gated utility maximization.

        Returns an array of quantities demanded per good.
        """
        # Save first
        save_amount = self.budget * self.savings_rate * (1.0 + self.risk_aversion * 0.5)
        save_amount = min(save_amount, self.budget)
        self.savings += save_amount
        spending_budget = self.budget - save_amount

        if spending_budget <= 0 or np.all(prices <= 0):
            return np.zeros(len(prices))

        # Awareness gate: only consider goods above threshold
        mask = self.awareness >= self._cfg.awareness_threshold
        mask &= available > 0
        mask &= prices > 0

        if not np.any(mask):
            return np.zeros(len(prices))

        # Cobb-Douglas allocation: spend_j = alpha_j / sum(alpha_visible) * budget
        visible_prefs = self.preferences * mask
        pref_sum = visible_prefs.sum()
        if pref_sum <= 0:
            return np.zeros(len(prices))

        allocation_shares = visible_prefs / pref_sum

        # Price elasticity adjustment: reduce demand for expensive goods
        price_factor = np.ones_like(prices)
        if len(self.memory) >= 2:
            avg_past_price = np.mean(
                [m["prices"] for m in self.memory[-3:]], axis=0
            )
            # If price rose, reduce demand proportionally to elasticity
            safe_avg = np.where(avg_past_price > 0, avg_past_price, prices)
            price_ratio = prices / safe_avg
            price_factor = np.power(price_ratio, -self._cfg.demand_elasticity)
            price_factor = np.clip(price_factor, 0.2, 3.0)

        adjusted_shares = allocation_shares * price_factor
        adj_sum = adjusted_shares.sum()
        if adj_sum > 0:
            adjusted_shares /= adj_sum

        spend_per_good = adjusted_shares * spending_budget
        quantities = spend_per_good / np.where(prices > 0, prices, 1.0)
        quantities = np.maximum(quantities, 0)

        return quantities

    def update_memory(self, prices: np.ndarray, purchased: np.ndarray) -> None:
        """Store this tick's market data in memory (sliding window)."""
        self.memory.append({"prices": prices.copy(), "purchased": purchased.copy()})
        if len(self.memory) > self._cfg.memory_window:
            self.memory.pop(0)

    def snapshot(self) -> Dict:
        """Return a serializable snapshot of agent state."""
        return {
            "id": self.id,
            "budget": self.budget,
            "savings": self.savings,
            "employed": self.employed,
            "employer_id": self.employer_id,
            "wage": self.wage,
            "risk_aversion": self.risk_aversion,
            "savings_rate": self.savings_rate,
            "awareness": self.awareness.tolist(),
        }


# ════════════════════════════════════════════════════════════════════════
#  Firm
# ════════════════════════════════════════════════════════════════════════

class Firm:
    """A producer that hires agents, sets prices, and responds to demand.

    Pricing rule: adaptive markup — raise price when demand exceeds supply,
    lower price when inventory accumulates beyond the target buffer.
    """

    __slots__ = (
        "id", "good_produced", "cash", "price", "quality", "inventory",
        "production_capacity", "employees", "revenue_history",
        "demand_history", "input_cost_multiplier", "_cfg",
        "cumulative_revenue", "cumulative_cost", "wage_offer",
    )

    def __init__(
        self,
        firm_id: int,
        good_produced: int,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.id = firm_id
        self.good_produced = good_produced
        self._cfg = config
        self.cash = rng.uniform(config.initial_firm_cash_min, config.initial_firm_cash_max)
        self.price = rng.uniform(config.base_price_min, config.base_price_max)
        self.quality = rng.uniform(0.5, 1.5)
        self.inventory = 50.0
        self.production_capacity = rng.uniform(
            config.production_capacity_min, config.production_capacity_max
        )
        self.employees: List[int] = []  # agent IDs
        self.wage_offer = rng.uniform(config.base_wage_min, config.base_wage_max)
        self.revenue_history: List[float] = []
        self.demand_history: List[float] = []
        self.input_cost_multiplier = 1.0
        self.cumulative_revenue = 0.0
        self.cumulative_cost = 0.0

    def produce(self) -> float:
        """Produce goods up to min(capacity, workforce output).

        Returns quantity produced this tick.
        """
        labor_output = len(self.employees) * self._cfg.productivity_per_worker
        raw_output = min(self.production_capacity, labor_output)
        # Input cost multiplier reduces effective output (supply disruption)
        effective_output = raw_output / max(self.input_cost_multiplier, 0.01)
        self.inventory += effective_output

        # Track input costs
        input_cost = effective_output * self._cfg.input_cost_base * self.input_cost_multiplier
        self.cash -= input_cost
        self.cumulative_cost += input_cost

        return effective_output

    def pay_wages(self, agents: Dict[int, Agent]) -> float:
        """Pay all employees; returns total wages paid."""
        total_wages = 0.0
        for agent_id in self.employees:
            agent = agents.get(agent_id)
            if agent is not None:
                agent.receive_wage(self.wage_offer)
                total_wages += self.wage_offer
        self.cash -= total_wages
        self.cumulative_cost += total_wages
        return total_wages

    def record_sale(self, quantity: float, revenue: float) -> None:
        """Record a completed sale."""
        self.inventory -= quantity
        self.cash += revenue
        self.cumulative_revenue += revenue

    def adjust_price(self) -> None:
        """Adaptive pricing based on recent demand vs. supply."""
        if len(self.demand_history) < 2:
            return

        recent_demand = np.mean(self.demand_history[-3:])
        target_inv = recent_demand * self._cfg.target_inventory_buffer

        if self.inventory < target_inv * 0.5:
            # Demand outstripping supply → raise price
            self.price *= (1.0 + self._cfg.price_adjustment_rate)
        elif self.inventory > target_inv * 1.5:
            # Excess inventory → lower price
            self.price *= (1.0 - self._cfg.price_adjustment_rate)

        # Floor price at input cost
        self.price = max(self.price, self._cfg.input_cost_base * self.input_cost_multiplier * 1.1)

    def hire_fire(
        self,
        unemployed_agents: List[Agent],
        rng: np.random.Generator,
    ) -> Tuple[List[int], List[int]]:
        """Expand or contract workforce based on profitability and demand.

        Returns (hired_ids, fired_ids).
        """
        hired, fired = [], []

        recent_rev = np.mean(self.revenue_history[-3:]) if self.revenue_history else 0
        wage_bill = len(self.employees) * self.wage_offer

        # Profit signal
        if len(self.demand_history) >= 2:
            recent_demand = np.mean(self.demand_history[-3:])
            labor_output = len(self.employees) * self._cfg.productivity_per_worker

            # Hire if demand exceeds output and we're profitable
            if recent_demand > labor_output * 0.8 and recent_rev > wage_bill * 0.8:
                candidates = [a for a in unemployed_agents if not a.employed]
                n_hire = min(2, len(candidates))
                for a in candidates[:n_hire]:
                    a.employed = True
                    a.employer_id = self.id
                    a.wage = self.wage_offer
                    self.employees.append(a.id)
                    hired.append(a.id)

            # Fire if deeply unprofitable
            elif recent_rev < wage_bill * 0.5 and len(self.employees) > 1:
                n_fire = min(1, len(self.employees) - 1)
                for _ in range(n_fire):
                    agent_id = self.employees.pop()
                    fired.append(agent_id)

        # Adjust wage offer based on labor market tightness
        if len(unemployed_agents) < self._cfg.num_agents * 0.1:
            self.wage_offer *= (1.0 + self._cfg.wage_adjustment_rate)
        elif len(unemployed_agents) > self._cfg.num_agents * 0.3:
            self.wage_offer *= (1.0 - self._cfg.wage_adjustment_rate)
        self.wage_offer = max(self.wage_offer, self._cfg.base_wage_min * 0.5)

        return hired, fired

    def snapshot(self) -> Dict:
        """Return a serializable snapshot of firm state."""
        return {
            "id": self.id,
            "good_produced": self.good_produced,
            "cash": self.cash,
            "price": self.price,
            "quality": self.quality,
            "inventory": self.inventory,
            "production_capacity": self.production_capacity,
            "num_employees": len(self.employees),
            "wage_offer": self.wage_offer,
            "cumulative_revenue": self.cumulative_revenue,
            "cumulative_cost": self.cumulative_cost,
            "input_cost_multiplier": self.input_cost_multiplier,
        }


# ════════════════════════════════════════════════════════════════════════
#  Market
# ════════════════════════════════════════════════════════════════════════

class Market:
    """Market-clearing mechanism: proportional rationing when demand > supply.

    For each good, aggregates demand from agents and supply from firms,
    then executes trades at posted prices. If total demand exceeds supply,
    buyers receive proportional allocations.
    """

    @staticmethod
    def clear(
        agents: List[Agent],
        firms: List[Firm],
        agent_demands: Dict[int, np.ndarray],
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> Dict:
        """Execute one round of market clearing for all goods.

        Args:
            agents: All agent objects.
            firms: All firm objects.
            agent_demands: Mapping agent_id → desired quantities per good.
            config: Simulation config.
            rng: Random generator for tie-breaking.

        Returns:
            Market data dict with per-good prices, volumes, unmet demand.
        """
        num_goods = config.num_goods
        prices = np.zeros(num_goods)
        volumes = np.zeros(num_goods)
        unmet_demand = np.zeros(num_goods)
        total_demand_per_good = np.zeros(num_goods)

        # Index firms by good
        firms_by_good: Dict[int, List[Firm]] = {}
        for f in firms:
            firms_by_good.setdefault(f.good_produced, []).append(f)

        # Compute average price per good and total supply
        supply = np.zeros(num_goods)
        for g in range(num_goods):
            good_firms = firms_by_good.get(g, [])
            if good_firms:
                prices[g] = np.mean([f.price for f in good_firms])
                supply[g] = sum(max(f.inventory, 0) for f in good_firms)

        # Aggregate demand per good
        for aid, demand_vec in agent_demands.items():
            total_demand_per_good += demand_vec

        # Clear each good
        agent_dict = {a.id: a for a in agents}
        actual_purchased = {a.id: np.zeros(num_goods) for a in agents}

        for g in range(num_goods):
            good_firms = firms_by_good.get(g, [])
            if not good_firms or total_demand_per_good[g] <= 0:
                continue

            total_supply_g = supply[g]
            total_demand_g = total_demand_per_good[g]

            # Rationing factor
            ratio = min(1.0, total_supply_g / total_demand_g) if total_demand_g > 0 else 0

            # Execute trades per agent
            remaining_supply = total_supply_g
            for aid, demand_vec in agent_demands.items():
                if demand_vec[g] <= 0:
                    continue

                agent = agent_dict[aid]
                allocated = demand_vec[g] * ratio
                allocated = min(allocated, remaining_supply)

                # Pick cheapest firm with inventory
                selling_firm = min(
                    [f for f in good_firms if f.inventory > 0],
                    key=lambda f: f.price,
                    default=None,
                )
                if selling_firm is None:
                    break

                actual_qty = min(allocated, selling_firm.inventory)
                cost = actual_qty * selling_firm.price
                if cost > agent.budget:
                    actual_qty = agent.budget / selling_firm.price
                    cost = actual_qty * selling_firm.price

                if actual_qty > 0:
                    agent.budget -= cost
                    selling_firm.record_sale(actual_qty, cost)
                    remaining_supply -= actual_qty
                    volumes[g] += actual_qty
                    actual_purchased[aid][g] = actual_qty

            unmet_demand[g] = max(0, total_demand_g - volumes[g])

        # Update demand history on firms
        for g in range(num_goods):
            for f in firms_by_good.get(g, []):
                f.demand_history.append(total_demand_per_good[g])
                f.revenue_history.append(f.cumulative_revenue)

        # Update agent memory
        for agent in agents:
            agent.update_memory(prices, actual_purchased[agent.id])

        return {
            "prices": prices,
            "volumes": volumes,
            "supply": supply,
            "total_demand": total_demand_per_good,
            "unmet_demand": unmet_demand,
        }
