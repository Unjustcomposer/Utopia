"""
NexusAI — Agent Economy Simulator
==================================
CLI entry point for running simulations, experiments, strategy searches,
and the interactive dashboard.

All results are statements about the simulation's internal dynamics
— never predictions about real companies, markets, or events.

Usage:
    python main.py run                          — single simulation
    python main.py experiment --scenario marketing  — run experiment
    python main.py search --objective profit    — strategy search
    python main.py dashboard                    — launch web dashboard
    python main.py demo                         — run all 3 example experiments
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import numpy as np

from config import SimulationConfig
from simulation import Simulation
from scenario import (
    MarketingCampaign, SupplyDisruption, DemandShock, FeatureChange,
    TradeDisruption, CompositeScenario, create_scenario,
)
from experiment import Experiment, ExperimentResult
from search import (
    StrategySearch, SearchResult,
    firm_profit_objective, total_welfare_objective, price_stability_objective,
)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a single simulation and print summary."""
    config = SimulationConfig(num_ticks=args.ticks, num_agents=args.agents)
    sim = Simulation(config=config, seed=args.seed)
    print(f"Running simulation: {config.num_agents} agents, {config.num_firms} firms, "
          f"{config.num_ticks} ticks, seed={args.seed}")
    result = sim.run()
    summary = result.summary()
    print("\n=== SIMULATION RESULT (simulated) ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 40)
    print("  All values are simulated, not real-world predictions.")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run a control/treatment experiment."""
    config = SimulationConfig(num_ticks=args.ticks, num_agents=args.agents)

    scenario_map = {
        "marketing": lambda: MarketingCampaign(
            start_tick=20, duration=40, target_good=0,
            spend=args.spend, reach=0.6, awareness_boost=0.4,
        ),
        "supply_shock": lambda: SupplyDisruption(
            start_tick=30, duration=20, target_firm=0,
            capacity_reduction=0.5, cost_increase=1.5,
        ),
        "demand_shock": lambda: DemandShock(
            start_tick=25, duration=30,
            risk_aversion_delta=0.2, savings_rate_delta=0.1,
        ),
        "trade_disruption": lambda: TradeDisruption(
            start_tick=20, duration=25, affected_goods=[0, 1],
            cost_increase=1.8, availability_reduction=0.4,
        ),
    }

    factory = scenario_map.get(args.scenario)
    if factory is None:
        print(f"Unknown scenario: {args.scenario}. Available: {list(scenario_map.keys())}")
        return

    scenario = factory()
    print(f"Running experiment: {scenario.describe()}")
    print(f"  Seeds: {args.seeds}, Config: {config.num_agents} agents, {config.num_ticks} ticks")

    exp = Experiment(config=config, scenario=scenario, num_seeds=args.seeds, base_seed=args.seed)

    def progress(done, total):
        pct = done / total * 100
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:.0f}% ({done}/{total})", end="", flush=True)

    result = exp.run(progress_callback=progress)
    print()
    print(result.print_report())


def cmd_search(args: argparse.Namespace) -> None:
    """Run a strategy search."""
    config = SimulationConfig(num_ticks=args.ticks, num_agents=args.agents)

    objectives = {
        "profit": firm_profit_objective(firm_id=0),
        "welfare": total_welfare_objective(),
        "price_stability": price_stability_objective(),
    }

    objective = objectives.get(args.objective)
    if objective is None:
        print(f"Unknown objective: {args.objective}. Available: {list(objectives.keys())}")
        return

    # Default: pricing strategy search
    def pricing_factory(params):
        return FeatureChange(
            start_tick=20, duration=60, target_good=0,
            new_price=params.get("price", 12.0),
            new_quality=params.get("quality", 1.0),
        )

    param_space = {
        "price": (args.price_min, args.price_max, args.price_step),
        "quality": [0.7, 1.0, 1.3],
    }

    print(f"Running strategy search: objective={args.objective}, method={args.method}")
    print(f"  Param space: {param_space}")

    search = StrategySearch(
        config=config,
        scenario_factory=pricing_factory,
        param_space=param_space,
        objective=objective,
        method=args.method,
        num_seeds_per_eval=args.seeds_per_eval,
        validation_num_seeds=args.validation_seeds,
        top_k_validate=args.top_k,
    )

    def progress(done, total):
        pct = done / total * 100
        print(f"\r  Evaluating candidates: {pct:.0f}% ({done}/{total})", end="", flush=True)

    result = search.run(progress_callback=progress)
    print()
    print(result.print_report())


def cmd_demo(args: argparse.Namespace) -> None:
    """Run all 3 example experiments from the README."""
    config = SimulationConfig(num_ticks=90, num_agents=150)
    seeds = 8  # fewer for demo speed

    print("=" * 78)
    print("  NexusAI DEMO: 3 Example Experiments (simulated)")
    print("=" * 78)

    # ── Experiment 1: Marketing Spend Sweep ────────────────────────────
    print("\n>> Experiment 1: Marketing Spend Sweep")
    print("  Sweeping ad spend $0-$10,000 for good 0, measuring sales uplift\n")

    def marketing_factory(params):
        return MarketingCampaign(
            start_tick=15, duration=40, target_good=0,
            spend=params["spend"], reach=params.get("reach", 0.5),
            awareness_boost=0.3 + params["spend"] / 20000,  # spend -> awareness
        )

    search1 = StrategySearch(
        config=config,
        scenario_factory=marketing_factory,
        param_space={"spend": [0, 2000, 4000, 6000, 8000, 10000]},
        objective=firm_profit_objective(firm_id=0),
        method="grid",
        num_seeds_per_eval=seeds,
        validation_num_seeds=seeds * 2,
        top_k_validate=3,
    )
    result1 = search1.run()
    print(result1.print_report())

    # ── Experiment 2: Supply Shock Response ────────────────────────────
    print("\n>> Experiment 2: Supply Shock Response")
    print("  50% capacity reduction for firm 0 over 20 ticks\n")

    scenario2 = SupplyDisruption(
        start_tick=20, duration=20, target_firm=0,
        capacity_reduction=0.5, cost_increase=1.5,
    )
    exp2 = Experiment(config=config, scenario=scenario2, num_seeds=seeds, base_seed=77)
    result2 = exp2.run()
    print(result2.print_report())

    # ── Experiment 3: Pricing Strategy Under Demand Shock ─────────────
    print("\n>> Experiment 3: Best Pricing Under a Demand Shock")
    print("  Search price x quality while agents face +0.2 risk aversion\n")

    def pricing_under_shock_factory(params):
        demand_shock = DemandShock(
            start_tick=15, duration=50,
            risk_aversion_delta=0.2, savings_rate_delta=0.05,
        )
        feature = FeatureChange(
            start_tick=15, duration=50, target_good=0,
            new_price=params["price"], new_quality=params.get("quality", 1.0),
        )
        return CompositeScenario([demand_shock, feature])

    search3 = StrategySearch(
        config=config,
        scenario_factory=pricing_under_shock_factory,
        param_space={
            "price": [8, 10, 12, 14, 16, 18, 20],
            "quality": [0.8, 1.0, 1.2],
        },
        objective=firm_profit_objective(firm_id=0),
        method="grid",
        num_seeds_per_eval=seeds,
        validation_num_seeds=seeds * 2,
        top_k_validate=3,
    )
    result3 = search3.run()
    print(result3.print_report())

    print("\n" + "=" * 78)
    print("  DEMO COMPLETE. All results are simulated, not real-world predictions.")
    print("=" * 78)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the interactive web dashboard."""
    from dashboard import start_dashboard
    start_dashboard(port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="NexusAI",
        description="Agent Economy Simulator — Scenario Sandbox & Strategy Search. "
                    "All results are simulated, not real-world predictions.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Run a single simulation")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--ticks", type=int, default=120)
    p_run.add_argument("--agents", type=int, default=200)
    p_run.set_defaults(func=cmd_run)

    # ── experiment ─────────────────────────────────────────────────────
    p_exp = subparsers.add_parser("experiment", help="Run a control/treatment experiment")
    p_exp.add_argument("--scenario", type=str, default="marketing",
                       choices=["marketing", "supply_shock", "demand_shock", "trade_disruption"])
    p_exp.add_argument("--spend", type=float, default=5000)
    p_exp.add_argument("--seeds", type=int, default=15)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--ticks", type=int, default=90)
    p_exp.add_argument("--agents", type=int, default=200)
    p_exp.set_defaults(func=cmd_experiment)

    # ── search ─────────────────────────────────────────────────────────
    p_search = subparsers.add_parser("search", help="Run a strategy search")
    p_search.add_argument("--objective", type=str, default="profit",
                          choices=["profit", "welfare", "price_stability"])
    p_search.add_argument("--method", type=str, default="grid", choices=["grid", "random"])
    p_search.add_argument("--price-min", type=float, default=8.0)
    p_search.add_argument("--price-max", type=float, default=20.0)
    p_search.add_argument("--price-step", type=float, default=2.0)
    p_search.add_argument("--seeds-per-eval", type=int, default=8)
    p_search.add_argument("--validation-seeds", type=int, default=15)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--ticks", type=int, default=90)
    p_search.add_argument("--agents", type=int, default=200)
    p_search.set_defaults(func=cmd_search)

    # ── demo ───────────────────────────────────────────────────────────
    p_demo = subparsers.add_parser("demo", help="Run all 3 example experiments")
    p_demo.set_defaults(func=cmd_demo)

    # ── dashboard ──────────────────────────────────────────────────────
    p_dash = subparsers.add_parser("dashboard", help="Launch interactive web dashboard")
    p_dash.add_argument("--port", type=int, default=8765)
    p_dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
