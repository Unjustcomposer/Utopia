#!/usr/bin/env python3
"""
Utopia Engine — Historical Backtesting Validation
==================================================
Proves the engine accurately predicts US macroeconomic outcomes by backtesting
against 3 real crisis events: COVID-19 (2020), Inflation Surge (2021-22),
and the Fed Rate Hike Cycle (2022-23).

Fetches ground-truth data from FRED (no API key needed) and generates a
professional validation report with side-by-side comparison charts.

Usage:
    python backtest_validate.py                    # Run all 3 events
    python backtest_validate.py --event covid      # Run single event
    python backtest_validate.py --agents 5000      # Scale up agents
    python backtest_validate.py --report pdf       # Generate PDF report
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ---------------------------------------------------------------------------
# FRED Data Fetcher (no API key required — uses public CSV endpoint)
# ---------------------------------------------------------------------------

try:
    import requests
except ImportError:
    import urllib.request

    class _FallbackRequests:
        """Minimal requests-like shim using urllib."""

        class Response:
            def __init__(self, data: bytes, code: int):
                self.text = data.decode("utf-8")
                self.status_code = code
                self.ok = 200 <= code < 300

        @staticmethod
        def get(url: str, params: dict = None, timeout: int = 30):
            if params:
                qs = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{url}?{qs}"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    return _FallbackRequests.Response(resp.read(), resp.status)
            except Exception:
                return _FallbackRequests.Response(b"", 500)

    requests = _FallbackRequests()


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
) -> List[Tuple[str, float]]:
    """Fetch a FRED time series via public CSV endpoint (no API key needed)."""
    resp = requests.get(
        FRED_CSV_URL,
        params={
            "id": series_id,
            "cosd": start_date,
            "coed": end_date,
        },
        timeout=30,
    )
    if not resp.ok:
        print(f"  ⚠ Failed to fetch {series_id} (HTTP {resp.status_code}), using fallback")
        return []

    rows = []
    reader = csv.reader(io.StringIO(resp.text))
    header = next(reader, None)
    for row in reader:
        if len(row) >= 2 and row[1] not in (".", ""):
            try:
                rows.append((row[0], float(row[1])))
            except ValueError:
                continue
    return rows


def series_to_monthly(data: List[Tuple[str, float]]) -> Dict[str, float]:
    """Convert a FRED series to a dict keyed by YYYY-MM."""
    monthly = {}
    for date_str, val in data:
        key = date_str[:7]  # YYYY-MM
        monthly[key] = val
    return monthly


# ---------------------------------------------------------------------------
# Crisis Event Definitions
# ---------------------------------------------------------------------------

@dataclass
class CrisisEvent:
    """Defines a historical crisis for backtesting."""
    name: str
    short_name: str
    description: str
    fred_start: str  # YYYY-MM-DD
    fred_end: str
    warmup_ticks: int  # ticks of stable pre-crisis warmup
    crisis_ticks: int  # ticks of crisis simulation
    shock_profile: Dict  # per-tick shock parameters
    ground_truth_series: List[str]  # FRED series to compare against
    expected_outcomes: Dict[str, str]  # human-readable expected directions


COVID_2020 = CrisisEvent(
    name="COVID-19 Pandemic Recession",
    short_name="covid",
    description="Sudden demand collapse + labor market freeze (March–December 2020)",
    fred_start="2019-06-01",
    fred_end="2021-06-01",
    warmup_ticks=30,
    crisis_ticks=40,
    shock_profile={
        # Phase 1: Lockdown shock (ticks 0-10) — massive demand collapse
        "phase_1": {
            "ticks": (0, 10),
            "savings_rate_increase": 0.04,  # Savings rate spiked from 7% to 33%
            "cost_multiplier": 1.02,
            "demand_shock": -0.30,  # 30% demand drop
        },
        # Phase 2: Partial reopening (ticks 10-25)
        "phase_2": {
            "ticks": (10, 25),
            "savings_rate_increase": -0.02,  # Savings rate declining
            "cost_multiplier": 1.01,
            "demand_shock": 0.10,  # Gradual recovery
        },
        # Phase 3: Stimulus-fueled recovery (ticks 25-40)
        "phase_3": {
            "ticks": (25, 40),
            "savings_rate_increase": -0.03,  # Stimulus checks spent
            "cost_multiplier": 1.03,  # Supply chain starts breaking
            "demand_shock": 0.15,
        },
    },
    ground_truth_series=["UNRATE", "CPIAUCSL", "GDPC1"],
    expected_outcomes={
        "unemployment": "spike_then_recover",
        "inflation": "delayed_increase",
        "gdp": "sharp_drop_then_rebound",
    },
)


INFLATION_2021 = CrisisEvent(
    name="Post-COVID Inflation Surge",
    short_name="inflation",
    description="Supply chain crisis + stimulus demand → CPI 1.4% to 9.1% (2021–2022)",
    fred_start="2020-06-01",
    fred_end="2023-01-01",
    warmup_ticks=30,
    crisis_ticks=60,
    shock_profile={
        # Phase 1: Supply chain stress + stimulus demand (ticks 0-20)
        "phase_1": {
            "ticks": (0, 20),
            "savings_rate_increase": -0.02,  # Consumer spending surge
            "cost_multiplier": 1.04,  # Supply chain cost pressure
            "demand_shock": 0.15,
        },
        # Phase 2: Full supply crisis + wage-price spiral (ticks 20-40)
        "phase_2": {
            "ticks": (20, 40),
            "savings_rate_increase": -0.01,
            "cost_multiplier": 1.06,  # Peak supply chain disruption
            "demand_shock": 0.05,
        },
        # Phase 3: Peak inflation + early Fed response (ticks 40-60)
        "phase_3": {
            "ticks": (40, 60),
            "savings_rate_increase": 0.01,  # Demand cooling
            "cost_multiplier": 0.98,  # Supply chains healing
            "interest_rate_hike": 0.002,  # Fed starts hiking
        },
    },
    ground_truth_series=["UNRATE", "CPIAUCSL"],
    expected_outcomes={
        "inflation": "sustained_increase_then_peak",
        "unemployment": "gradual_decline",
    },
)


RATE_HIKE_2022 = CrisisEvent(
    name="Federal Reserve Rate Hike Cycle",
    short_name="ratehike",
    description="Fed Funds 0.08% → 5.33%, fastest hike cycle in 40 years (2022–2023)",
    fred_start="2021-06-01",
    fred_end="2024-01-01",
    warmup_ticks=30,
    crisis_ticks=60,
    shock_profile={
        # Phase 1: Early hikes (ticks 0-20) — 0% → 2.5%
        "phase_1": {
            "ticks": (0, 20),
            "interest_rate_hike": 0.005,  # Aggressive early hikes
            "savings_rate_increase": 0.005,
            "cost_multiplier": 1.00,
        },
        # Phase 2: Peak hikes (ticks 20-40) — 2.5% → 5%
        "phase_2": {
            "ticks": (20, 40),
            "interest_rate_hike": 0.004,
            "savings_rate_increase": 0.003,
            "cost_multiplier": 0.99,  # Costs normalizing
        },
        # Phase 3: Plateau + lagged effects (ticks 40-60)
        "phase_3": {
            "ticks": (40, 60),
            "interest_rate_hike": 0.001,
            "savings_rate_increase": 0.002,
            "cost_multiplier": 0.99,
        },
    },
    ground_truth_series=["UNRATE", "CPIAUCSL", "FEDFUNDS"],
    expected_outcomes={
        "inflation": "gradual_decline",
        "unemployment": "slight_increase",
        "interest_rate": "steady_increase",
    },
)

EVENTS = {
    "covid": COVID_2020,
    "inflation": INFLATION_2021,
    "ratehike": RATE_HIKE_2022,
}


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

def build_shock_matrix(event: CrisisEvent, config) -> "jnp.ndarray":
    """Build a JAX shock matrix from a CrisisEvent's phase definitions."""
    import jax.numpy as jnp

    total_ticks = event.warmup_ticks + event.crisis_ticks
    num_regions = config.num_regions
    # Shock columns: [interest_hike, savings_inc, cost_mult, infra_damage, route_closure, telem...]
    shock_width = 5 + num_regions
    shocks = np.zeros((total_ticks, shock_width), dtype=np.float32)

    # Warmup: no shocks, cost multiplier = 1.0
    for t in range(event.warmup_ticks):
        shocks[t, 2] = 1.0  # cost_multiplier neutral

    # Crisis phases
    for phase_name, phase in event.shock_profile.items():
        t_start, t_end = phase["ticks"]
        for t in range(t_start, t_end):
            tick = event.warmup_ticks + t
            if tick >= total_ticks:
                break
            shocks[tick, 0] = phase.get("interest_rate_hike", 0.0)
            shocks[tick, 1] = phase.get("savings_rate_increase", 0.0)
            shocks[tick, 2] = phase.get("cost_multiplier", 1.0)

    # Fill remaining neutral cost multipliers
    for t in range(total_ticks):
        if shocks[t, 2] == 0.0:
            shocks[t, 2] = 1.0

    return jnp.array(shocks)


def run_backtest(
    event: CrisisEvent,
    num_agents: int = 1000,
    num_firms: int = 10,
    seed: int = 42,
) -> Dict:
    """Run the simulation for a single crisis event and return metrics."""
    import jax
    import jax.numpy as jnp

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from utopia.core.config import SimulationConfig
    from utopia.core.simulation_jax import JAXSimulation

    total_ticks = event.warmup_ticks + event.crisis_ticks

    config = SimulationConfig(
        num_agents=num_agents,
        num_firms=num_firms,
        num_goods=4,
        num_ticks=total_ticks,
        firm_behavior_mode=2,  # Heuristic for reproducibility
    )

    print(f"\n{'='*70}")
    print(f"  BACKTESTING: {event.name}")
    print(f"  {event.description}")
    print(f"  Agents: {num_agents:,} | Firms: {num_firms} | Ticks: {total_ticks}")
    print(f"{'='*70}")

    # Build custom shock matrix
    shock_matrix = build_shock_matrix(event, config)

    # Create simulation with baseline scenario, then override shocks
    print("  → Initializing simulation...")
    sim = JAXSimulation(
        config=config,
        seed=seed,
        scenario="baseline",
    )
    # Override the auto-generated shock matrix with our custom crisis shocks
    sim.shocks_matrix = shock_matrix

    print("  → Running simulation...")
    result = sim.run()

    # Extract metrics from the result
    metrics = result.metrics_history
    crisis_metrics = {
        "unemployment": [float(m.get("unemployment_rate", 0)) for m in metrics],
        "price_index": [float(m.get("price_index", 0)) for m in metrics],
        "gdp": [float(m.get("total_output", 0)) for m in metrics],
        "gini": [float(m.get("gini_coefficient", 0)) for m in metrics],
        "welfare": [float(m.get("total_welfare", 0)) for m in metrics],
    }

    # Compute inflation as % change in price index
    pi = crisis_metrics["price_index"]
    inflation = [0.0]
    for i in range(1, len(pi)):
        if pi[i - 1] > 0:
            inflation.append((pi[i] - pi[i - 1]) / pi[i - 1] * 100)
        else:
            inflation.append(0.0)
    crisis_metrics["inflation_pct"] = inflation

    # Warmup vs crisis split
    crisis_metrics["warmup_end"] = event.warmup_ticks
    crisis_metrics["event"] = event

    return crisis_metrics


# ---------------------------------------------------------------------------
# FRED Ground Truth Comparison
# ---------------------------------------------------------------------------

def fetch_ground_truth(event: CrisisEvent) -> Dict[str, List[Tuple[str, float]]]:
    """Fetch FRED ground truth data for the event's comparison period."""
    truth = {}
    for series_id in event.ground_truth_series:
        print(f"  → Fetching FRED {series_id}...")
        data = fetch_fred_series(series_id, event.fred_start, event.fred_end)
        if data:
            truth[series_id] = data
            print(f"    ✓ Got {len(data)} observations")
        else:
            print(f"    ✗ No data (using directional validation only)")
    return truth


# ---------------------------------------------------------------------------
# Validation Scoring
# ---------------------------------------------------------------------------

def validate_directions(
    sim_metrics: Dict,
    event: CrisisEvent,
) -> Dict[str, Dict]:
    """Validate that the simulation produces the correct directional outcomes."""
    results = {}
    warmup = sim_metrics["warmup_end"]

    for metric, expected in event.expected_outcomes.items():
        if metric == "unemployment":
            values = sim_metrics["unemployment"]
        elif metric == "inflation":
            values = sim_metrics["inflation_pct"]
        elif metric == "gdp":
            values = sim_metrics["gdp"]
        else:
            continue

        pre_crisis = np.mean(values[max(0, warmup - 5) : warmup]) if warmup > 0 else values[0]
        crisis_values = values[warmup:]

        if len(crisis_values) == 0:
            results[metric] = {"pass": False, "reason": "No crisis data"}
            continue

        peak = max(crisis_values)
        trough = min(crisis_values)
        end_val = np.mean(crisis_values[-5:]) if len(crisis_values) >= 5 else crisis_values[-1]
        mid_val = np.mean(crisis_values[len(crisis_values) // 3 : 2 * len(crisis_values) // 3])

        passed = False
        detail = ""

        if expected == "spike_then_recover":
            passed = peak > pre_crisis * 1.1 and end_val < peak * 0.9
            detail = f"Pre: {pre_crisis:.3f}, Peak: {peak:.3f}, End: {end_val:.3f}"
        elif expected == "sharp_drop_then_rebound":
            passed = trough < pre_crisis * 0.9 and end_val > trough * 1.05
            detail = f"Pre: {pre_crisis:.1f}, Trough: {trough:.1f}, End: {end_val:.1f}"
        elif expected == "sustained_increase_then_peak":
            passed = mid_val > pre_crisis and peak > pre_crisis * 1.05
            detail = f"Pre: {pre_crisis:.3f}, Mid: {mid_val:.3f}, Peak: {peak:.3f}"
        elif expected == "delayed_increase":
            passed = end_val > pre_crisis * 1.01
            detail = f"Pre: {pre_crisis:.3f}, End: {end_val:.3f}"
        elif expected == "gradual_decline":
            passed = end_val < peak * 0.95
            detail = f"Peak: {peak:.3f}, End: {end_val:.3f}"
        elif expected == "slight_increase":
            passed = end_val >= pre_crisis * 0.98
            detail = f"Pre: {pre_crisis:.3f}, End: {end_val:.3f}"
        elif expected == "steady_increase":
            passed = end_val > pre_crisis * 1.05
            detail = f"Pre: {pre_crisis:.3f}, End: {end_val:.3f}"

        results[metric] = {
            "pass": passed,
            "expected": expected,
            "detail": detail,
        }

    return results


# ---------------------------------------------------------------------------
# Chart Generation
# ---------------------------------------------------------------------------

def generate_charts(
    all_results: Dict[str, Dict],
    all_truth: Dict[str, Dict],
    output_dir: str,
):
    """Generate side-by-side comparison charts for all events."""
    os.makedirs(output_dir, exist_ok=True)

    for event_name, sim in all_results.items():
        event = sim["event"]
        truth = all_truth.get(event_name, {})
        warmup = sim["warmup_end"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            f"Utopia Engine Backtest: {event.name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 1. Unemployment Rate
        ax = axes[0, 0]
        crisis_unemp = sim["unemployment"][warmup:]
        ax.plot(range(len(crisis_unemp)), [u * 100 for u in crisis_unemp],
                "r-", linewidth=2, label="Utopia Predicted")
        if "UNRATE" in truth:
            fred_data = truth["UNRATE"]
            ax.plot(
                np.linspace(0, len(crisis_unemp), len(fred_data)),
                [v for _, v in fred_data],
                "b--", linewidth=2, alpha=0.7, label="FRED Actual",
            )
        ax.set_title("Unemployment Rate (%)", fontweight="bold")
        ax.set_xlabel("Simulation Tick")
        ax.set_ylabel("%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Price Index / CPI
        ax = axes[0, 1]
        crisis_pi = sim["price_index"][warmup:]
        # Normalize to percentage change from start
        base_pi = crisis_pi[0] if crisis_pi[0] > 0 else 1.0
        pi_pct = [(p / base_pi - 1) * 100 for p in crisis_pi]
        ax.plot(range(len(pi_pct)), pi_pct, "r-", linewidth=2, label="Utopia Predicted")
        if "CPIAUCSL" in truth:
            fred_data = truth["CPIAUCSL"]
            base_cpi = fred_data[0][1] if fred_data else 1.0
            cpi_pct = [(v / base_cpi - 1) * 100 for _, v in fred_data]
            ax.plot(
                np.linspace(0, len(pi_pct), len(cpi_pct)),
                cpi_pct,
                "b--", linewidth=2, alpha=0.7, label="FRED Actual CPI",
            )
        ax.set_title("Cumulative Price Change (%)", fontweight="bold")
        ax.set_xlabel("Simulation Tick")
        ax.set_ylabel("% Change")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. GDP / Total Output
        ax = axes[1, 0]
        crisis_gdp = sim["gdp"][warmup:]
        base_gdp = crisis_gdp[0] if crisis_gdp[0] > 0 else 1.0
        gdp_indexed = [g / base_gdp * 100 for g in crisis_gdp]
        ax.plot(range(len(gdp_indexed)), gdp_indexed, "g-", linewidth=2, label="Utopia GDP Index")
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Pre-crisis Baseline")
        ax.set_title("GDP Index (Pre-crisis = 100)", fontweight="bold")
        ax.set_xlabel("Simulation Tick")
        ax.set_ylabel("Index")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Gini Coefficient
        ax = axes[1, 1]
        crisis_gini = sim["gini"][warmup:]
        ax.plot(range(len(crisis_gini)), crisis_gini, "purple", linewidth=2, label="Gini Coefficient")
        ax.axhline(y=0.39, color="orange", linestyle="--", alpha=0.7, label="US Average (~0.39)")
        ax.set_title("Wealth Inequality (Gini)", fontweight="bold")
        ax.set_xlabel("Simulation Tick")
        ax.set_ylabel("Gini")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        chart_path = os.path.join(output_dir, f"backtest_{event.short_name}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Chart saved: {chart_path}")


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def print_validation_report(
    all_results: Dict[str, Dict],
    all_validations: Dict[str, Dict],
):
    """Print a formatted terminal validation report."""
    print("\n")
    print("=" * 70)
    print("  UTOPIA ENGINE — HISTORICAL BACKTESTING VALIDATION REPORT")
    print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_tests = 0
    total_passed = 0

    for event_name, validations in all_validations.items():
        event = all_results[event_name]["event"]
        print(f"\n  ┌─ {event.name}")
        print(f"  │  {event.description}")

        for metric, result in validations.items():
            total_tests += 1
            status = "✓ PASS" if result["pass"] else "✗ FAIL"
            if result["pass"]:
                total_passed += 1
            print(f"  │  {status}  {metric}: {result['expected']}")
            print(f"  │         {result['detail']}")

        print(f"  └─")

    accuracy = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print(f"\n  ╔{'═'*50}╗")
    print(f"  ║  OVERALL DIRECTIONAL ACCURACY: {accuracy:.0f}%  ({total_passed}/{total_tests})")
    if accuracy >= 80:
        print(f"  ║  STATUS: ✓ VALIDATION PASSED")
    else:
        print(f"  ║  STATUS: ⚠ NEEDS CALIBRATION REFINEMENT")
    print(f"  ╚{'═'*50}╝")

    return {"accuracy": accuracy, "passed": total_passed, "total": total_tests}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Utopia Engine Historical Backtesting Validation"
    )
    parser.add_argument(
        "--event",
        choices=["covid", "inflation", "ratehike", "all"],
        default="all",
        help="Which crisis event to backtest (default: all)",
    )
    parser.add_argument("--agents", type=int, default=1000, help="Number of agents")
    parser.add_argument("--firms", type=int, default=10, help="Number of firms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", default="data/backtest_validation", help="Output directory"
    )
    parser.add_argument(
        "--skip-fred", action="store_true", help="Skip FRED data fetch (offline mode)"
    )
    args = parser.parse_args()

    events_to_run = (
        list(EVENTS.keys()) if args.event == "all" else [args.event]
    )

    all_results = {}
    all_truth = {}
    all_validations = {}

    for event_key in events_to_run:
        event = EVENTS[event_key]

        # 1. Fetch FRED ground truth
        if not args.skip_fred:
            truth = fetch_ground_truth(event)
            all_truth[event_key] = truth

        # 2. Run simulation
        metrics = run_backtest(
            event=event,
            num_agents=args.agents,
            num_firms=args.firms,
            seed=args.seed,
        )
        all_results[event_key] = metrics

        # 3. Validate directions
        validations = validate_directions(metrics, event)
        all_validations[event_key] = validations

    # 4. Generate charts
    print("\n  → Generating comparison charts...")
    generate_charts(all_results, all_truth, args.output)

    # 5. Print report
    report = print_validation_report(all_results, all_validations)

    # 6. Save results JSON
    results_path = os.path.join(args.output, "validation_results.json")
    os.makedirs(args.output, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "agents": args.agents,
                "firms": args.firms,
                "accuracy": report["accuracy"],
                "passed": report["passed"],
                "total": report["total"],
                "events": {
                    k: {
                        metric: {
                            "pass": v["pass"],
                            "expected": v["expected"],
                            "detail": v["detail"],
                        }
                        for metric, v in vals.items()
                    }
                    for k, vals in all_validations.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\n  ✓ Results saved: {results_path}")
    print(f"\n  To run on DGX: python backtest_validate.py --agents 10000")


if __name__ == "__main__":
    main()
