"""
NexusAI — Agent Economy Simulator
==================================
CLI entry point for running pure JAX simulations and LMM training.
"""

import argparse
from config import SimulationConfig
from simulation_jax import run_simulation
from train_rl import train_lmm

def cmd_run(args: argparse.Namespace) -> None:
    """Run a single pure JAX simulation and print summary."""
    config = SimulationConfig(num_ticks=args.ticks, num_agents=args.agents)
    print(f"Running simulation: {config.num_agents} agents, {config.num_firms} firms, "
          f"{config.num_ticks} ticks, seed={args.seed}")
    
    result = run_simulation(config=config, seed=args.seed)
    
    print("\n=== SIMULATION METRICS (simulated) ===")
    if result.metrics_history:
        final_metrics = result.metrics_history[-1]
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    print("=" * 40)

def cmd_train(args: argparse.Namespace) -> None:
    """Train the Large Macroeconomic Model (LMM)."""
    print(f"Training LMM for {args.epochs} epochs, with {args.ticks} ticks per episode, seed={args.seed}")
    train_lmm(seed=args.seed, epochs=args.epochs, num_ticks=args.ticks)

def cmd_demo(args: argparse.Namespace) -> None:
    """Run a quick visual demo of the simulator."""
    print("Running Interactive Demo Mode...")
    config = SimulationConfig(num_ticks=args.ticks, num_agents=args.agents)
    result = run_simulation(config=config, seed=args.seed)
    final = result.metrics_history[-1] if result.metrics_history else {}
    print(f"\n[DEMO COMPLETE] Output: {final.get('total_output', 0):.2f} | Gini: {final.get('gini', 0):.4f}")

def cmd_experiment(args: argparse.Namespace) -> None:
    """Run A/B testing between two scenarios."""
    import numpy as np
    print(f"Running A/B Test: {args.scenario_a} vs {args.scenario_b}")
    config = SimulationConfig(num_ticks=args.ticks)
    
    print(f"--> Simulating Scenario A ({args.scenario_a})")
    res_a = run_simulation(config=config, seed=args.seed, scenario=args.scenario_a)
    out_a = res_a.metrics_history[-1].get('total_output', 0) if res_a.metrics_history else 0
    
    print(f"--> Simulating Scenario B ({args.scenario_b})")
    res_b = run_simulation(config=config, seed=args.seed, scenario=args.scenario_b)
    out_b = res_b.metrics_history[-1].get('total_output', 0) if res_b.metrics_history else 0
    
    diff = out_b - out_a
    pct = (diff / out_a * 100) if out_a else 0
    print(f"\n[A/B TEST RESULT] Output diff: {diff:+.2f} ({pct:+.2f}%)")

def cmd_search(args: argparse.Namespace) -> None:
    """Run seed robustness checks across multiple initializations."""
    import numpy as np
    print(f"Running Seed Robustness Check ({args.num_seeds} seeds)")
    config = SimulationConfig(num_ticks=args.ticks)
    outputs = []
    
    for i in range(args.num_seeds):
        seed = args.seed + i
        res = run_simulation(config=config, seed=seed)
        if res.metrics_history:
            outputs.append(res.metrics_history[-1].get('total_output', 0))
            
    if outputs:
        mean_out = np.mean(outputs)
        std_out = np.std(outputs)
        print(f"\n[ROBUSTNESS] Mean Output: {mean_out:.2f} ± {std_out:.2f} (across {args.num_seeds} seeds)")
    else:
        print("\n[ROBUSTNESS] No output generated.")

def main():
    parser = argparse.ArgumentParser(
        prog="NexusAI",
        description="Agent Economy Simulator — Pure JAX LMM Training & Simulation.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Run a single simulation")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--ticks", type=int, default=120)
    p_run.add_argument("--agents", type=int, default=1000)
    p_run.set_defaults(func=cmd_run)

    # ── train ──────────────────────────────────────────────────────────
    p_train = subparsers.add_parser("train", help="Train the Large Macroeconomic Model")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--ticks", type=int, default=50)
    p_train.set_defaults(func=cmd_train)

    # ── demo ───────────────────────────────────────────────────────────
    p_demo = subparsers.add_parser("demo", help="Run an interactive demo")
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--ticks", type=int, default=30)
    p_demo.add_argument("--agents", type=int, default=500)
    p_demo.set_defaults(func=cmd_demo)

    # ── experiment ─────────────────────────────────────────────────────
    p_exp = subparsers.add_parser("experiment", help="Run A/B testing between scenarios")
    p_exp.add_argument("--scenario-a", type=str, default="baseline")
    p_exp.add_argument("--scenario-b", type=str, default="tariffs")
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--ticks", type=int, default=120)
    p_exp.set_defaults(func=cmd_experiment)

    # ── search ─────────────────────────────────────────────────────────
    p_search = subparsers.add_parser("search", help="Run seed robustness checks")
    p_search.add_argument("--num-seeds", type=int, default=5)
    p_search.add_argument("--seed", type=int, default=42)
    p_search.add_argument("--ticks", type=int, default=120)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()
