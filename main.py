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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()
