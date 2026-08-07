# NexusAI: Automated Tariff Impact & Supply Chain Digital Twin

NexusAI is a gradient-based (JAX/autodiff) macroeconomic simulation engine. It is empirically calibrated and validated against FRED 2008 data, featuring quantified error tracking against actual historical GDP and unemployment figures, alongside a from-scratch performance benchmark demonstrating massive speedups over traditional object-oriented frameworks like Mesa.

It features SAP/Oracle-shaped connector interfaces with structurally correct auth and pagination, validated against mock responses, to demonstrate how such an engine could ingest supply chain data.

*See our [Contributors](TEAM.md).*

> **⚠️ Every result produced by this simulator is a statement about the
> simulation's internal dynamics — never a prediction about a real company,
> market, or geopolitical event. This is a decision-support and portfolio-
> demonstration tool, not a trading system.**

---

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
│  config.py  │────▶│ engine_jax.py  │────▶│ simulation_jax.py│
│  Parameters │     │ JAX Simulation │     │    Main Loop     │
│             │     │      Core      │     │    + Metrics     │
└─────────────┘     └────────────────┘     └─────────┬────────┘
                                                     │
                                           ┌─────────▼────────┐
                                           │    server.py     │
                                           │   FastAPI Node   │
                                           └──────────────────┘
```

## Performance

| Engine | 100K Agents × 50 Ticks | Speedup |
|--------|----------------------|---------|
| Pure Python (Mesa-style) | ~180s | 1× |
| NexusAI JAX Engine | ~0.3s | **~600×** |

> Benchmark: `python benchmark_mesa_vs_jax.py` on a single CPU. 
> JAX execution excludes one-time XLA compilation (~45s).
> Numbers are approximate; run the benchmark script for your hardware.
## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single simulation
python main.py run --seed 42 --ticks 120

# Train the Large Macroeconomic Model
python main.py train --seed 42 --epochs 100 --ticks 50

# Run an interactive demo
python main.py demo --seed 42 --ticks 30

# Run A/B testing between policy scenarios (e.g., baseline vs tariffs)
python main.py experiment --scenario-a baseline --scenario-b tariffs --ticks 120

# Run seed robustness checks
python main.py search --num-seeds 5 --ticks 120

# Run the API server & Dashboard UI
uvicorn server:app --reload
```

---

## Components

### `config.py` — Simulation Parameters
Central `SimulationConfig` dataclass with ~30 parameters covering population
(agents, wages, savings), firms (capacity, pricing, production), market
mechanics (elasticity, awareness, memory), and experiment defaults.

### `engine_jax.py` — Agent, Firm, Market (JAX Core)
- **Agent**: Cobb-Douglas utility maximization gated by per-good awareness.
  Agents save a fraction of income (modulated by risk aversion), then spend
  the remainder across visible goods. A sliding memory window tracks past
  prices for price-elasticity adjustments.
- **Firm**: Produces one good, hires/fires agents, and adaptively prices
  based on inventory vs. target buffer. Input cost multiplier models supply
  disruptions.
- **Market**: Proportional rationing when demand exceeds supply; cheapest-
  firm-first allocation.

### `server.py` — API Server
FastAPI server exposing asynchronous endpoints to run simulations and integrate with external systems. It also serves the frontend UI.

### `dashboard_ui.py` — Web Dashboard
A modern, dark-themed interactive web dashboard built with HTML/CSS and Chart.js. It integrates seamlessly with the FastAPI backend, allowing users to run simulations, view real-time metrics (like Gini coefficient and Unemployment rate), and explain Firm pricing policies via the LMM.

### `simulation_jax.py` — Tick Loop
Phased execution per tick:
1. Firms produce → 2. Pay wages → 3. Agents demand
→ 4. Market clears → 5. Firms adjust → 6. Memory update → 7. Metrics

---

## Guardrails

1. **Simulation only.** Every result is about the simulation's internal
   dynamics. No predictions about real companies, markets, or events.
2. **No narrative violence.** Disasters, wars, and conflicts are implemented
   purely as numeric shocks to supply, demand, or trade cost parameters.
3. **No alpha claims.** This is a decision-support tool, not a trading
   system. 

---

## Dependencies

- Python 3.9+
- JAX
- FastAPI
- NumPy ≥ 1.24
- SciPy ≥ 1.10

## License

MIT

## Phase 2.1 UI - React Frontend
To build the frontend, run:
```bash
cd frontend
npm install
npm run build
```
The FastAPI server (`server.py`) will automatically serve the built UI from `frontend/dist` at the root path (`/`).
**Warning**: `dashboard_ui.py` is deprecated. Use the React Frontend instead.
