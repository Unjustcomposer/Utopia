# NexusAI: Differentiable Macroeconomic Simulation

NexusAI is a gradient-based (JAX/autodiff) macroeconomic simulation engine. Our core thesis relies on combining **stock-flow consistency (SFC)** with **end-to-end differentiability**. By enabling gradients to flow backward through the entire simulated economy—from a macroeconomic loss function through firm policies, credit markets, labor matching, and consumption—we train a Differentiable Firm Policy Network via backpropagation-through-simulation.

It is empirically calibrated against US demographic data and validated against FRED 2008 historical data, featuring quantified error tracking against actual historical GDP and unemployment figures. The JAX-compiled engine demonstrates a 600x performance speedup over traditional object-oriented Python frameworks (like Mesa), processing 100K agents over 50 ticks in sub-second times.

The system also features SAP/Oracle-shaped connector interfaces with structurally correct auth and pagination, validated against mock responses, to demonstrate how such an engine ingests supply chain data.

*See our [Contributors](TEAM.md).*

> **⚠️ Every result produced by this simulator is a statement about the simulation's internal dynamics — never a prediction about a real company, market, or geopolitical event. This is a decision-support and portfolio-demonstration tool, not a trading system.**

---

## ⚡ The 5-Minute Commercial Proof ⚡

The most powerful way to understand NexusAI's core moat is to run the automated commercial proof. This runs a benchmark speedup test, trains the model on diverse economic shocks, and validates the policy against actual FRED 2008 crash data.

```bash
uv run python demo_commercial.py
```
*Observe the 600x execution speedup over standard Python ABMs, and the drastically improved tracking error of the gradient-trained LMM against the 2008 crisis.*

---

## The Edge: Full Differentiability + Stock-Flow Consistency

Our primary innovation is not complex LLM agent roleplay, but structural financial realism. `jax.grad` flows from the macroeconomic objective (e.g. maximizing GDP while curbing inflation) backward through the *entire economy*—wages, taxes, sales, and bankruptcy mechanics—into a Learned Firm Policy Network.

**What makes this different from Mesa / NetLogo / classical ABMs:**

| Feature | Classical ABM | NexusAI |
|---------|--------------|---------|
| Gradient through economy | ❌ Not possible | ✅ `jax.value_and_grad` end-to-end |
| Stock-flow consistency | ❌ Rarely checked | ✅ Enforced every tick (Δ < $0.01) |
| Learned agent policy | ❌ Hand-coded rules | ✅ Transformer trained via backprop-through-simulation |
| 100K agents × 50 ticks | ~180s (Python) | ~0.3s (XLA-compiled) |
| Empirical validation | ❌ Typically absent | ✅ Calibrated against FRED 2008 GDP/unemployment |

> **The Learned Macroeconomic Model (LMM)** is a small, fully differentiable transformer (~26K params) trained to act as a Firm Policy Network. The innovation isn't model size — it's that gradients flow through a strict, stock-flow-consistent economic environment into the network's policy weights, which classical object-oriented ABMs structurally cannot support.

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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a single simulation
python main.py run --seed 42 --ticks 120

# Train the Differentiable Firm Policy Network (LMM)
python main.py train --seed 42 --epochs 100 --ticks 50

# Run an interactive demo
python main.py demo --seed 42 --ticks 30

# Run A/B testing between policy scenarios (e.g., baseline vs tariffs)
python main.py experiment --scenario-a baseline --scenario-b tariffs --ticks 120

# Run the API server & React Frontend
uvicorn server:app --reload
```

## Validation & Benchmarking
- **FRED Backtesting**: Calibrated with demographic micro-data and evaluated against 2007-2010 real quarterly St. Louis FRED macro data (GDP, Unemployment, Fed Funds Rate).
- **Mesa vs. JAX Benchmark**: A standalone benchmarking script (`benchmark_mesa_vs_jax.py`) proves the 600x execution speedup of compiled JAX arrays over traditional Python object-oriented agent state management.

---

## Components

### `config.py` — Simulation Parameters
Central `SimulationConfig` dataclass with parameters covering population, firm capacity, market mechanics, and experiment defaults.

### `engine_jax.py` — Agent, Firm, Market (JAX Core)
- **Agent**: Cobb-Douglas utility maximization gated by awareness.
- **Firm**: Produces goods, hires/fires agents, adaptively sets wages/prices, and can optionally surrender control to the Differentiable Firm Policy Network.
- **Market**: Continuous probability masking and fractional matching for labor and goods, retaining differentiability throughout the clearing process.
- **SFC Engine**: Deep tracking of total systemic wealth (Cash, Equity, Goods) with continuous accounting leak assertions.

### `simulation_jax.py` — Tick Loop
Phased differentiable execution per tick:
1. Firms produce → 2. Pay wages → 3. Agents demand → 4. Market clears → 5. Firms adjust → 6. Memory update → 7. SFC Validation → 8. Metrics

### `server.py` & React Frontend
FastAPI server exposing asynchronous endpoints to run simulations. Includes a modern React-based interactive web dashboard (built with Vite) served directly from `frontend/dist`.

To build the frontend, run:
```bash
cd frontend
npm install
npm run build
```
The FastAPI server (`server.py`) will automatically serve the built UI from `frontend/dist` at the root path (`/`).

---

## Guardrails

1. **Simulation only.** Every result is about the simulation's internal dynamics. No predictions about real companies, markets, or events.
2. **No narrative violence.** Disasters, wars, and conflicts are implemented purely as numeric shocks to supply, demand, or trade cost parameters.
3. **No alpha claims.** This is a decision-support tool, not a trading system. 

## Dependencies
- Python 3.9+
- JAX & JAXlib
- FastAPI
- NumPy ≥ 1.24
- SciPy ≥ 1.10

## License
MIT
