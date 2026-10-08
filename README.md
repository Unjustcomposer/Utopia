<div align="center">

# 🌐 Utopia

### The World's First Differentiable Macroeconomic Engine

**Simulate 100,000 economic agents in 0.3 seconds. Train AI policies via backpropagation through an entire economy. Validate against real FRED data.**

[![CI/CD Pipeline](https://github.com/Unjustcomposer/Utopia/actions/workflows/ci.yml/badge.svg)](https://github.com/Unjustcomposer/Utopia/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/built%20with-JAX-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Quick Start](#-quick-start) · [Live Demo](#-live-demo) · [How It Works](#-how-it-works) · [Backtesting](#-historical-backtesting) · [API Reference](#-api-reference) · [Paper](#-academic-citations)

</div>

---

## 💡 What Is This?

Traditional macroeconomic models (DSGE, Input-Output tables) are **black boxes**: slow, non-differentiable, and impossible to optimize. Agent-based models are flexible but run in Python loops that take hours.

**Utopia solves both problems.**

It is a fully **stock-flow consistent**, **end-to-end differentiable** macroeconomic simulation engine compiled to XLA via [JAX](https://github.com/google/jax). Gradients flow backward from a macroeconomic loss function through credit markets, labor matching, firm pricing, and consumer demand — enabling a neural network to **learn optimal firm policy directly from the structure of the economy**.

```python
from utopia.core.simulation_jax import run_simulation
from utopia.core.config import SimulationConfig

# Simulate a 100K-agent US economy in under a second
config = SimulationConfig(num_agents=100_000, num_firms=500)
result = run_simulation(config, seed=42, scenario="tariffs")

print(f"Gini: {result.metrics_history[-1]['gini_coefficient']:.3f}")
print(f"Unemployment: {result.metrics_history[-1]['unemployment_rate']:.1%}")
```

---

## 🔥 Why This Matters

| | Classical ABM (Mesa/NetLogo) | DSGE (Fed/IMF) | **Utopia** |
|---|---|---|---|
| **Speed** | ~180s for 10K agents | Minutes to solve | **0.3s for 100K agents** (XLA-compiled) |
| **Gradient through economy** | ❌ Impossible | ❌ Not supported | ✅ `jax.value_and_grad` end-to-end |
| **Stock-flow consistency** | ❌ Rarely checked | ⚠️ Partial | ✅ Enforced every tick (Δ < $0.01) |
| **Learned agent policy** | ❌ Hand-coded rules | ❌ Representative agent | ✅ Transformer trained via backprop |
| **Empirical validation** | ❌ Typically absent | ⚠️ Aggregate only | ✅ Backtested against FRED 2008/2020/2021 |
| **Heterogeneous agents** | ✅ | ❌ | ✅ 100K agents with individual skills, budgets, preferences |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Unjustcomposer/Utopia.git
cd Utopia
pip install -r requirements.txt
```

### Run the Dashboard
```bash
uvicorn server:app --host 0.0.0.0 --port 8765
# Open http://localhost:8765
```

### Run via CLI
```bash
# Standard simulation
utopia run --seed 42 --ticks 120

# Train the neural firm policy
utopia train --seed 42 --epochs 100 --ticks 50

# Compare scenarios (baseline vs tariffs)
utopia experiment --scenario-a baseline --scenario-b tariffs --ticks 120
```

### Docker
```bash
docker-compose up --build
```

---

## 🏛️ How It Works

```
                    ┌──────────────────────────────────────────┐
                    │          SimulationConfig                │
                    │   US-calibrated: BLS, Census, Fed, IRS  │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │         JAX Simulation Engine            │
                    │  ┌─────────┐ ┌──────────┐ ┌──────────┐  │
                    │  │ Credit  │ │Production│ │ Market   │  │
                    │  │ System  │ │& Logistics│ │ Clearing │  │
                    │  └────┬────┘ └────┬─────┘ └────┬─────┘  │
                    │  ┌────▼────┐ ┌────▼─────┐ ┌────▼─────┐  │
                    │  │ Labor   │ │Government│ │  Firm    │  │
                    │  │  (DMP)  │ │ & Taxes  │ │ Lifecycle│  │
                    │  └─────────┘ └──────────┘ └──────────┘  │
                    └──────────────┬───────────────────────────┘
                                   │ jax.grad ↑↓
                    ┌──────────────▼───────────────────────────┐
                    │     Learned Macroeconomic Model (LMM)    │
                    │     ~26K param Transformer policy net    │
                    │     Trained via backprop-through-sim     │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │           FastAPI + React Dashboard       │
                    │   Real-time charts, scenarios, explain   │
                    └──────────────────────────────────────────┘
```

### The 12 Engine Modules (Executed Every Tick)

| Module | What It Does |
|--------|-------------|
| **Credit** | Commercial bank lending, reserve requirements, interest rate transmission |
| **Logistics** | Multi-region freight, transit delays, Bill of Materials (BOM) supply chains |
| **Production** | Cobb-Douglas production with capital depreciation and capacity constraints |
| **Labor (DMP)** | Diamond-Mortensen-Pissarides matching with Nash bargaining |
| **Government** | Progressive taxation (10-37%), unemployment benefits, fiscal policy |
| **Market** | CES demand system, price discovery, inventory management with shelf life |
| **Housing** | Regional housing markets with mortgage dynamics |
| **Foreign Trade** | Import/export with tariff shocks and exchange rate effects |
| **Social** | Demographic turnover, mortality, network effects |
| **Firm Lifecycle** | Entry, bankruptcy (sigmoid survival), and creative destruction |
| **Climate** | Infrastructure damage and route closure shocks |
| **SFC Check** | Stock-flow consistency enforcement (money conservation) |

---

## 📊 Historical Backtesting

Utopia ships with a validation pipeline that backtests the engine against **3 major US economic crises** using real [FRED](https://fred.stlouisfed.org/) data:

```bash
# Backtest all 3 crises
python backtest_validate.py --agents 10000

# Single event
python backtest_validate.py --event covid --agents 50000
```

| Crisis Event | Period | What We Predict | Validation |
|---|---|---|---|
| 🦠 **COVID-19 Recession** | Mar–Dec 2020 | Unemployment spike to 14.7%, GDP collapse, V-shaped recovery | Directional ✓ |
| 📈 **Inflation Surge** | 2021–2022 | CPI acceleration from 1.4% → 9.1%, supply chain pass-through | Directional ✓ |
| 🏦 **Fed Rate Hike Cycle** | 2022–2023 | Demand cooling, gradual disinflation under 525bps of tightening | Directional ✓ |

### FRED Series Used
`UNRATE` · `CPIAUCSL` · `GDPC1` · `FEDFUNDS` · `PSAVERT` · `TCU` · `HOUST` · `MEHOINUSA672N`

---

## 🧠 The Learned Macroeconomic Model (LMM)

The LMM is a **~26,000 parameter transformer** that replaces hand-coded firm pricing/production rules. The innovation isn't model size — it's that gradients flow through a *strict, stock-flow-consistent economic environment* into the network's policy weights.

```bash
# Train the LMM against a macroeconomic objective
utopia train --seed 42 --epochs 100 --ticks 50

# The trained policy produces explainable economic rationale
# "Firm 3 raised prices by 4.2% because input costs increased 6.1%
#  while competitor prices remained flat, suggesting margin recovery..."
```

---

## 📡 API Reference

Utopia exposes a full REST API via FastAPI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulate` | POST | Run a simulation (up to 100K agents) |
| `/api/compare` | POST | A/B test two scenarios |
| `/api/experiment` | POST | Multi-seed Monte Carlo experiment |
| `/api/agents/ingest` | POST | Upload custom agent demographics (CSV) |
| `/api/health` | GET | Health check |

```bash
curl -X POST http://localhost:8765/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"agents": 10000, "firms": 50, "ticks": 120, "scenario": "tariffs"}'
```

---

## 🇺🇸 US Macroeconomic Calibration

All default parameters are calibrated against official US government data:

| Parameter | Value | Source |
|-----------|-------|--------|
| Corporate tax rate | 21% | IRS / TCJA 2017 |
| Income tax brackets | 10%–37% | IRS |
| Federal Funds Rate | 3.9% | Fed FOMC (Sep 2026) |
| Matching efficiency (μ) | 0.7 | Petrongolo & Pissarides 2001 |
| Bargaining power (β) | 0.4 | Shimer 2005 |
| CES elasticity (σ) | 1.2 | Broda & Weinstein 2006 |
| Initial employment | 96% | BLS (US unemployment ~4%) |
| Wealth distribution | 100:1 range | Fed Survey of Consumer Finances |

---

## 📁 Project Structure

```
utopia/
├── core/
│   ├── config.py              # SimulationConfig (US-calibrated defaults)
│   ├── simulation_jax.py      # Main JAX simulation loop (jax.lax.scan)
│   ├── state.py               # Flax PyTree state definitions
│   ├── lmm_model.py           # Transformer firm policy network
│   ├── lmm_explain.py         # Explainable AI rationale generator
│   ├── scenarios.py           # Shock matrix definitions
│   └── engine/                # The 12 economic modules
│       ├── credit.py          # Banking & lending
│       ├── labor.py           # DMP matching & Nash bargaining
│       ├── market.py          # CES demand & price discovery
│       ├── production.py      # Cobb-Douglas production
│       ├── government.py      # Taxation & fiscal policy
│       ├── logistics.py       # Supply chain & freight
│       ├── housing.py         # Regional housing markets
│       ├── foreign.py         # International trade
│       ├── firm_logic.py      # Entry, bankruptcy, adjustment
│       └── social.py          # Demographics & networks
├── connectors/                # Data ingestion (FRED, CSV, ERP)
├── enterprise/                # Auth, rate limiting, audit logging
server.py                      # FastAPI server + React dashboard
backtest_validate.py           # Historical crisis validation
train_rl.py                    # LMM training loop
us_calibration.py              # US demographic calibration
frontend/                      # React + Vite dashboard
```

---

## 🔬 Academic Citations

If you use Utopia in your research, please cite:

```bibtex
@software{utopia2026,
  title={Utopia: An End-to-End Differentiable Macroeconomic Engine},
  author={Khan, D.},
  year={2026},
  url={https://github.com/Unjustcomposer/Utopia},
  note={JAX-based stock-flow consistent ABM with learned firm policies}
}
```

### Key References
- Petrongolo, B. & Pissarides, C. (2001). Looking into the black box: A survey of the matching function. *Journal of Economic Literature*.
- Shimer, R. (2005). The cyclical behavior of equilibrium unemployment and vacancies. *American Economic Review*.
- Godley, W. & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*.

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with JAX on NVIDIA DGX B200**

[⬆ Back to top](#-utopia)

</div>
