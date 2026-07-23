# NexusAI — Agent Economy Simulator
### Scenario Sandbox & Strategy-Search Extension

A multi-agent economic simulation with scenario injection, A/B-style
control/treatment experiments, and parameter-search optimization.

> **⚠️ Every result produced by this simulator is a statement about the
> simulation's internal dynamics — never a prediction about a real company,
> market, or geopolitical event. This is a decision-support and portfolio-
> demonstration tool, not a trading system.**

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  config.py  │────▶│   core.py    │────▶│ simulation.py │
│  Parameters │     │ Agent, Firm, │     │  Tick Loop    │
│             │     │   Market     │     │  + Metrics    │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌──────────────┐               │
                    │ scenario.py  │◀──────────────┤
                    │ 6 Scenario   │               │
                    │   Types      │               │
                    └──────┬───────┘               │
                           │                       │
                    ┌──────▼───────┐               │
                    │experiment.py │◀──────────────┘
                    │ Control vs   │
                    │  Treatment   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌───────────────┐
                    │  search.py   │────▶│ dashboard.py  │
                    │ Grid/Random  │     │ Interactive   │
                    │  + Robustness│     │  Web UI       │
                    └──────────────┘     └───────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install numpy scipy

# Run a single simulation
python main.py run --seed 42 --ticks 120

# Run a control/treatment experiment
python main.py experiment --scenario marketing --spend 5000 --seeds 15

# Run a strategy search
python main.py search --objective profit --method grid --price-min 8 --price-max 20

# Run all 3 demo experiments
python main.py demo

# Launch the interactive dashboard
python main.py dashboard
```

---

## Components

### `config.py` — Simulation Parameters
Central `SimulationConfig` dataclass with ~30 parameters covering population
(agents, wages, savings), firms (capacity, pricing, production), market
mechanics (elasticity, awareness, memory), and experiment defaults.

### `core.py` — Agent, Firm, Market
- **Agent**: Cobb-Douglas utility maximization gated by per-good awareness.
  Agents save a fraction of income (modulated by risk aversion), then spend
  the remainder across visible goods. A sliding memory window tracks past
  prices for price-elasticity adjustments.
- **Firm**: Produces one good, hires/fires agents, and adaptively prices
  based on inventory vs. target buffer. Input cost multiplier models supply
  disruptions.
- **Market**: Proportional rationing when demand exceeds supply; cheapest-
  firm-first allocation.

### `metrics.py` — KPI Computation
Gini coefficient, volume-weighted price index, employment rate, per-firm
revenue/profit/market share, total welfare, average wage — all computed
each tick.

### `simulation.py` — Tick Loop
Phased execution per tick:
1. Firms produce → 2. Pay wages → 3. **Scenario hook** → 4. Agents demand
→ 5. Market clears → 6. Firms adjust → 7. Memory update → 8. Metrics

### `scenario.py` — Scenario Injection (6 Types)

| Scenario | Parameters | Effect |
|---|---|---|
| **MarketingCampaign** | spend, reach, decay_rate | Boosts agent awareness for a good, decaying over time |
| **ProductLaunch** | price, quality, awareness | Adds a new good; agents re-evaluate preferences |
| **FeatureChange** | new_price, new_quality, availability | Modifies an existing good mid-run |
| **SupplyDisruption** | capacity_reduction, cost_increase | Reduces firm output / raises input costs |
| **DemandShock** | risk_aversion_delta, savings_rate_delta | Shifts agent behavior (recession/boom) |
| **TradeDisruption** | affected_goods, cost_increase, availability_reduction | Raises costs / removes supply for goods |

**CompositeScenario** combines multiple scenarios for complex experiments.

All scenarios are **pure parameter mutations** — no new subsystems, no
narrative content. Disasters/shocks are numeric changes to supply, demand,
or cost parameters.

### `experiment.py` — Control/Treatment Framework
- Uses **Common Random Numbers (CRN)**: identical seed produces both control
  and treatment, so the only difference is the scenario.
- Reports treatment − control delta with **95% confidence intervals**
  (t-distribution), **Cohen's d** effect sizes, and **paired t-tests**.
- Flags metrics where CI includes zero ("no significant simulated effect").

### `search.py` — Strategy Search
- **Grid search** or **random search** over a defined parameter space
- Evaluates each combination against an **explicit objective** (firm profit,
  total welfare, price stability)
- **Seed-robustness validation**: top candidates re-evaluated with **fresh
  seeds** not used during search
- Reports **degradation ratio** and **Kolmogorov-Smirnov test** to detect
  overfitting
- Catches and reports cases where a "winning" strategy doesn't hold up

### `dashboard.py` — Interactive Web Dashboard
Dark-themed, glassmorphism-styled web UI with:
- Simulation/scenario configuration
- Real-time Chart.js visualizations
- Experiment and search result panels

---

## Example Experiments

### Experiment 1: Marketing Spend Sweep

**Question (simulated):** How much should Firm 0 spend on marketing for
Good 0 to maximize cumulative profit?

**Setup:**
- 150 agents, 5 firms, 4 goods, 90 ticks
- Sweep ad spend: $0, $2k, $4k, $6k, $8k, $10k
- Scenario: `MarketingCampaign(start=15, duration=40, reach=50%)`
- Objective: `firm_0_profit`
- 8 seeds per candidate, 16 for validation

**How to run:**
```bash
python main.py demo  # runs all 3 experiments
```

**What to look for:**
- Diminishing returns: profit gains flatten at higher spend levels
- Seed robustness: does the top spend level hold across fresh seeds, or
  does a mid-range value prove more stable?

---

### Experiment 2: Supply Shock Response

**Question (simulated):** What happens to prices, employment, and welfare
when the largest firm loses 50% capacity for 20 ticks?

**Setup:**
- 150 agents, 5 firms, 90 ticks
- Scenario: `SupplyDisruption(firm=0, capacity_loss=50%, cost_increase=1.5x,
  start=20, duration=20)`
- 8 paired control/treatment seeds

**What to look for:**
- Price spike during disruption, gradual recovery after
- Employment may dip if the firm can't pay wages
- Other firms may partially absorb demand (competitive substitution)
- Welfare effect magnitude and statistical significance

---

### Experiment 3: Pricing Strategy Under a Demand Shock

**Question (simulated):** What price × quality combination maximizes
Firm 0's profit when agents suddenly become more risk-averse?

**Setup:**
- 150 agents, 90 ticks
- Composite scenario: `DemandShock(risk_aversion +0.2, savings +0.05)` +
  `FeatureChange(price=X, quality=Y)` for Good 0
- Grid: price ∈ {8, 10, 12, 14, 16, 18, 20}, quality ∈ {0.8, 1.0, 1.2}
- 8 seeds per candidate, 16 for validation

**What to look for:**
- During a demand shock, lower prices often perform better (agents are
  spending less, so affordability dominates)
- Higher quality may not compensate for higher prices when agents are
  savings-constrained
- Robustness check: does the "best" strategy survive fresh seeds?

---

## Guardrails

1. **Simulation only.** Every result is about the simulation's internal
   dynamics. No predictions about real companies, markets, or events.
2. **No narrative violence.** Disasters, wars, and conflicts are implemented
   purely as numeric shocks to supply, demand, or trade cost parameters.
3. **No alpha claims.** This is a decision-support tool, not a trading
   system. The strategy search is a documented parameter sweep, not a
   black-box magic strategy finder.
4. **Seed robustness.** The search module explicitly checks for overfitting
   to specific random seeds and reports failures — this is a feature, not
   a bug.

---

## Suggested Resume Line

> "Extended a multi-agent economic simulation into a scenario-testing sandbox
> with A/B-style control/treatment experiments and a parameter-search module
> for strategy optimization (e.g., pricing, marketing spend) under simulated
> demand and supply shocks, with explicit seed-robustness validation to avoid
> overfitting to a single run."

---

## Dependencies

- Python 3.9+
- NumPy ≥ 1.24
- SciPy ≥ 1.10

## License

MIT
