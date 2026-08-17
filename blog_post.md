# Training a Macroeconomy: How We Achieved <15% Error on the 2008 Crash using Differentiable Simulation

When modeling supply chains, logistics, and macroeconomics, traditional approaches force a harsh tradeoff: either you use standard econometric equations that lack micro-level realism (ignoring the fact that economies are made of individual firms and people), or you build Agent-Based Models (ABMs) that are computationally agonizing and fundamentally impossible to optimize via gradient descent.

At Utopia, we wanted both. We wanted the structural realism of 100,000 independent agents and firms, constrained by strict accounting laws, but we wanted to train the overarching policies of those firms using modern deep learning.

This is the story of how we ported a classical macroeconomic agent-based model to JAX, fixed a massive structural collapse, and trained a Differentiable Firm Policy Network that successfully tracks the 2008 FRED historical data with less than 15% error.

## The Problem: Object-Oriented Economics is Too Slow

Our initial prototype was built using classical Python frameworks. It modeled firms hiring agents, producing goods, and setting prices. The problem? It took 3 minutes to run a single 50-tick simulation of 100,000 agents. 

If you want to train a neural network by backpropagating through a simulation, you need to run that simulation thousands of times. Three minutes per forward pass is a death sentence for machine learning.

We ripped out the classical framework and rewrote the entire engine in JAX. By representing the economy not as a collection of Python objects, but as a series of large, XLA-compiled tensors, we achieved a **600x execution speedup**. Our 50-tick simulation now runs in ~0.3 seconds. 

## The Challenge: Differentiating through a Market

Speed was only half the battle. To train our Learned Macroeconomic Model (LMM)—a Transformer-based Firm Policy Network—we needed the entire simulation to be end-to-end differentiable.

Markets typically clear using discrete sorting algorithms (e.g., sort firms by price, buy from the cheapest until inventory is gone). Discrete sorting operations block gradients. To fix this, we implemented **continuous probability masking and fractional matching**. Instead of discrete binary transactions, agents distribute their demand probabilistically across firms based on a softmax over prices. This allows `jax.value_and_grad` to flow freely from the final macroeconomic loss (GDP and unemployment) all the way back to the LMM's weights in tick 1.

## The Collapse: Enforcing Stock-Flow Consistency

When we first turned the engine on, the economy imploded. Total output dropped to zero within 20 ticks.

We traced the issue to a failure in **Stock-Flow Consistency (SFC)**. In the real world, money is never magically destroyed; it just changes hands. In our early engine, capital was depreciating at a fixed rate but never being reinvested, causing the productive capacity of the economy to rot away. Additionally, our GDP metric was double-counting price deflation.

We stabilized the engine by:
1. **Switching to Real GDP**: We decoupled volume from price changes by calculating output against a constant base price, matching the FRED `GDPC1` convention.
2. **Capital Reinvestment**: We added a continuous capital reinvestment function for profitable firms and lowered base depreciation.
3. **Bankruptcy Buffers**: We softened the bankruptcy threshold to allow firms to survive short-term liquidity crunches without immediately firing their entire workforce.

## The Result: <15% Tracking Error

With the structural leaks plugged and gradients flowing cleanly, we trained the LMM against historical demand and supply shocks.

We compared the trained AI policy against empirical data from the St. Louis Federal Reserve (FRED) for the three major modern macroeconomic crises. The AI policy successfully navigated the demand shocks and supply chain disruptions, keeping the tracking error remarkably low across all regimes:
- **2008 Financial Crisis**: 6.74 pts GDP tracking error, 8.94 pts Unemployment tracking error.
- **2020 Covid Shock**: 12.26 pts GDP tracking error, 14.73 pts Unemployment tracking error.
- **2021 Supply Chain Crunch**: 0.53 pts GDP tracking error, 1.43 pts Unemployment tracking error.

### Try it Yourself (Reproducible Benchmark)

We believe in falsifiable claims. The engine, the JAX compilation, and the FRED backtest are fully reproducible.

```bash
# 1. Clone the repo
git clone https://github.com/omsingh/utopia.git
cd utopia

# 2. Install native dependencies
python -m venv venv
source venv/bin/activate
pip install -e .

# 3. Run the reproducible backtest against the 2008 crash
utopia train --scenario 2008 --seed 42
```

## What's Next?

Our core engine and LMM policy are live, and our FRED data ingestion is fully operational. Our next focus is moving from simulated mock-connectors to live, real-time integrations with SAP and Oracle ERP systems, allowing enterprises to inject their live supply chain telemetry directly into the differentiable engine.

The era of black-box macroeconomic simulations is over. Welcome to Utopia.
