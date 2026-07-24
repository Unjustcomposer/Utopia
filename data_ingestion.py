import datetime
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import jax.numpy as jnp

from config import SimulationConfig

# ── FRED Data Schemas ──────────────────────────────

class FredMacroIndicator(BaseModel):
    timestamp: datetime.datetime
    value: float

# ── Real Data Client (FRED API) ──────────────────────────────────────

class FredDataClient:
    """Fetches real-world macroeconomic data directly from FRED CSV endpoints."""
    
    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    
    def _fetch_latest_value(self, series_id: str) -> float:
        """Downloads the CSV from FRED and returns the most recent observation."""
        url = f"{self.BASE_URL}{series_id}"
        try:
            # Pandas can read directly from the URL
            df = pd.read_csv(url)
            # FRED CSVs have 'observation_date' and the series_id as columns
            # Get the very last row's value
            latest_val = float(df[series_id].iloc[-1])
            return latest_val
        except Exception as e:
            # Fallback values if network fails during fetch
            fallbacks = {
                'PSAVERT': 4.0,       # 4% savings rate
                'TCU': 78.0,          # 78% capacity utilization
                'HOUST': 1400.0,      # 1.4M housing starts
                'MEHOINUSA672N': 75000.0 # $75k median income
            }
            return fallbacks.get(series_id, 0.0)

    def fetch_savings_rate(self) -> FredMacroIndicator:
        """Personal Saving Rate (PSAVERT) - percentage."""
        val = self._fetch_latest_value('PSAVERT')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val)

    def fetch_capacity_utilization(self) -> FredMacroIndicator:
        """Capacity Utilization: Total Industry (TCU) - percentage."""
        val = self._fetch_latest_value('TCU')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val)

    def fetch_housing_starts(self) -> FredMacroIndicator:
        """New Privately-Owned Housing Units Started (HOUST) - thousands of units."""
        val = self._fetch_latest_value('HOUST')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val)
        
    def fetch_median_income(self) -> FredMacroIndicator:
        """Real Median Household Income in the United States (MEHOINUSA672N)."""
        val = self._fetch_latest_value('MEHOINUSA672N')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val)

# ── Global Baseline Compiler ─────────────────────────────────────────

class GlobalBaselineCompiler:
    """Transforms raw real-world macro data into JAX tensors for SimState overrides."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client = FredDataClient()
        
    def compile_baseline(self) -> Dict[str, jnp.ndarray]:
        """
        Fetches live FRED data and broadcasts/distributes it across JAX tensors,
        adding synthetic variance to maintain heterogeneous agent/firm behavior.
        """
        # 1. Fetch real macro data
        savings_macro = self.client.fetch_savings_rate().value
        capacity_macro = self.client.fetch_capacity_utilization().value
        housing_macro = self.client.fetch_housing_starts().value
        income_macro = self.client.fetch_median_income().value
        
        # FRED provides percentages (e.g., 4.5 for 4.5%). Convert to decimal.
        base_savings_rate = savings_macro / 100.0
        base_capacity_util = capacity_macro / 100.0
        
        # 2. Distribute to Agents (with variance for heterogeneity)
        # Agents have different savings rates around the macro mean
        np.random.seed(42) # Deterministic compilation for baseline
        
        agent_savings_rates = np.clip(
            np.random.normal(loc=base_savings_rate, scale=0.02, size=self.config.num_agents), 
            0.0, 0.5
        ).astype(np.float32)
        
        # Agent budgets based on median income (scaled down for tick-based simulation, e.g. monthly)
        monthly_income = income_macro / 12.0
        agent_budgets = np.random.lognormal(
            mean=np.log(monthly_income) - (0.5**2 / 2), 
            sigma=0.5, 
            size=self.config.num_agents
        ).astype(np.float32)
        
        # 3. Distribute to Firms
        # Assume base production capacity is 100, and current inventory reflects utilization
        firm_capacities = np.random.normal(loc=100.0, scale=10.0, size=self.config.num_firms).astype(np.float32)
        firm_cash = (firm_capacities * base_capacity_util * 10.0).astype(np.float32)
        
        # 4. Macro/Housing
        # Distribute housing starts equally across regions
        housing_supply = (np.ones(self.config.num_regions) * housing_macro / self.config.num_regions).astype(np.float32)
        
        # 5. JAX Tensor Emission
        overrides = {
            "agent_budgets": jnp.array(agent_budgets),
            "agent_savings_rates": jnp.array(agent_savings_rates),
            "firm_capacities": jnp.array(firm_capacities),
            "firm_cash": jnp.array(firm_cash),
            "housing_supply": jnp.array(housing_supply)
        }
        
        return overrides
