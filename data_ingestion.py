import datetime
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
    is_fallback: bool = False

# ── Real Data Client (FRED API) ──────────────────────────────────────

class FredDataClient:
    """Fetches real-world macroeconomic data directly from FRED CSV endpoints."""
    
    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    
    def _fetch_latest_value(self, series_id: str) -> tuple[float, bool]:
        """Downloads the CSV from FRED and returns the most recent observation."""
        url = f"{self.BASE_URL}{series_id}"
        try:
            # Pandas can read directly from the URL
            df = pd.read_csv(url)
            # FRED CSVs have 'observation_date' and the series_id as columns
            # Get the very last row's value
            latest_val = float(df[series_id].iloc[-1])
            return latest_val, False
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"FRED fetch failed for {series_id}, using fallback value. Error: {e}")
            # Fallback values if network fails during fetch
            fallbacks = {
                'PSAVERT': 4.0,       # 4% savings rate
                'TCU': 78.0,          # 78% capacity utilization
                'HOUST': 1400.0,      # 1.4M housing starts
                'MEHOINUSA672N': 75000.0 # $75k median income
            }
            return fallbacks.get(series_id, 0.0), True

    def fetch_savings_rate(self) -> FredMacroIndicator:
        """Personal Saving Rate (PSAVERT) - percentage."""
        val, is_fallback = self._fetch_latest_value('PSAVERT')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val, is_fallback=is_fallback)

    def fetch_capacity_utilization(self) -> FredMacroIndicator:
        """Capacity Utilization: Total Industry (TCU) - percentage."""
        val, is_fallback = self._fetch_latest_value('TCU')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val, is_fallback=is_fallback)

    def fetch_housing_starts(self) -> FredMacroIndicator:
        """New Privately-Owned Housing Units Started (HOUST) - thousands of units."""
        val, is_fallback = self._fetch_latest_value('HOUST')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val, is_fallback=is_fallback)
        
    def fetch_median_income(self) -> FredMacroIndicator:
        """Real Median Household Income in the United States (MEHOINUSA672N)."""
        val, is_fallback = self._fetch_latest_value('MEHOINUSA672N')
        return FredMacroIndicator(timestamp=datetime.datetime.now(), value=val, is_fallback=is_fallback)

# ── Global Baseline Compiler ─────────────────────────────────────────

class GlobalBaselineCompiler:
    """Transforms raw real-world macro data into JAX tensors for SimState overrides."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.client = FredDataClient()
        
    def compile_baseline(self, seed: int = 42) -> tuple[Dict[str, jnp.ndarray], bool]:
        """
        Fetches live FRED data and broadcasts/distributes it across JAX tensors,
        adding synthetic variance to maintain heterogeneous agent/firm behavior.
        Returns a tuple of (overrides, is_fallback).
        """
        # 1. Fetch real macro data
        savings = self.client.fetch_savings_rate()
        capacity = self.client.fetch_capacity_utilization()
        housing = self.client.fetch_housing_starts()
        income = self.client.fetch_median_income()
        
        savings_macro = savings.value
        capacity_macro = capacity.value
        housing_macro = housing.value
        income_macro = income.value
        
        is_fallback = savings.is_fallback or capacity.is_fallback or housing.is_fallback or income.is_fallback
        
        # FRED provides percentages (e.g., 4.5 for 4.5%). Convert to decimal.
        base_savings_rate = savings_macro / 100.0
        base_capacity_util = capacity_macro / 100.0
        
        # 2. Distribute to Agents (with variance for heterogeneity)
        # Agents have different savings rates around the macro mean
        rng = np.random.default_rng(seed)
        
        agent_savings_rates = np.clip(
            rng.normal(loc=base_savings_rate, scale=0.02, size=self.config.num_agents), 
            0.0, 0.5
        ).astype(np.float32)
        
        # Agent budgets based on median income (scaled down for tick-based simulation, e.g. monthly)
        monthly_income = income_macro / 12.0
        agent_budgets = rng.lognormal(
            mean=np.log(monthly_income) - (0.5**2 / 2), 
            sigma=0.5, 
            size=self.config.num_agents
        ).astype(np.float32)
        
        # 3. Distribute to Firms
        # Assume base production capacity is 100, and current inventory reflects utilization
        firm_capacities = rng.normal(loc=100.0, scale=10.0, size=self.config.num_firms).astype(np.float32)
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
        
        return overrides, is_fallback
