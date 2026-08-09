"""
US Macroeconomic Calibration Module
===================================
Provides sampling functions to initialize the agent-based simulation with 
realistic United States demographics (Age, Region) and correlated wealth/wage 
distributions to model the actual US economy.

Vectorized for AgentPopulation architecture.
"""

import numpy as np
from typing import Dict, Any, Tuple

import os
import json
import pandas as pd
from nexusai.core.config import SimulationConfig, CalibrationProfile

def load_calibration_profile(name: str) -> dict:
    """Load a calibration profile from a JSON file."""
    profile_path = os.path.join(os.path.dirname(__file__), "data", "calibration_profiles", f"{name}.json")
    with open(profile_path, 'r') as f:
        data = json.load(f)
    return data

def config_from_calibration(profile_name: str) -> SimulationConfig:
    data = load_calibration_profile(profile_name)
    profile = CalibrationProfile(**data)
    return SimulationConfig.from_profile(profile)


# Load empirical data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "us_demographics.csv")
try:
    _df = pd.read_csv(DATA_PATH)
    REGIONS = _df["Region"].unique().tolist()
    AGE_GROUPS = _df["AgeGroup"].unique().tolist()
    
    # Precompute probability weights for combinations
    _pop_weights = _df["PopulationWeight"].values
    _pop_weights = _pop_weights / np.sum(_pop_weights)
    
    # Maps for median wealth and mean wage
    _wealth_map = {}
    _wage_map = {}
    for _, row in _df.iterrows():
        key = (row["Region"], row["AgeGroup"])
        _wealth_map[key] = row["MedianWealth"]
        _wage_map[key] = row["MeanWage"]
except Exception as e:
    # Fallback for CI/CD if file missing
    print(f"Warning: Could not load {DATA_PATH}, using fallback. Error: {e}")
    REGIONS = ["Northeast", "Midwest", "South", "West"]
    AGE_GROUPS = ["18-25", "26-35", "36-50", "51-65", "65+"]
    _pop_weights = np.ones(20) / 20.0
    _df = None

def sample_demographics(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n demographic pairs (region, age_group) using empirical combinations."""
    if _df is not None:
        indices = rng.choice(len(_df), size=n, p=_pop_weights)
        regions = _df["Region"].values[indices]
        ages = _df["AgeGroup"].values[indices]
    else:
        regions = rng.choice(REGIONS, size=n)
        ages = rng.choice(AGE_GROUPS, size=n)
    return regions, ages

def sample_agent_financials(
    rng: np.random.Generator, 
    regions: np.ndarray, 
    ages: np.ndarray,
    base_budget: float,
    base_wage: float,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample budget, expected wage, and savings rate based on US data.
    Fully vectorized over the agent population arrays.
    
    Optional overriding kwargs for evolutionary calibration:
        - sigma_wealth (float): Variance in lognormal wealth distribution. Default 0.8.
        - sigma_wage (float): Variance in lognormal expected wage. Default 0.5.
        - savings_mean (float): Mean of normal savings distribution. Default 0.05.
        - savings_std (float): Std dev of normal savings distribution. Default 0.03.
    """
    n = len(regions)
    
    target_mean_wealth = np.zeros(n)
    target_mean_wage = np.zeros(n)
    
    if _df is not None:
        for i in range(n):
            key = (regions[i], ages[i])
            target_mean_wealth[i] = _wealth_map.get(key, base_budget)
            target_mean_wage[i] = _wage_map.get(key, base_wage)
    else:
        target_mean_wealth = np.full(n, base_budget)
        target_mean_wage = np.full(n, base_wage)
        
    # Lognormal parameters for wealth (budget)
    sigma_wealth = kwargs.get("sigma_wealth", 0.8)
    mu_wealth = np.log(target_mean_wealth + 1e-6) - (sigma_wealth**2 / 2)
    budgets = rng.lognormal(mu_wealth, sigma_wealth)
    
    # Lognormal parameters for expected wage
    sigma_wage = kwargs.get("sigma_wage", 0.5)
    mu_wage = np.log(target_mean_wage + 1e-6) - (sigma_wage**2 / 2)
    expected_wages = rng.lognormal(mu_wage, sigma_wage)
    
    # Savings rate
    savings_mean = kwargs.get("savings_mean", 0.05)
    savings_std = kwargs.get("savings_std", 0.03)
    savings_rates = np.clip(rng.normal(loc=savings_mean, scale=savings_std, size=n), 0.0, 0.4)
    
    return budgets, expected_wages, savings_rates
