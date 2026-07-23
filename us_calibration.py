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

# US Census Regions (approximate population weights)
REGIONS = ["Northeast", "Midwest", "South", "West"]
REGION_WEIGHTS = [0.17, 0.21, 0.38, 0.24]

# US Age Groups (approximate adult population weights)
AGE_GROUPS = ["18-25", "26-35", "36-50", "51-65", "65+"]
AGE_WEIGHTS = [0.12, 0.17, 0.25, 0.23, 0.23]

# Cost of Living (COL) Multiplier by Region (Northeast/West are higher)
REGION_COL = {
    "Northeast": 1.15,
    "Midwest": 0.90,
    "South": 0.95,
    "West": 1.15
}

# Wealth Multiplier by Age (wealth accumulates with age)
AGE_WEALTH_MULT = {
    "18-25": 0.3,
    "26-35": 0.6,
    "36-50": 1.2,
    "51-65": 1.8,
    "65+": 1.5
}

def sample_demographics(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n demographic pairs (region, age_group) using vectorization."""
    regions = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    ages = rng.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS)
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
    
    col_mult = np.zeros(n)
    for r, mult in REGION_COL.items():
        col_mult[regions == r] = mult
        
    age_mult = np.zeros(n)
    for a, mult in AGE_WEALTH_MULT.items():
        age_mult[ages == a] = mult
        
    # Lognormal parameters for wealth (budget)
    sigma_wealth = kwargs.get("sigma_wealth", 0.8)
    target_mean_wealth = base_budget * col_mult * age_mult
    mu_wealth = np.log(target_mean_wealth) - (sigma_wealth**2 / 2)
    budgets = rng.lognormal(mu_wealth, sigma_wealth)
    
    # Lognormal parameters for expected wage
    sigma_wage = kwargs.get("sigma_wage", 0.5)
    target_mean_wage = base_wage * col_mult
    mu_wage = np.log(target_mean_wage) - (sigma_wage**2 / 2)
    expected_wages = rng.lognormal(mu_wage, sigma_wage)
    
    # Savings rate
    savings_mean = kwargs.get("savings_mean", 0.05)
    savings_std = kwargs.get("savings_std", 0.03)
    savings_rates = np.clip(rng.normal(loc=savings_mean, scale=savings_std, size=n), 0.0, 0.4)
    
    return budgets, expected_wages, savings_rates
