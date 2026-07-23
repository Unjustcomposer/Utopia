"""
US Macroeconomic Calibration Module
===================================
Provides sampling functions to initialize the agent-based simulation with 
realistic United States demographics (Age, Region) and correlated wealth/wage 
distributions to model the actual US economy.
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

def sample_demographics(rng: np.random.Generator, n: int) -> list[Tuple[str, str]]:
    """Sample n demographic pairs (region, age_group)."""
    regions = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    ages = rng.choice(AGE_GROUPS, size=n, p=AGE_WEIGHTS)
    return list(zip(regions, ages))

def sample_agent_financials(
    rng: np.random.Generator, 
    region: str, 
    age_group: str,
    base_budget: float,
    base_wage: float
) -> Tuple[float, float, float]:
    """
    Sample budget, expected wage, and savings rate based on US data.
    Uses lognormal distributions for wealth inequality (Gini ~0.48).
    
    Returns:
        (budget, expected_wage, savings_rate)
    """
    col_mult = REGION_COL[region]
    age_mult = AGE_WEALTH_MULT[age_group]
    
    # Lognormal parameters for wealth (budget)
    # Mean of lognormal = exp(mu + sigma^2 / 2)
    # A sigma of ~0.8 gives a Gini of ~0.45
    sigma_wealth = 0.8
    target_mean_wealth = base_budget * col_mult * age_mult
    mu_wealth = np.log(target_mean_wealth) - (sigma_wealth**2 / 2)
    budget = float(rng.lognormal(mu_wealth, sigma_wealth))
    
    # Lognormal parameters for expected wage (less skewed than wealth)
    sigma_wage = 0.5
    target_mean_wage = base_wage * col_mult
    mu_wage = np.log(target_mean_wage) - (sigma_wage**2 / 2)
    expected_wage = float(rng.lognormal(mu_wage, sigma_wage))
    
    # Savings rate (US average ~5%, normally distributed but bounded)
    savings_rate = float(np.clip(rng.normal(loc=0.05, scale=0.03), 0.0, 0.4))
    
    return budget, expected_wage, savings_rate
