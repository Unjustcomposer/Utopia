import numpy as np
from nexusai.core import climate_shocks

def generate_shock_matrix(num_ticks: int, scenario_name: str, telematics_multiplier: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generates a matrix of shape (num_ticks, 5) where columns are:
    0: Interest Rate Hike (additive)
    1: Savings Rate Increase (additive)
    2: Input Cost Multiplier (multiplicative, baseline 1.0)
    3: Infrastructure Damage Severity (additive)
    4: Route Closure Penalty (additive)
    
    This is passed to the JAX lax.scan to inject dynamic shocks.
    """
    # Baseline: No shocks (0.0, 0.0, 1.0, 0.0, 0.0), applied with baseline telematics risk
    shocks = np.zeros((num_ticks, 5), dtype=np.float32)
    shocks[:, 2] = telematics_multiplier
    
    if scenario_name == "baseline":
        return shocks
        
    elif scenario_name == "tariff_shock":
        # At tick 20, input costs permanently increase by 20%
        if num_ticks > 20:
            shocks[20:, 2] = 1.2 * telematics_multiplier
            
    elif scenario_name == "rate_hike":
        # At tick 10, macro rates increase by 500 bps
        if num_ticks > 10:
            shocks[10:, 0] = 0.05
            
    elif scenario_name == "oil_shock":
        # Gradual massive increase in input costs starting tick 15, peaking at 2.0x at tick 30
        for t in range(15, num_ticks):
            mult = 1.0 + min(1.0, (t - 15) / 15.0)
            shocks[t, 2] = mult * telematics_multiplier
            
    elif scenario_name == "recession":
        # Sudden panic at tick 10: savings rates spike by 10% (demand drops)
        if num_ticks > 10:
            shocks[10:30, 1] = 0.10 # Panic lasts 20 ticks
            
    elif scenario_name == "pandemic":
        # Tick 5: 30% savings spike (lockdowns) + 50% cost increase (supply chains)
        if num_ticks > 5:
            shocks[5:25, 1] = 0.30
            shocks[5:25, 2] = 1.5 * telematics_multiplier
            
    elif scenario_name == "supply_chain_2021":
        for t in range(num_ticks):
            if t < 30:
                progress = t / 30.0
                shocks[t, 2] = (1.0 + 1.5 * progress) * telematics_multiplier
            elif t < 60:
                progress = (t - 30) / 30.0
                shocks[t, 2] = (2.5 + 1.5 * progress) * telematics_multiplier
                shocks[t, 1] = -0.02
            elif t < 90:
                progress = (t - 60) / 30.0
                shocks[t, 2] = (4.0 - 2.5 * progress) * telematics_multiplier
                shocks[t, 1] = 0.04
                shocks[t, 0] = 0.001
            else:
                progress = min(1.0, (t - 90) / 30.0)
                shocks[t, 2] = (1.5 - 0.3 * progress) * telematics_multiplier
                shocks[t, 1] = 0.0
                shocks[t, 0] = 0.0025
                
    elif scenario_name == "hurricane_gulf_coast":
        # Massive localized infrastructure damage for 15 ticks, followed by a slow 40-tick rebuild phase.
        # Let's start at tick 10.
        for t in range(10, num_ticks):
            if t < 10 + 15:
                shocks[t, 3] = 0.8  # high severity
            elif t < 10 + 15 + 40:
                progress = (t - (10 + 15)) / 40.0
                shocks[t, 3] = 0.8 * (1.0 - progress) # rebuild slowly reduces severity
                
    elif scenario_name == "panama_canal_drought":
        # Chronic route closure penalty that slowly increases in severity over 60 ticks.
        # Let's start at tick 5
        for t in range(5, num_ticks):
            if t < 5 + 60:
                progress = (t - 5) / 60.0
                shocks[t, 4] = 0.5 * progress # penalty up to 0.5
            else:
                shocks[t, 4] = 0.5

    return shocks

SCENARIO_LIST = ["baseline", "tariff_shock", "rate_hike", "oil_shock", "recession", "pandemic", "supply_chain_2021", "hurricane_gulf_coast", "panama_canal_drought"]
