import numpy as np

def generate_shock_matrix(num_ticks: int, scenario_name: str, telematics_multiplier: float = 1.0) -> np.ndarray:
    """
    Generates a matrix of shape (num_ticks, 3) where columns are:
    0: Interest Rate Hike (additive)
    1: Savings Rate Increase (additive)
    2: Input Cost Multiplier (multiplicative, baseline 1.0)
    
    This is passed to the JAX lax.scan to inject dynamic shocks.
    """
    # Baseline: No shocks (0.0, 0.0, 1.0), applied with baseline telematics risk
    shocks = np.zeros((num_ticks, 3), dtype=np.float32)
    shocks[:, 2] = telematics_multiplier
    
    if scenario_name == "baseline":
        return shocks
        
    elif scenario_name == "tariff_shock":
        # At tick 20, input costs permanently increase by 20%
        if num_ticks > 20:
            shocks[20:, 2] = 1.2 * telematics_multiplier
            
    elif scenario_name == "rate_hike":
        # At tick 10, central bank raises rates by 500 bps
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
            
    return shocks

SCENARIO_LIST = ["baseline", "tariff_shock", "rate_hike", "oil_shock", "recession", "pandemic"]
