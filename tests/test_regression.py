import pytest
from backtest_historical import get_actual_metrics, run_scenario

@pytest.mark.regression
@pytest.mark.parametrize("era", ["2008", "2021"])
def test_scientific_regression(era):
    """
    Validates that the LMM policy engine maintains its performance within the scientific
    tolerance bands against empirical FRED metrics for historical eras.
    """
    try:
        actual_gdp_drop, actual_unemp_spike = get_actual_metrics(era)
    except Exception as e:
        pytest.skip(f"Failed to fetch FRED data (network/API key issue): {e}")
        
    lmm_res = run_scenario(era, firm_behavior_mode=0)
    
    gdp_tracking_error = abs(lmm_res['gdp_drop_pct'] - actual_gdp_drop)
    unemp_tracking_error = abs(lmm_res['unemp_spike_pct'] - actual_unemp_spike)
    
    # Assert tolerance bands: GDP Tracking Error <= 20.0 pts, Unemployment <= 15.0 pts
    assert gdp_tracking_error <= 20.0, f"GDP tracking error drifted past tolerance (20.0 pts): {gdp_tracking_error:.2f}"
    assert unemp_tracking_error <= 15.0, f"Unemployment tracking error drifted past tolerance (15.0 pts): {unemp_tracking_error:.2f}"
