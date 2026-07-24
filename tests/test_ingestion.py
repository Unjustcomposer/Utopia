import pytest
import jax.numpy as jnp
from config import SimulationConfig
from data_ingestion import GlobalBaselineCompiler, FredMacroIndicator, FredDataClient
from simulation_jax import init_sim_state
import datetime

def test_pydantic_validation():
    # Test valid FRED schema
    indicator = FredMacroIndicator(
        timestamp=datetime.datetime.now(),
        value=4.5
    )
    assert indicator.value == 4.5

def test_fred_client_fallback(monkeypatch):
    # Force the fetch to fail to ensure fallbacks work
    client = FredDataClient()
    def mock_read_csv(url):
        raise Exception("Network Error")
    
    import pandas as pd
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    
    # Should not crash, should return the fallback 4.0
    val = client.fetch_savings_rate().value
    assert isinstance(val, float)
    assert val == 4.0

def test_compiler_output_shapes():
    config = SimulationConfig(num_agents=50, num_firms=10, num_regions=3)
    compiler = GlobalBaselineCompiler(config)
    overrides = compiler.compile_baseline()
    
    assert "agent_budgets" in overrides
    assert overrides["agent_budgets"].shape == (50,)
    assert overrides["firm_capacities"].shape == (10,)
    assert overrides["housing_supply"].shape == (3,)
    assert not jnp.isnan(overrides["agent_budgets"]).any()

def test_init_state_with_overrides():
    config = SimulationConfig(num_agents=50, num_firms=10, num_regions=3)
    compiler = GlobalBaselineCompiler(config)
    overrides = compiler.compile_baseline()
    
    # Initialize without overrides
    state_default = init_sim_state(config, seed=42)
    
    # Initialize with overrides
    state_override = init_sim_state(config, seed=42, baseline_state_overrides=overrides)
    
    # Compare
    assert not jnp.array_equal(state_default.firms.cash, state_override.firms.cash)
    assert jnp.array_equal(state_override.firms.cash, overrides["firm_cash"])
    
    # Ensure equity was recalculated correctly (should not just be the default)
    assert not jnp.array_equal(state_default.firms.equity, state_override.firms.equity)
