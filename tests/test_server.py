import pytest
from fastapi.testclient import TestClient
from server import app
from utopia.enterprise.auth import get_current_user, get_admin_user, User

def override_get_current_user():
    return User(username="test_user", tenant_id="tenant_123")

def override_get_admin_user():
    return User(username="admin_user", tenant_id="tenant_123")

app.dependency_overrides[get_current_user] = override_get_current_user
app.dependency_overrides[get_admin_user] = override_get_admin_user

client = TestClient(app)

def test_get_scenarios():
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert isinstance(scenarios, list)
    assert "baseline" in scenarios

def test_model_status():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "present" in data

def test_api_explain():
    payload = {
        "demand_history": [10.0, 11.0, 10.5],
        "profit_history": [100.0, 105.0, 102.0],
        "price_history": [10.0, 10.0, 10.0],
        "macro_price_history": [10.0, 10.1, 10.2],
        "macro_rate_history": [0.05, 0.05, 0.05]
    }
    response = client.post("/api/explain?format=executive", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "decisions" in data
    assert "safety_status" in data

def test_api_run():
    payload = {
        "agents": 50,
        "firms": 2,
        "goods": 2,
        "ticks": 5,
        "scenario": "baseline",
        "seed": 42
    }
    response = client.post("/api/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics_history" in data
    assert "summary" in data

def test_api_compare():
    payload = {
        "agents": 50,
        "firms": 2,
        "goods": 2,
        "ticks": 5,
        "scenario": "tariffs",
        "seed": 42
    }
    response = client.post("/api/run/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "baseline" in data
    assert "scenario" in data

def test_api_experiment():
    payload = {
        "agents": 50,
        "firms": 2,
        "goods": 2,
        "ticks": 5,
        "scenario_a": "baseline",
        "scenario_b": "tariffs",
        "seed": 42,
        "num_seeds": 2
    }
    response = client.post("/api/experiment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scenario_a" in data
    assert "scenario_b" in data
    assert "deltas" in data
