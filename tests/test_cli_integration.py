"""Integration tests that exercise each CLI subcommand end-to-end."""
import subprocess
import sys
import pytest

@pytest.mark.integration
def test_cli_run():
    result = subprocess.run(
        [sys.executable, "main.py", "run", "--seed", "42", "--ticks", "5", "--agents", "50"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "SIMULATION METRICS" in result.stdout

@pytest.mark.integration
def test_cli_train():
    result = subprocess.run(
        [sys.executable, "main.py", "train", "--seed", "42", "--epochs", "1", "--ticks", "5"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "Training Complete" in result.stdout

@pytest.mark.integration
def test_cli_experiment():
    result = subprocess.run(
        [sys.executable, "main.py", "experiment", "--scenario-a", "baseline", "--scenario-b", "tariffs", "--ticks", "5"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "A/B TEST RESULT" in result.stdout

@pytest.mark.integration
def test_cli_search():
    result = subprocess.run(
        [sys.executable, "main.py", "search", "--num-seeds", "2", "--ticks", "5"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "ROBUSTNESS" in result.stdout

@pytest.mark.integration
def test_cli_backtest():
    result = subprocess.run(
        [sys.executable, "main.py", "backtest"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "Tracking Error" in result.stdout

@pytest.mark.integration
def test_cli_report():
    result = subprocess.run(
        [sys.executable, "main.py", "report", "--scenario", "baseline"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "NexusAI_Report_baseline.pdf" in result.stdout

@pytest.mark.integration
def test_cli_demo():
    result = subprocess.run(
        [sys.executable, "main.py", "demo", "--ticks", "5"],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0
    assert "Executing Full Feature Demo" in result.stdout
