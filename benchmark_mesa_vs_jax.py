import time
import random
import jax
import jax.numpy as jnp
from utopia.core.config import SimulationConfig
from utopia.core.simulation_jax import init_sim_state
from utopia.core.engine_jax import simulation_step

# ── Pure Python (Mesa-style) Baseline ──

class PythonAgent:
    def __init__(self, agent_id, budget):
        self.agent_id = agent_id
        self.budget = budget
        self.employed = False
        self.employer_id = -1
        self.inventory = [0.0] * 5

class PythonFirm:
    def __init__(self, firm_id, cash):
        self.firm_id = firm_id
        self.cash = cash
        self.inventory = 10.0
        self.price = 1.0
        self.employees = []
        self.is_active = True

def run_python_simulation(num_agents, num_firms, num_ticks):
    agents = [PythonAgent(i, random.uniform(10, 100)) for i in range(num_agents)]
    firms = [PythonFirm(i, random.uniform(100, 1000)) for i in range(num_firms)]
    
    start_time = time.time()
    for tick in range(num_ticks):
        # 1. Firms produce (simplified)
        for firm in firms:
            if firm.is_active:
                prod = len(firm.employees) * 1.5
                firm.inventory += prod
                firm.cash -= len(firm.employees) * 1.0 # Wage cost
                
        # 2. Agents consume (simplified random matching)
        for agent in agents:
            if agent.budget > 1.0:
                target_firm = random.choice(firms)
                if target_firm.is_active and target_firm.inventory > 0:
                    bought = min(1.0, target_firm.inventory)
                    target_firm.inventory -= bought
                    cost = bought * target_firm.price
                    agent.budget -= cost
                    target_firm.cash += cost
                    
        # 3. Bankruptcy & Hiring
        for firm in firms:
            if firm.cash < 0:
                firm.is_active = False
                for emp in firm.employees:
                    emp.employed = False
                    emp.employer_id = -1
                firm.employees = []
            elif firm.is_active and len(firm.employees) < 10:
                # Hire random unemployed agent
                candidates = [a for a in agents if not a.employed]
                if candidates:
                    new_hire = random.choice(candidates)
                    new_hire.employed = True
                    new_hire.employer_id = firm.firm_id
                    firm.employees.append(new_hire)
                    
    end_time = time.time()
    return end_time - start_time

# ── JAX Engine ──

def run_jax_simulation(num_agents, num_firms, num_ticks):
    config = SimulationConfig(
        num_agents=num_agents,
        num_firms=num_firms,
        num_ticks=num_ticks,
        firm_behavior_mode=2 # Heuristic for fair comparison
    )
    
    # 1. Initialize
    state = init_sim_state(config, seed=42)
    
    # 2. Compile loop using scan
    @jax.jit
    def run_all_ticks(initial_state):
        def scan_step(state, _):
            new_state = simulation_step(state, config)
            return new_state, None
        
        final_state, _ = jax.lax.scan(scan_step, initial_state, None, length=num_ticks)
        return final_state
        
    # Compile
    print("  Compiling JAX graph...")
    compile_start = time.time()
    compiled_fn = run_all_ticks.lower(state).compile()
    compile_time = time.time() - compile_start
    print(f"  Compilation took {compile_time:.2f}s")
    
    # Execute
    print("  Executing JAX loop...")
    exec_start = time.time()
    final_state = compiled_fn(state)
    # Block until execution finishes
    final_state.agents.budget.block_until_ready()
    exec_time = time.time() - exec_start
    
    return exec_time

def run_benchmark():
    num_agents = 1000
    num_firms = 100
    num_ticks = 50
    
    print(f"=== Utopia Benchmarking ===")
    print(f"Agents: {num_agents:,} | Firms: {num_firms:,} | Ticks: {num_ticks}")
    
    print("\n[1/2] Running Pure Python (Mesa-style) Baseline...")
    py_time = run_python_simulation(num_agents, num_firms, num_ticks)
    print(f"  -> Python Execution Time: {py_time:.4f} seconds")
    
    print("\n[2/2] Running JAX Engine (Vectorized)...")
    try:
        jax_time = run_jax_simulation(num_agents, num_firms, num_ticks)
        print(f"  -> JAX Execution Time: {jax_time:.4f} seconds")
        
        speedup = py_time / jax_time
        print(f"\n=== RESULTS ===")
        print(f"JAX is {speedup:,.0f}x faster than the Python baseline.")
    except Exception as e:
        print(f"JAX execution failed (likely Application Control blocking DLL in this env): {e}")
        
if __name__ == "__main__":
    run_benchmark()
