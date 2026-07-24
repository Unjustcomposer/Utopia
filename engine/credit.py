import jax
import jax.numpy as jnp
from config import SimulationConfig
from state import SimState

def _credit_market_step(state: SimState, config: SimulationConfig) -> SimState:
    """Endogenous credit and deposit step."""
    agents, firms, macro = state.agents, state.firms, state.macro
    
    deposit_rate = jnp.maximum(0.001, macro.base_rate - config.commercial_bank_spread / 2)
    lending_rate = macro.base_rate + config.commercial_bank_spread / 2
    
    # Agents earn interest on savings
    interest_earned = agents.savings * deposit_rate
    new_savings = agents.savings + interest_earned
    new_deposits = jnp.sum(new_savings)
    
    # Firms pay interest and request loans
    # Pay interest
    interest_payment = firms.debt * lending_rate
    firm_cash_after_interest = firms.cash - interest_payment
    
    # Try to pay down principal
    paydown = jnp.where(
        firm_cash_after_interest > firms.debt,
        firms.debt,
        jnp.where(firm_cash_after_interest > 0, firm_cash_after_interest * 0.5, 0.0)
    )
    firm_cash_after_paydown = firm_cash_after_interest - paydown
    firm_debt_after_paydown = firms.debt - paydown
    
    # Request loans
    expected_labor_cost = firms.num_employees * firms.wage_offer
    expected_input_cost = firms.production_capacity * config.input_cost_base * firms.input_cost_multiplier
    required_cash = expected_labor_cost + expected_input_cost
    
    shortfall = jnp.maximum(0.0, required_cash - firm_cash_after_paydown)
    max_new_lending = (new_deposits / config.reserve_requirement) - macro.loans
    
    # Distribute available lending proportionally or capped
    total_shortfall = jnp.sum(shortfall)
    # Avoid division by zero and NaN gradients
    safe_shortfall = jnp.where(total_shortfall > 0, total_shortfall, 1.0)
    ratio = jnp.where(total_shortfall > 0, jnp.minimum(1.0, max_new_lending / safe_shortfall), 0.0)
    ratio = jnp.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = jnp.maximum(0.0, ratio)
    
    granted_loans = shortfall * ratio
    
    new_cash = firm_cash_after_paydown + granted_loans
    new_debt = firm_debt_after_paydown + granted_loans
    new_macro_loans = macro.loans - jnp.sum(paydown) + jnp.sum(granted_loans)
    
    new_bank_equity = macro.bank_equity + jnp.sum(interest_payment) - jnp.sum(interest_earned)
    
    # Update Macro Base Rate (Taylor rule based on previous tick inflation)
    # We will do this at the end of the tick when we have new prices
    
    new_agents = agents._replace(savings=new_savings)
    new_firms = firms._replace(cash=new_cash, debt=new_debt)
    new_macro = macro._replace(deposits=new_deposits, loans=new_macro_loans, bank_equity=new_bank_equity)
    
    return state._replace(agents=new_agents, firms=new_firms, macro=new_macro)


