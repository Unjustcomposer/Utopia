import jax.numpy as jnp
import matplotlib.pyplot as plt
from backtest_2021_supply_chain import run_2021_backtest

history, _, _ = run_2021_backtest()

print("Inventory over time:", history["inventory"][-20:])
print("Prices over time:", history["price"][-20:])
print("Employment over time:", history["employment"][-20:])
