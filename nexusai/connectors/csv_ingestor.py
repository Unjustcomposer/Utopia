"""
CSV Data Ingestor
=================
Provides a static, zero-cost data ingestion pipeline for Design Partners.
Instead of requiring a live SAP S/4HANA connection (which requires months of security audits),
this module allows mid-market retailers to upload anonymized historical inventory extracts
for offline simulation and backtesting.
"""

import pandas as pd
import jax.numpy as jnp
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def ingest_historical_csv(filepath: str) -> Dict[str, jnp.ndarray]:
    """
    Ingests a static CSV dump of historical inventory levels and converts it
    into JAX-compatible state override arrays for initializing the simulation.
    
    Expected CSV format:
    SKU,RegionID,InitialInventory,UnitCost,HistoricalDemand,LeadTimeDays
    """
    logger.info(f"Ingesting static CSV data from {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return {}
        
    required_cols = ["SKU", "RegionID", "InitialInventory", "UnitCost", "HistoricalDemand"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing required column: {col}")
            
    # Convert to JAX arrays for state initialization
    try:
        num_firms = len(df)
        firm_capacities = jnp.array(df["HistoricalDemand"].values * 1.5, dtype=jnp.float32)
        firm_cash = jnp.array(df["InitialInventory"].values * df["UnitCost"].values, dtype=jnp.float32)
        
        overrides = {
            "firm_capacities": firm_capacities,
            "firm_cash": firm_cash
        }
        logger.info(f"Successfully processed {num_firms} records into JAX state overrides.")
        return overrides
    except Exception as e:
        logger.error(f"Error converting dataframe to JAX arrays: {e}")
        return {}

from nexusai.connectors.usitc_client import UsitcDataClient

def parse_usitc_tariff_csv(filepath: str, hts_codes: list = None) -> Dict[str, Any]:
    """Parses USITC HTS CSV format to a tariff rates dictionary, or fetches live if possible."""
    logger.info(f"Attempting to fetch live tariff rates, falling back to CSV {filepath}")
    tariff_rates = {}
    
    # Try fetching live first
    if hts_codes:
        client = UsitcDataClient()
        try:
            live_rates = client.fetch_tariff_rates(hts_codes)
            if live_rates:
                tariff_rates.update(live_rates)
        except Exception as e:
            logger.warning(f"Live USITC fetch failed: {e}. Falling back to CSV.")
            
    try:
        df = pd.read_csv(filepath)
        if 'HTS8' in df.columns and 'Rate' in df.columns:
            for _, row in df.iterrows():
                code = str(row['HTS8'])
                # Only use CSV fallback if we didn't get it live
                if code not in tariff_rates:
                    tariff_rates[code] = float(row['Rate'])
    except Exception as e:
        logger.error(f"Failed to parse USITC CSV: {e}")
    return tariff_rates

def parse_bls_qcew_csv(filepath: str) -> Dict[str, Any]:
    """Parses BLS QCEW CSV to wage and employment data."""
    logger.info(f"Parsing BLS QCEW CSV from {filepath}")
    labor_data = {}
    try:
        df = pd.read_csv(filepath)
        if 'avg_wkly_wage' in df.columns:
            labor_data['avg_weekly_wage'] = float(df['avg_wkly_wage'].mean())
        if 'month3_emplvl' in df.columns:
            labor_data['total_employment'] = float(df['month3_emplvl'].sum())
    except Exception as e:
        logger.error(f"Failed to parse BLS QCEW CSV: {e}")
    return labor_data

if __name__ == "__main__":
    # Test stub
    # print(ingest_historical_csv("data/sample_inventory.csv"))
    pass
