"""
USITC DataWeb API Client
========================
Client for fetching live HTS (Harmonized Tariff Schedule) rates from the USITC DataWeb API.
If no API key is provided, falls back to mocked data for CI/CD and unconfigured demo environments.
"""

import os
import requests
import logging
from typing import Dict, List, Any
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

class UsitcDataClient:
    """Client for fetching live tariff rates from USITC."""
    
    BASE_URL = "https://dataweb.usitc.gov/api/v1/tariff"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("USITC_API_KEY")
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch_tariff_rates(self, hts_codes: List[str]) -> Dict[str, float]:
        """
        Fetches live tariff rates for the specified HTS codes.
        If the API key is missing or the request fails, returns static fallback rates.
        """
        rates = {}
        
        if not self.api_key:
            logger.warning("No USITC_API_KEY found. Falling back to static mock tariff rates.")
            return self._get_fallback_rates(hts_codes)
            
        for hts in hts_codes:
            try:
                # Assuming standard REST endpoints, e.g. /api/v1/tariff/{hts}
                url = f"{self.BASE_URL}/{hts}"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = self.session.get(url, headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                # Assuming JSON payload has a 'rate' field representing the ad valorem equivalent
                rates[hts] = float(data.get("rate", 0.0))
            except Exception as e:
                logger.error(f"Failed to fetch live USITC rate for {hts}: {e}. Using fallback.")
                rates[hts] = self._get_fallback_rates([hts]).get(hts, 0.0)
                
        return rates

    def _get_fallback_rates(self, hts_codes: List[str]) -> Dict[str, float]:
        """Provides static fallback rates for common HTS codes if the API fails."""
        mock_rates = {
            "85171200": 0.15, # Smartphones
            "87032300": 0.025, # Vehicles
            "62034240": 0.166, # Cotton trousers
        }
        return {hts: mock_rates.get(hts, 0.05) for hts in hts_codes} # Default to 5% if unknown
