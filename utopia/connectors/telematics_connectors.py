import json
import logging
import random
import requests
from typing import Dict, Any, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class NOAA_Weather_Client:
    """Production client for NOAA / OpenWeatherMap probabilistic severe weather APIs."""
    
    def __init__(self, api_key: str = "mock_noaa_key"):
        self.api_key = api_key
        # In a real environment, this connects to the NOAA NHC (National Hurricane Center) API
        self.endpoint_url = "https://api.weather.gov/alerts/active"
        
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
    def get_maritime_weather_alerts(self, region: str = "Trans-Pacific") -> Dict[str, Any]:
        """Fetches active severe weather alerts (e.g., Typhoons) for major shipping lanes."""
        
        if self.api_key == "mock_noaa_key":
            # Mocking a Category 3 Typhoon in the South China Sea
            return {
                "alerts": [
                    {
                        "event": "Typhoon Warning",
                        "severity": "Extreme",
                        "certainty": "Likely",
                        "headline": "Category 3 Typhoon approaching Port of Shenzhen",
                        "probability": 0.85
                    }
                ]
            }
            
        # Real HTTP logic would go here
        resp = self.session.get(self.endpoint_url, timeout=10)
        resp.raise_for_status()
        return resp.json()


class Project44_Telematics_Client:
    """Production client for Project44 / MarineTraffic real-time port congestion API."""
    
    def __init__(self, api_key: str = "mock_p44_key"):
        self.api_key = api_key
        self.endpoint_url = "https://api.project44.com/v1/ocean/ports/congestion"
        
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
    def get_port_congestion_index(self, port_code: str = "CNSZX") -> Dict[str, Any]:
        """Fetches real-time vessel dwelling times and container wait times."""
        
        if self.api_key == "mock_p44_key":
            # CNSZX = Port of Shenzhen
            # Mocking severe congestion due to weather
            return {
                "port": port_code,
                "congestion_level": "Severe",
                "average_vessel_dwell_time_days": 8.5,
                "container_rollover_ratio": 0.35, # 35% of containers rolled to next vessel
                "status": "Degraded"
            }
            
        # Real HTTP logic would go here
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self.session.get(f"{self.endpoint_url}/{port_code}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


class PhysicalShockCompiler:
    """Translates physical real-world events into JAX macroeconomic shock parameters."""
    
    def __init__(self):
        self.weather_client = NOAA_Weather_Client()
        self.telematics_client = Project44_Telematics_Client()
        
    def compile_live_shock(self) -> Tuple[float, str]:
        """
        Polls weather and telematics, calculates a combined severity, 
        and outputs a JAX cost multiplier and a descriptive summary.
        """
        try:
            weather_data = self.weather_client.get_maritime_weather_alerts()
            port_data = self.telematics_client.get_port_congestion_index()
            
            base_multiplier = 1.0
            description = []
            
            # 1. Weather Impact
            weather_alerts = weather_data.get("alerts", [])
            for alert in weather_alerts:
                if alert.get("severity") == "Extreme" and alert.get("probability", 0) > 0.7:
                    base_multiplier += 0.3 # 30% cost spike
                    description.append(f"{alert.get('headline')} ({int(alert.get('probability')*100)}% prob)")
                    
            # 2. Port Congestion Impact
            dwell_time = port_data.get("average_vessel_dwell_time_days", 0)
            rollover = port_data.get("container_rollover_ratio", 0)
            
            if dwell_time > 5.0:
                base_multiplier += (dwell_time - 5.0) * 0.05
                description.append(f"{dwell_time} day dwell time at {port_data.get('port')}")
                
            if rollover > 0.2:
                base_multiplier += rollover * 0.5
                description.append(f"{int(rollover*100)}% container rollover")
                
            summary = " | ".join(description) if description else "Normal Operations"
            
            # Cap the multiplier at 2.5 (150% cost spike) to prevent extreme numerical instability
            final_multiplier = min(base_multiplier, 2.5)
            
            return final_multiplier, summary
            
        except Exception as e:
            logger.error(f"Failed to compile physical shock: {e}")
            # Fallback to no shock
            return 1.0, "Telematics API Unavailable"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compiler = PhysicalShockCompiler()
    multiplier, summary = compiler.compile_live_shock()
    print(f"Physical Shock Compiler Output:")
    print(f"JAX Cost Multiplier: {multiplier:.3f}")
    print(f"Causal Events: {summary}")
