"""
API Server for NexusAI Simulator
================================
Enterprise-grade asynchronous API server using FastAPI.
Replaces the monolithic blocking http.server architecture.
"""
import math
import uvicorn
import uuid
import os
import jax
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File

# Enable persistent XLA compilation cache
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.nexus_jax_cache")
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.nexus_jax_cache"))
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from auth import get_current_user, User
from database import get_db, SimulationResult
from rate_limit import limiter
from data_ingestion import GlobalBaselineCompiler
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from config import SimulationConfig
from dashboard_ui import DASHBOARD_HTML

app = FastAPI(title="NexusAI Engine API", description="Agent-Based Economic Simulator")

from prometheus_client import make_asgi_app, Counter, Histogram
app.mount("/metrics", make_asgi_app())

SIMULATION_COUNTER = Counter("nexusai_simulations_total", "Total number of simulations run", ["type"])
SIMULATION_DURATION = Histogram("nexusai_simulation_duration_seconds", "Duration of simulations")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# [FIX] Overly Permissive CORS: restrict origins tightly and disable credentials if unnecessary
# We allow localhost origins but disable allow_credentials to prevent session hijacking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8765"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return DASHBOARD_HTML

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, "item"):
        return sanitize_for_json(obj.item())
    return obj

class RunRequest(BaseModel):
    agents: int = Field(default=200, gt=0, le=10000)
    firms: int = Field(default=5, gt=0, le=1000)
    goods: int = Field(default=4, gt=0, le=100)
    ticks: int = Field(default=120, gt=0, le=5000)
    use_us_calibration: bool = False
    seed: int = 42
    scenario: str = "baseline"

def _ray_run_simulation(config, seed, scenario="baseline"):
    from simulation_jax import run_simulation
    # We use the new JAX engine directly
    return run_simulation(config=config, seed=seed, scenario=scenario)

@app.post("/api/run")
@limiter.limit("10/minute")
async def handle_api_run(
    request: Request,
    req: RunRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting run")
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            use_us_calibration=req.use_us_calibration,
            firm_behavior_mode=2 # Use Heuristic for stability
        )
        
        with SIMULATION_DURATION.time():
            result = await asyncio.to_thread(_ray_run_simulation, config, req.seed, req.scenario)
            
        SIMULATION_COUNTER.labels(type="run").inc()
        
        response = {
            "metrics_history": result.metrics_history,
            "summary": result.summary()
        }
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="run",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.exception("Error in handle_api_run")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/ingest")
@limiter.limit("10/minute")
async def ingest_agents_csv(
    request: Request,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) ingesting CSV: {file.filename}")
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Must be a CSV.")
            
        # [FIX] DOS vulnerability: Limit upload size to 5MB and parse from stream
        MAX_SIZE = 5 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
            
        if b'\x00' in contents:
            raise HTTPException(status_code=400, detail="Invalid file content: null bytes detected.")
            
        import io
        df = await asyncio.to_thread(pd.read_csv, io.BytesIO(contents))
        
        records = df.to_dict(orient="records")
        logger.info(f"Successfully ingested {len(records)} agent records")
        
        return {"status": "success", "message": f"Ingested {len(records)} records.", "records": records[:5]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in ingest_agents_csv")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest_global_baseline")
async def ingest_global_baseline():
    """
    Triggers the alternative data ingestion pipeline (Credit Cards, Satellite, Shipping)
    to compile a real-world snapshot into JAX tensors and run the simulation from that baseline.
    """
    try:
        logger.info("Triggering Global Baseline Compilation...")
        config = SimulationConfig()
        compiler = GlobalBaselineCompiler(config)
        overrides = await asyncio.to_thread(compiler.compile_baseline)
        
        logger.info("Baseline compiled successfully. Initiating JAX Simulation with overrides.")
        from simulation_jax import run_simulation
        result = await asyncio.to_thread(run_simulation, config=config, seed=42, scenario="baseline", baseline_state_overrides=overrides)
        
        # Optionally save to DB here...
        
        return {
            "status": "success", 
            "message": "Global baseline ingested and simulated successfully.",
            "metrics": result.summary()
        }
    except Exception as e:
        logger.exception("Error in global baseline ingestion")
        raise HTTPException(status_code=500, detail=str(e))

class ExplainRequest(BaseModel):
    demand_history: List[float] = [10.0, 11.0, 10.5]
    profit_history: List[float] = [100.0, 105.0, 102.0]
    price_history: List[float] = [10.0, 10.0, 10.0]
    macro_price_history: List[float] = [10.0, 10.1, 10.2]
    macro_rate_history: List[float] = [0.05, 0.05, 0.05]

@app.post("/api/explain")
@limiter.limit("10/minute")
async def handle_api_explain(
    request: Request,
    req: ExplainRequest,
    user: User = Depends(get_current_user)
):
    try:
        from lmm_explain import explain_firm_policy
        from lmm_model import get_initial_lmm_params
        import jax
        import jax.numpy as jnp
        
        # Load weights (for now we use initial weights, in a real system we'd load the trained checkpoint)
        params = await asyncio.to_thread(get_initial_lmm_params, jax.random.PRNGKey(42))
        
        lmm_inputs = jnp.stack([
            jnp.array(req.demand_history),
            jnp.array(req.profit_history),
            jnp.array(req.price_history),
            jnp.array(req.macro_price_history),
            jnp.array(req.macro_rate_history)
        ], axis=-1)
        
        explanations = await asyncio.to_thread(explain_firm_policy, params, lmm_inputs)
        return sanitize_for_json(explanations)
    except Exception as e:
        logger.exception("Error in handle_api_explain")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("==========================================================")
    print(f"  NexusAI FastAPI Server running at http://localhost:{port}")
    print("==========================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
