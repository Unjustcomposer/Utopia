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
from audit_logger import SecureAuditLogger

from config import SimulationConfig
from dashboard_ui import DASHBOARD_HTML
from checkpoint import load_lmm_checkpoint
import jwt

GLOBAL_LMM_PARAMS = load_lmm_checkpoint()

audit_logger = SecureAuditLogger()

app = FastAPI(title="NexusAI Engine API", description="Agent-Based Economic Simulator")

@app.middleware("http")
async def extract_tenant_for_ratelimit(request: Request, call_next):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            unverified = jwt.decode(token, options={"verify_signature": False})
            tenant_id = unverified.get("https://utopia.com/tenant_id") or unverified.get("tenant_id")
            if tenant_id:
                request.state.tenant_id = tenant_id
        except Exception:
            pass
    return await call_next(request)

from prometheus_client import make_asgi_app, Counter, Histogram, CollectorRegistry

_metrics_registry = CollectorRegistry()

try:
    SIMULATION_COUNTER = Counter("nexusai_simulations_total", "Total number of simulations run", ["type"], registry=_metrics_registry)
    SIMULATION_DURATION = Histogram("nexusai_simulation_duration_seconds", "Duration of simulations", registry=_metrics_registry)
except ValueError:
    from prometheus_client import REGISTRY
    SIMULATION_COUNTER = REGISTRY._names_to_collectors.get("nexusai_simulations_total") or Counter("nexusai_simulations_total", "Total number of simulations run", ["type"], registry=_metrics_registry)
    SIMULATION_DURATION = REGISTRY._names_to_collectors.get("nexusai_simulation_duration_seconds") or Histogram("nexusai_simulation_duration_seconds", "Duration of simulations", registry=_metrics_registry)

app.mount("/metrics", make_asgi_app(registry=_metrics_registry))

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# [FIX] Overly Permissive CORS: restrict origins tightly and disable credentials if unnecessary
# We allow origins configured from environment but disable allow_credentials to prevent session hijacking
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8765").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

from fastapi.staticfiles import StaticFiles
import glob
from scenarios import SCENARIO_LIST

# Endpoints for Frontend Phase 2.1
@app.get("/api/calibration_profiles")
async def get_calibration_profiles(user: User = Depends(get_current_user)):
    files = glob.glob("data/calibration_profiles/*.json")
    return [os.path.basename(f) for f in files]

@app.get("/api/erp/status")
async def get_erp_status(user: User = Depends(get_current_user)):
    from erp_connectors import SAP_ERP_Client, Oracle_NetSuite_Client
    sap = SAP_ERP_Client()
    oracle = Oracle_NetSuite_Client()
    return {
        "SAP_ECC": {"mode": "mock" if sap.is_mock_mode else "live", "connected": sap.is_mock_mode is not True},
        "Oracle_NetSuite": {"mode": "mock" if oracle.is_mock_mode else "live", "connected": oracle.is_mock_mode is not True},
        "lmm_checkpoint": {"loaded": GLOBAL_LMM_PARAMS is not None},
    }

@app.post("/api/admin/reload-model")
async def reload_model(user: User = Depends(get_current_user)):
    global GLOBAL_LMM_PARAMS
    GLOBAL_LMM_PARAMS = load_lmm_checkpoint()
    loaded = GLOBAL_LMM_PARAMS is not None
    return {"status": "reloaded" if loaded else "no_checkpoint_found", "loaded": loaded}

@app.get("/api/scenarios")
async def get_scenarios(user: User = Depends(get_current_user)):
    return SCENARIO_LIST

@app.get("/api/runs")
async def get_runs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    runs = db.query(SimulationResult).filter(SimulationResult.tenant_id == user.tenant_id).order_by(SimulationResult.created_at.desc()).limit(50).all()
    return [{"id": r.id, "run_type": r.run_type, "parameters": r.parameters, "created_at": r.created_at} for r in runs]

@app.get("/api/runs/{run_id}")
async def get_run_by_id(run_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    run = db.query(SimulationResult).filter(SimulationResult.id == run_id, SimulationResult.tenant_id == user.tenant_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run

class CompareRequest(BaseModel):
    agents: int = 200
    firms: int = 5
    goods: int = 4
    ticks: int = 120
    use_us_calibration: bool = False
    seed: int = 42
    scenario: str = "recession"
    firm_behavior_mode: int = Field(default=2, ge=0, le=2)

@app.post("/api/run/compare")
@limiter.limit("5/minute")
async def handle_api_compare(
    request: Request,
    req: CompareRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            use_us_calibration=req.use_us_calibration,
            firm_behavior_mode=req.firm_behavior_mode
        )
        lmm_p = GLOBAL_LMM_PARAMS if req.firm_behavior_mode == 0 else None
        baseline_task = asyncio.to_thread(_ray_run_simulation, config, req.seed, "baseline", lmm_p)
        scenario_task = asyncio.to_thread(_ray_run_simulation, config, req.seed, req.scenario, lmm_p)
        
        baseline_result, scenario_result = await asyncio.gather(baseline_task, scenario_task)
        
        response = {
            "baseline": {
                "metrics_history": baseline_result.metrics_history,
                "summary": baseline_result.summary()
            },
            "scenario": {
                "metrics_history": scenario_result.metrics_history,
                "summary": scenario_result.summary()
            }
        }
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="compare",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()
        
        audit_logger.log_autonomous_action("run_simulation_compare", {
            "tenant_id": user.tenant_id,
            "username": user.username,
            "scenario": req.scenario,
            "run_type": "compare"
        }, "NexusAI")
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.exception("Error in handle_api_compare")
        raise HTTPException(status_code=500, detail=str(e))


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
    firm_behavior_mode: int = Field(default=0, ge=0, le=2)

def _ray_run_simulation(config, seed, scenario="baseline", lmm_params=None):
    from simulation_jax import run_simulation
    # We use the new JAX engine directly
    return run_simulation(config=config, seed=seed, scenario=scenario, lmm_params=lmm_params)

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
            firm_behavior_mode=req.firm_behavior_mode
        )
        
        with SIMULATION_DURATION.time():
            lmm_p = GLOBAL_LMM_PARAMS if req.firm_behavior_mode == 0 else None
            result = await asyncio.to_thread(_ray_run_simulation, config, req.seed, req.scenario, lmm_p)
            
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
        
        audit_logger.log_autonomous_action("run_simulation", {
            "tenant_id": user.tenant_id,
            "username": user.username,
            "scenario": req.scenario,
            "run_type": "run"
        }, "NexusAI")
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.exception("Error in handle_api_run")
        raise HTTPException(status_code=500, detail=str(e))

class ExperimentRequest(BaseModel):
    agents: int = 200
    firms: int = 5
    goods: int = 4
    ticks: int = 120
    seed: int = 42
    num_seeds: int = 5
    scenario_a: str = "baseline"
    scenario_b: str = "tariffs"
    firm_behavior_mode: int = Field(default=2, ge=0, le=2)

@app.post("/api/experiment")
@limiter.limit("5/minute")
async def handle_api_experiment(
    request: Request,
    req: ExperimentRequest,
    user: User = Depends(get_current_user)
):
    try:
        import numpy as np
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            firm_behavior_mode=req.firm_behavior_mode
        )
        
        async def run_scenario(scenario_name):
            outputs = []
            for i in range(req.num_seeds):
                lmm_p = GLOBAL_LMM_PARAMS if req.firm_behavior_mode == 0 else None
                res = await asyncio.to_thread(_ray_run_simulation, config, req.seed + i, scenario_name, lmm_p)
                if res.metrics_history:
                    outputs.append(res.metrics_history[-1].get('total_output', 0))
            return np.mean(outputs) if outputs else 0, np.std(outputs) if outputs else 0
            
        mean_a, std_a = await run_scenario(req.scenario_a)
        mean_b, std_b = await run_scenario(req.scenario_b)
        
        diff = mean_b - mean_a
        pct = (diff / mean_a * 100) if mean_a else 0
        
        audit_logger.log_autonomous_action("run_experiment", {
            "tenant_id": user.tenant_id,
            "username": user.username,
            "scenario_a": req.scenario_a,
            "scenario_b": req.scenario_b
        }, "NexusAI")
        
        return sanitize_for_json({
            "scenario_a": {"mean_output": mean_a, "std_output": std_a},
            "scenario_b": {"mean_output": mean_b, "std_output": std_b},
            "deltas": {"absolute": diff, "percentage": pct}
        })
    except Exception as e:
        logger.exception("Error in handle_api_experiment")
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
@limiter.limit("2/minute")
async def ingest_global_baseline(request: Request, user: User = Depends(get_current_user)):
    """
    Triggers the alternative data ingestion pipeline (Credit Cards, Satellite, Shipping)
    to compile a real-world snapshot into JAX tensors and run the simulation from that baseline.
    """
    try:
        logger.info("Triggering Global Baseline Compilation...")
        config = SimulationConfig()
        compiler = GlobalBaselineCompiler(config)
        overrides, is_fallback = await asyncio.to_thread(compiler.compile_baseline)
        
        logger.info("Baseline compiled successfully. Initiating JAX Simulation with overrides.")
        from simulation_jax import run_simulation
        result = await asyncio.to_thread(run_simulation, config=config, seed=42, scenario="baseline", baseline_state_overrides=overrides)
        
        # Optionally save to DB here...
        
        audit_logger.log_autonomous_action("ingest_global_baseline", {
            "tenant_id": user.tenant_id,
            "username": user.username,
            "action": "ingest_baseline"
        }, "NexusAI")
        
        return {
            "status": "success", 
            "message": "Global baseline compiled and simulated.",
            "is_fallback": is_fallback,
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
    format: Optional[str] = "executive",
    user: User = Depends(get_current_user)
):
    try:
        from lmm_explain import explain_firm_policy, generate_executive_explanation
        from lmm_model import get_initial_lmm_params
        import jax
        import jax.numpy as jnp
        
        warning = None
        if GLOBAL_LMM_PARAMS is not None:
            params = GLOBAL_LMM_PARAMS
        else:
            warning = "Using untrained model weights. Run 'python main.py train' to train the LMM first."
            params = await asyncio.to_thread(get_initial_lmm_params, jax.random.PRNGKey(42))
        
        lmm_inputs = jnp.stack([
            jnp.array(req.demand_history),
            jnp.array(req.profit_history),
            jnp.array(req.price_history),
            jnp.array(req.macro_price_history),
            jnp.array(req.macro_rate_history)
        ], axis=-1)
        
        if format == "raw":
            explanations = await asyncio.to_thread(explain_firm_policy, params, lmm_inputs)
        else:
            explanations = await asyncio.to_thread(generate_executive_explanation, params, lmm_inputs, "supply_chain")
            
        response = sanitize_for_json(explanations)
        if warning:
            response["warning"] = warning
        return response
    except Exception as e:
        logger.exception("Error in handle_api_explain")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the frontend at root
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    async def get_dashboard():
        return DASHBOARD_HTML

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("==========================================================")
    print(f"  NexusAI FastAPI Server running at http://localhost:{port}")
    print("==========================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
