"""
API Server for NexusAI Simulator
================================
Enterprise-grade asynchronous API server using FastAPI.
Replaces the monolithic blocking http.server architecture.
Now integrated with Ray for distributed simulation.
"""
import math
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from auth import get_current_user, User
from database import get_db, SimulationResult
from rate_limit import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from config import SimulationConfig
from simulation import Simulation
from experiment import Experiment
from scenario import create_scenario
from search import StrategySearch, firm_profit_objective, total_welfare_objective, price_stability_objective
from dashboard_ui import DASHBOARD_HTML

app = FastAPI(title="NexusAI Engine API", description="Agent-Based Economic Simulator")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8765"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    if False:
        ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized for distributed simulation.")

@app.on_event("shutdown")
async def shutdown_event():
    if False:
        ray.shutdown()
    logger.info("Ray shut down.")

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

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return DASHBOARD_HTML

class RunRequest(BaseModel):
    agents: int = 200
    firms: int = 5
    goods: int = 4
    ticks: int = 120
    use_us_calibration: bool = False
    seed: int = 42

def _ray_run_simulation(config, seed):
    sim = Simulation(config=config, seed=seed)
    return sim.run()

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
            use_us_calibration=req.use_us_calibration
        )
        result = await _ray_run_simulation(config, req.seed)
        
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
        logger.error(f"Error in run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ExperimentRequest(BaseModel):
    agents: int = 200
    firms: int = 5
    goods: int = 4
    ticks: int = 90
    use_us_calibration: bool = False
    seed: int = 42
    seeds: int = 10
    scenario_type: str = "marketing"
    scenario_params: Dict[str, Any] = {}

def _ray_run_experiment(config, scenario, seeds, base_seed):
    exp = Experiment(config=config, scenario=scenario, num_seeds=seeds, base_seed=base_seed)
    return exp.run()

@app.post("/api/experiment")
@limiter.limit("5/minute")
async def handle_api_experiment(
    request: Request,
    req: ExperimentRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting experiment")
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            use_us_calibration=req.use_us_calibration
        )
        scenario = create_scenario(req.scenario_type, **req.scenario_params)
        result = await _ray_run_experiment(config, scenario, req.seeds, req.seed)
        
        avg_control = [{k: float(v) for k, v in tick.items()} for tick in result.avg_control_metrics]
        avg_treatment = [{k: float(v) for k, v in tick.items()} for tick in result.avg_treatment_metrics]
        
        deltas = {}
        for metric, md in result.metric_deltas.items():
            deltas[metric] = {
                "mean_delta": md.mean_delta,
                "ci_lower": md.ci_lower,
                "ci_upper": md.ci_upper,
                "effect_size": md.effect_size,
                "p_value": md.p_value,
                "significant": md.significant,
            }

        response = {
            "control_metrics": avg_control,
            "treatment_metrics": avg_treatment,
            "deltas": deltas,
            "scenario": result.scenario_description,
            "num_seeds": result.num_seeds,
            "runtime": result.runtime_seconds,
        }
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="experiment",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SearchRequest(BaseModel):
    agents: int = 150
    firms: int = 5
    goods: int = 4
    ticks: int = 90
    use_us_calibration: bool = False
    scenario_type: str = "feature_change"
    scenario_params: Dict[str, Any] = {}
    param_space: Optional[Dict[str, Any]] = None
    target_region: str = "All"
    target_age_group: str = "All"
    objective: str = "profit"
    method: str = "grid"
    seeds_per_eval: int = 5
    validation_seeds: int = 10
    top_k: int = 3

def _ray_run_search(config, scenario_factory, param_space, objective, method, seeds_per_eval, validation_seeds, top_k):
    search = StrategySearch(
        config=config,
        scenario_factory=scenario_factory,
        param_space=param_space,
        objective=objective,
        method=method,
        num_seeds_per_eval=seeds_per_eval,
        validation_num_seeds=validation_seeds,
        top_k_validate=top_k,
    )
    return search.run()

@app.post("/api/search")
@limiter.limit("2/minute")
async def handle_api_search(
    request: Request,
    req: SearchRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting search")
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            use_us_calibration=req.use_us_calibration
        )
        
        default_param_space = {}
        if req.scenario_type == "marketing":
            default_param_space = {"spend": [1000, 5000, 10000], "reach": [0.2, 0.5, 0.8]}
        elif req.scenario_type == "demand_shock":
            default_param_space = {"risk_aversion_delta": [0.05, 0.1, 0.2]}
        elif req.scenario_type == "supply_disruption":
            default_param_space = {"capacity_reduction": [0.2, 0.5, 0.8]}
        elif req.scenario_type == "trade_disruption":
            default_param_space = {"cost_increase": [1.2, 1.5, 2.0]}
        else:
            default_param_space = {"price": [8, 10, 12, 14, 16, 18, 20]}

        param_space = req.param_space if req.param_space else default_param_space

        def scenario_factory(params):
            sp = req.scenario_params.copy()
            sp.update(params)
            sp["target_region"] = req.target_region
            sp["target_age_group"] = req.target_age_group
            if "start_tick" not in sp: sp["start_tick"] = 15
            if "duration" not in sp: sp["duration"] = 50
            
            if req.scenario_type == "feature_change":
                if "target_good" not in sp: sp["target_good"] = 0
                if "price" in sp: sp["new_price"] = sp.pop("price")
            
            return create_scenario(req.scenario_type, **sp)
        
        if req.objective == "profit":
            objective = firm_profit_objective(firm_id=0)
        elif req.objective == "price_stability":
            objective = price_stability_objective()
        else:
            objective = total_welfare_objective()

        result = await _ray_run_search(
            config, scenario_factory, param_space, objective, req.method,
            req.seeds_per_eval, req.validation_seeds, req.top_k
        )
        
        candidates = []
        for c in result.candidates[:20]:
            candidates.append({
                "rank": c.rank,
                "params": c.params,
                "objective_mean": c.objective_mean,
                "ci_lower": c.objective_ci_lower,
                "ci_upper": c.objective_ci_upper,
            })

        robustness = []
        for rc in result.robustness_checks:
            robustness.append({
                "params": rc.params,
                "search_mean": rc.search_mean,
                "validation_mean": rc.validation_mean,
                "degradation_ratio": rc.degradation_ratio,
                "robustness_score": rc.robustness_score,
                "is_overfit": rc.is_overfit,
                "ks_p_value": rc.ks_p_value,
            })
            
        response = {
            "candidates": candidates,
            "robustness_checks": robustness,
            "objective": result.objective_name,
            "method": result.method,
            "runtime": result.runtime_seconds,
        }
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="search",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SensitivityRequest(BaseModel):
    agents: int = 150
    firms: int = 5
    goods: int = 4
    ticks: int = 60
    use_us_calibration: bool = False
    objective: str = "profit"
    samples: int = 64
    seed: int = 42

def _ray_run_sensitivity(config, param_space, objective_name, samples, seed):
    from sensitivity import run_sensitivity
    return run_sensitivity(
        config=config,
        param_space=param_space,
        objective_name=objective_name,
        num_samples=samples,
        base_seed=seed
    )

@app.post("/api/sensitivity")
@limiter.limit("2/minute")
async def handle_api_sensitivity(
    request: Request,
    req: SensitivityRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting sensitivity")
        
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks,
            use_us_calibration=req.use_us_calibration
        )
        
        param_space = {
            "price": (8.0, 20.0),
            "quality": (0.7, 1.3),
            "ces_elasticity": (0.5, 3.0),
            "bargaining_power_agent": (0.1, 0.9),
            "matching_efficiency": (0.2, 0.8),
        }
        
        Si = await _ray_run_sensitivity(
            config, param_space, req.objective, req.samples, req.seed
        )
        
        results = []
        for i, name in enumerate(param_space.keys()):
            results.append({
                "parameter": name,
                "S1": float(Si["S1"][i]),
                "ST": float(Si["ST"][i]),
            })
            
        response = {"sensitivity": results, "objective": req.objective, "samples": req.samples}
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="sensitivity",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()

        return response
    except Exception as e:
        logger.error(f"Error in sensitivity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CalibrateRequest(BaseModel):
    agents: int = 150
    firms: int = 5
    goods: int = 4
    ticks: int = 60
    trials: int = 20
    seed: int = 42
    targets: Dict[str, float] = {"unemployment": 0.045, "gini": 0.48}
    param_space: Dict[str, Tuple[float, float]] = {
        "sigma_wealth": (0.4, 1.5),
        "sigma_wage": (0.2, 0.8),
        "savings_mean": (0.01, 0.15),
        "savings_std": (0.01, 0.08)
    }

def _ray_run_calibration(config, targets, param_space, trials, seed):
    from calibrate import run_calibration
    return run_calibration(
        config=config,
        target_macro_metrics=targets,
        param_space=param_space,
        trials=trials,
        base_seed=seed
    )

@app.post("/api/calibrate")
@limiter.limit("2/minute")
async def handle_api_calibrate(
    request: Request,
    req: CalibrateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting calibrate")
        
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks
        )
        
        best_params, best_mse = await _ray_run_calibration(
            config, req.targets, req.param_space, req.trials, req.seed
        )
        
        response = {"best_params": best_params, "best_mse": best_mse}
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="calibrate",
            parameters=req.model_dump(),
            results=sanitize_for_json(response)
        )
        db.add(db_result)
        db.commit()

        return response
    except Exception as e:
        logger.error(f"Error in calibrate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GradSearchRequest(BaseModel):
    agents: int = 150
    firms: int = 5
    goods: int = 4
    ticks: int = 60
    iters: int = 50
    seed: int = 42

def _ray_run_gradient_search(config, iters, seed):
    from gradient_search import run_gradient_search
    return run_gradient_search(config=config, iters=iters, seed=seed)

@app.post("/api/grad-search")
@limiter.limit("2/minute")
async def handle_api_grad_search(
    request: Request,
    req: GradSearchRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"User {user.username} (tenant: {user.tenant_id}) starting grad-search")
        config = SimulationConfig(
            num_agents=req.agents,
            num_firms=req.firms,
            num_goods=req.goods,
            num_ticks=req.ticks
        )
        result = await _ray_run_gradient_search(config, req.iters, req.seed)
        
        db_result = SimulationResult(
            tenant_id=user.tenant_id,
            run_type="grad-search",
            parameters=req.model_dump(),
            results=sanitize_for_json(result)
        )
        db.add(db_result)
        db.commit()

        return result
    except Exception as e:
        logger.error(f"Error in grad-search: {e}")
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
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Invalid file type. Must be a CSV.")
        df = pd.read_csv(file.file)
        
        records = df.to_dict(orient="records")
        logger.info(f"Successfully ingested {len(records)} agent records")
        
        return {"status": "success", "message": f"Ingested {len(records)} records.", "records": records[:5]}
    except Exception as e:
        logger.error(f"Error in ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("==========================================================")
    print(f"  NexusAI FastAPI Server running at http://localhost:{port}")
    print("==========================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
