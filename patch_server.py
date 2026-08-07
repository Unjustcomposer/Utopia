import re

with open("server.py", "r") as f:
    content = f.read()

# Replace the GET / route
mount_code = """from fastapi.staticfiles import StaticFiles
import glob
from scenarios import SCENARIO_LIST

# Try to mount frontend, fallback to dashboard
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/dashboard", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        return DASHBOARD_HTML

@app.get("/api/calibration_profiles")
async def get_calibration_profiles():
    files = glob.glob("data/calibration_profiles/*.json")
    return [os.path.basename(f) for f in files]

@app.get("/api/scenarios")
async def get_scenarios():
    return SCENARIO_LIST

@app.get("/api/runs")
async def get_runs(db: Session = Depends(get_db)):
    runs = db.query(SimulationResult).order_by(SimulationResult.created_at.desc()).limit(50).all()
    return [{"id": r.id, "run_type": r.run_type, "parameters": r.parameters, "created_at": r.created_at} for r in runs]

@app.get("/api/runs/{run_id}")
async def get_run_by_id(run_id: int, db: Session = Depends(get_db)):
    run = db.query(SimulationResult).filter(SimulationResult.id == run_id).first()
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
            firm_behavior_mode=2
        )
        baseline_task = asyncio.to_thread(_ray_run_simulation, config, req.seed, "baseline")
        scenario_task = asyncio.to_thread(_ray_run_simulation, config, req.seed, req.scenario)
        
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
        
        return sanitize_for_json(response)
    except Exception as e:
        logger.exception("Error in handle_api_compare")
        raise HTTPException(status_code=500, detail=str(e))
"""

content = re.sub(r'@app\.get\("/", response_class=HTMLResponse\)\nasync def get_dashboard\(\):\n    return DASHBOARD_HTML', mount_code, content)

# To properly serve root: we need to do app.mount("/", ...) at the end of the file. 
# It's better to remove the existing @app.get("/") and add it at the very bottom so it doesn't shadow API routes.

with open("server.py", "w") as f:
    f.write(content)
