"""
Results Routes - Retrieve results and generate charts.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_state():
    from app.server.app import get_state
    return get_state()


def get_results_dir() -> Path:
    state = get_state()
    return PROJECT_ROOT / state.config.get("output", {}).get("results_dir", "outputs/results")


@router.get("/list")
async def list_results():
    """List all available result files."""
    results_dir = get_results_dir()

    if not results_dir.exists():
        return {"results": []}

    results = []
    for path in sorted(results_dir.glob("results_*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                results.append({
                    "run_id": data.get("run_id", path.stem),
                    "filename": path.name,
                    "episode_count": len(data.get("episodes", [])),
                    "conditions": data.get("conditions", []),
                    "suite": data.get("suite"),
                })
        except Exception:
            pass

    return {"results": results}


@router.get("/latest")
async def get_latest_results():
    """Get the most recent results."""
    results_dir = get_results_dir()

    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="No results found")

    result_files = sorted(results_dir.glob("results_*.json"), reverse=True)
    if not result_files:
        raise HTTPException(status_code=404, detail="No results found")

    with open(result_files[0]) as f:
        return json.load(f)


@router.get("/{run_id}")
async def get_results(run_id: str):
    """Get results for a specific run."""
    results_dir = get_results_dir()
    results_path = results_dir / f"results_{run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run {run_id}")

    with open(results_path) as f:
        return json.load(f)


@router.get("/{run_id}/chart/comparison")
async def get_comparison_chart(run_id: str):
    """Get Plotly JSON for A/B/C comparison chart."""
    results_dir = get_results_dir()
    results_path = results_dir / f"results_{run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run {run_id}")

    with open(results_path) as f:
        results = json.load(f)

    from app.visualization.charts import create_comparison_chart
    return create_comparison_chart(results)


@router.get("/{run_id}/chart/interventions")
async def get_intervention_chart(run_id: str):
    """Get Plotly JSON for intervention effectiveness chart."""
    results_dir = get_results_dir()
    results_path = results_dir / f"results_{run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run {run_id}")

    with open(results_path) as f:
        results = json.load(f)

    from app.visualization.charts import create_intervention_chart
    return create_intervention_chart(results)


@router.get("/{run_id}/responses/{episode_id}/{condition}")
async def get_episode_responses(run_id: str, episode_id: str, condition: str):
    """Get actual model responses for an episode under a condition."""
    results_dir = get_results_dir()
    results_path = results_dir / f"results_{run_id}.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results not found for run {run_id}")

    with open(results_path) as f:
        results = json.load(f)

    for ep in results.get("episodes", []):
        if ep["episode_id"] == episode_id:
            cond_data = ep.get("conditions", {}).get(condition)
            if cond_data:
                return {
                    "episode_id": episode_id,
                    "condition": condition,
                    "data": cond_data,
                }

    raise HTTPException(status_code=404, detail="Episode/condition not found")
