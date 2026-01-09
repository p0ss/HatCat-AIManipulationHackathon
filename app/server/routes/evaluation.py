"""
Evaluation Routes - Run A/B/C comparison evaluations.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_state():
    from app.server.app import get_state
    return get_state()


class RunRequest(BaseModel):
    episode_ids: List[str]
    conditions: List[str] = ["A", "B", "C"]


@router.get("/episodes")
async def list_episodes():
    """List available evaluation episodes."""
    episodes_path = PROJECT_ROOT / "episodes" / "manipulation_suite.json"

    if not episodes_path.exists():
        return {"episodes": [], "error": "Episode file not found"}

    with open(episodes_path) as f:
        data = json.load(f)

    episodes = []
    for ep in data.get("episodes", []):
        episodes.append({
            "id": ep["id"],
            "behavior": ep["behavior"],
            "description": ep.get("description", ""),
            "difficulty": ep.get("difficulty", "medium"),
        })

    return {"episodes": episodes}


async def evaluation_generator(request: RunRequest) -> AsyncGenerator[str, None]:
    """Stream evaluation progress via SSE."""
    state = get_state()

    # Check prerequisites
    if not state.model_loaded:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Model not loaded'})}\n\n"
        return

    try:
        # Import evaluation runner
        from app.evaluation.runner import ABCEvaluationRunner

        yield f"data: {json.dumps({'type': 'start', 'total_episodes': len(request.episode_ids), 'conditions': request.conditions})}\n\n"

        # Load episodes
        episodes_path = PROJECT_ROOT / "episodes" / "manipulation_suite.json"
        with open(episodes_path) as f:
            all_episodes = json.load(f).get("episodes", [])

        # Filter to requested episodes
        episodes = [e for e in all_episodes if e["id"] in request.episode_ids]

        if not episodes:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No matching episodes found'})}\n\n"
            return

        # Create runner
        runner = ABCEvaluationRunner(
            model=state.model,
            tokenizer=state.tokenizer,
            lens_manager=state.lens_manager,
            config=state.config,
        )

        # Progress callback for SSE
        async def progress_callback(event_type: str, data: dict):
            yield f"data: {json.dumps({'type': event_type, **data})}\n\n"

        # Run evaluation
        total = len(episodes) * len(request.conditions)
        completed = 0
        results = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "episodes": [],
            "conditions": request.conditions,
        }

        for episode in episodes:
            episode_result = {
                "episode_id": episode["id"],
                "behavior": episode["behavior"],
                "conditions": {},
            }

            for condition in request.conditions:
                state.abort_requested = False

                yield f"data: {json.dumps({'type': 'episode_start', 'episode_id': episode['id'], 'condition': condition, 'behavior': episode['behavior']})}\n\n"

                # Run episode
                result = await runner.run_episode(
                    episode=episode,
                    condition=condition,
                )

                episode_result["conditions"][condition] = {
                    "manipulation_detected": result.manipulation_detected,
                    "peak_detection_score": result.peak_detection_score,
                    "interventions": result.interventions_triggered,
                    "corrections": result.steering_corrections,
                }

                completed += 1
                yield f"data: {json.dumps({'type': 'episode_complete', 'episode_id': episode['id'], 'condition': condition, 'result': episode_result['conditions'][condition], 'progress': completed / total})}\n\n"

                # Check for abort
                if state.abort_requested:
                    yield f"data: {json.dumps({'type': 'aborted'})}\n\n"
                    return

            results["episodes"].append(episode_result)

        # Save results
        results_dir = PROJECT_ROOT / state.config.get("output", {}).get("results_dir", "outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"results_{results['run_id']}.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Compute summary
        summary = compute_summary(results)
        results["summary"] = summary

        yield f"data: {json.dumps({'type': 'complete', 'run_id': results['run_id'], 'summary': summary})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def compute_summary(results: dict) -> dict:
    """Compute summary statistics from results."""
    summary = {
        "by_condition": {},
        "by_behavior": {},
    }

    conditions = results.get("conditions", ["A", "B", "C"])

    # Initialize counters
    for cond in conditions:
        summary["by_condition"][cond] = {"total": 0, "manipulated": 0, "rate": 0}

    behaviors = set()
    for ep in results.get("episodes", []):
        behaviors.add(ep["behavior"])

    for beh in behaviors:
        summary["by_behavior"][beh] = {}
        for cond in conditions:
            summary["by_behavior"][beh][cond] = {"total": 0, "manipulated": 0, "rate": 0}

    # Count
    for ep in results.get("episodes", []):
        behavior = ep["behavior"]
        for cond, data in ep.get("conditions", {}).items():
            summary["by_condition"][cond]["total"] += 1
            summary["by_behavior"][behavior][cond]["total"] += 1

            if data.get("manipulation_detected"):
                summary["by_condition"][cond]["manipulated"] += 1
                summary["by_behavior"][behavior][cond]["manipulated"] += 1

    # Compute rates
    for cond in conditions:
        total = summary["by_condition"][cond]["total"]
        if total > 0:
            summary["by_condition"][cond]["rate"] = (
                summary["by_condition"][cond]["manipulated"] / total * 100
            )

    for beh in behaviors:
        for cond in conditions:
            total = summary["by_behavior"][beh][cond]["total"]
            if total > 0:
                summary["by_behavior"][beh][cond]["rate"] = (
                    summary["by_behavior"][beh][cond]["manipulated"] / total * 100
                )

    return summary


@router.post("/run")
async def run_evaluation(request: RunRequest):
    """Run evaluation with SSE progress streaming."""
    return StreamingResponse(
        evaluation_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/abort")
async def abort_evaluation():
    """Abort the current evaluation run."""
    state = get_state()
    state.abort_requested = True
    return {"message": "Abort requested"}
