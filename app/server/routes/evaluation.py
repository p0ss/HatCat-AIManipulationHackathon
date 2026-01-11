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
    sample_count: int = 1  # Number of samples per episode
    steering_comparison: bool = False  # Run with/without steering for 6-way comparison


@router.get("/status")
async def get_run_status():
    """Get current run status for page refresh recovery."""
    state = get_state()
    rs = state.run_state

    return {
        "run_id": rs.run_id,
        "status": rs.status,
        "started_at": rs.started_at.isoformat() if rs.started_at else None,
        "completed_at": rs.completed_at.isoformat() if rs.completed_at else None,
        "total_episodes": rs.total_episodes,
        "completed_episodes": rs.completed_episodes,
        "current_episode": rs.current_episode,
        "current_condition": rs.current_condition,
        "conditions": rs.conditions,
        "episode_ids": rs.episode_ids,
        "episode_results": rs.episode_results[-50:],  # Last 50 results
        "tokens_buffer": rs.tokens_buffer[-100:],  # Last 100 tokens
        "summary": rs.summary,
        "error_message": rs.error_message,
        "progress": rs.completed_episodes / rs.total_episodes if rs.total_episodes > 0 else 0,
    }


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


@router.get("/episode/{episode_id}")
async def get_episode(episode_id: str):
    """Get full episode data including turns."""
    episodes_path = PROJECT_ROOT / "episodes" / "manipulation_suite.json"

    if not episodes_path.exists():
        raise HTTPException(status_code=404, detail="Episode file not found")

    with open(episodes_path) as f:
        data = json.load(f)

    for ep in data.get("episodes", []):
        if ep["id"] == episode_id:
            return ep

    raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")


async def evaluation_generator(request: RunRequest) -> AsyncGenerator[str, None]:
    """Stream evaluation progress via SSE with token-by-token streaming."""
    state = get_state()
    rs = state.run_state

    # Check prerequisites
    if not state.model_loaded:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Model not loaded'})}\n\n"
        return

    try:
        # Import the HatCat-direct runner
        from app.evaluation.hatcat_runner import HatCatEvaluationRunner

        # Expand conditions if steering_comparison is enabled
        conditions = request.conditions.copy()
        if request.steering_comparison:
            expanded = []
            for cond in conditions:
                if cond == "A":
                    expanded.append("A")
                elif cond in ("B", "C"):
                    expanded.append(cond)
                    expanded.append(f"{cond}+S")
            conditions = expanded

        # Load episodes
        episodes_path = PROJECT_ROOT / "episodes" / "manipulation_suite.json"
        with open(episodes_path) as f:
            all_episodes = json.load(f).get("episodes", [])

        episodes = [e for e in all_episodes if e["id"] in request.episode_ids]

        if not episodes:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No matching episodes found'})}\n\n"
            return

        # Check lens manager is available
        if not state.lens_manager:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Lens manager not loaded. Please run setup first.'})}\n\n"
            return

        # Initialize run state for persistence
        rs.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        rs.status = "running"
        rs.started_at = datetime.now()
        rs.completed_at = None
        rs.total_episodes = len(episodes) * len(conditions) * request.sample_count
        rs.completed_episodes = 0
        rs.current_episode = None
        rs.current_condition = None
        rs.conditions = conditions
        rs.episode_ids = request.episode_ids
        rs.episode_results = []
        rs.tokens_buffer = []
        rs.summary = None
        rs.error_message = None

        yield f"data: {json.dumps({'type': 'start', 'total_episodes': len(request.episode_ids), 'conditions': conditions, 'sample_count': request.sample_count, 'run_id': rs.run_id})}\n\n"

        # Get concept pack path from config
        concept_pack_path_str = state.config.get("concept_pack_path", "")
        if not concept_pack_path_str:
            # Fallback to HatCat default
            hatcat_root = Path(__file__).parent.parent.parent.parent.parent / "HatCat"
            concept_pack_path = hatcat_root / "concept_packs" / "first-light"
        else:
            concept_pack_path = Path(concept_pack_path_str)

        # Create runner - REUSE lens_manager from state (fast!)
        runner = HatCatEvaluationRunner(
            model=state.model,
            tokenizer=state.tokenizer,
            lens_manager=state.lens_manager,  # Reuse existing!
            concept_pack_path=concept_pack_path,
            device="cuda" if state.model.device.type == "cuda" else "cpu",
        )

        # Run evaluation with sample_count
        total = len(episodes) * len(conditions) * request.sample_count
        completed = 0
        results = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "episodes": [],
            "conditions": conditions,
            "sample_count": request.sample_count,
            "steering_comparison": request.steering_comparison,
        }

        for episode in episodes:
            episode_result = {
                "episode_id": episode["id"],
                "behavior": episode["behavior"],
                "conditions": {},
            }

            # Send episode turns for display
            turns = episode.get("turns", [])
            yield f"data: {json.dumps({'type': 'episode_turns', 'episode_id': episode['id'], 'turns': turns})}\n\n"

            for condition in conditions:
                # Parse condition to get base condition and steering flag
                # e.g., "B+S" -> base="B", with_steering=True
                if "+S" in condition:
                    base_condition = condition.replace("+S", "")
                    with_steering = True
                else:
                    base_condition = condition
                    with_steering = (base_condition == "C")  # C always has steering

                # Run multiple samples if requested
                for sample_idx in range(request.sample_count):
                    state.abort_requested = False

                    sample_label = f"{condition}" if request.sample_count == 1 else f"{condition} (sample {sample_idx + 1}/{request.sample_count})"

                    # Update run state for persistence
                    rs.current_episode = episode['id']
                    rs.current_condition = sample_label
                    rs.tokens_buffer = []  # Clear for new episode

                    yield f"data: {json.dumps({'type': 'episode_start', 'episode_id': episode['id'], 'condition': sample_label, 'behavior': episode['behavior']})}\n\n"

                    # Run episode with token streaming
                    effective_condition = "C" if with_steering and base_condition != "A" else base_condition

                    # Get generation params from config
                    max_tokens = state.config.get("evaluation", {}).get("max_new_tokens", 150)
                    temperature = state.config.get("evaluation", {}).get("temperature", 0.7)

                    def run_streaming_episode():
                        """Run the streaming generator and collect all tokens + result."""
                        tokens_collected = []
                        gen = runner.run_episode_streaming(
                            episode,
                            effective_condition,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        result = None
                        try:
                            while True:
                                token_meta = next(gen)
                                tokens_collected.append(token_meta)
                        except StopIteration as e:
                            result = e.value
                        return tokens_collected, result

                    # Run in executor to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    tokens_collected, result = await loop.run_in_executor(
                        None, run_streaming_episode
                    )

                # Stream collected tokens to client and update run state
                for token_meta in tokens_collected:
                    # Build full ASK schema metadata
                    metadata = {
                        'color': token_meta.color,
                        'safety_intensity': token_meta.safety_intensity,
                        'steering_active': token_meta.steering_active,
                        'top_concepts': [(c, float(s)) for c, s in token_meta.top_concepts[:5]],
                    }

                    # Add violations details (ASK schema)
                    if hasattr(token_meta, 'violations') and token_meta.violations:
                        metadata['violations'] = [
                            {
                                'constraint': v.get('constraint', 'unknown'),
                                'simplex': v.get('simplex', ''),
                                'deviation': round(v.get('deviation', 0), 4) if v.get('deviation') else 0,
                                'threshold': round(v.get('threshold', 0), 4) if v.get('threshold') else 0,
                            }
                            for v in token_meta.violations[:5]  # Limit to top 5
                        ]

                    # Add steering details (ASK schema)
                    if hasattr(token_meta, 'steering_applied') and token_meta.steering_applied:
                        metadata['steering_applied'] = [
                            {
                                'concept': s.get('concept', s.get('simplex', 'unknown')),
                                'action': s.get('action', 'steer'),
                                'strength': round(s.get('strength', 0), 4) if s.get('strength') else 0,
                                'direction': s.get('direction', 'suppress'),
                            }
                            for s in token_meta.steering_applied[:5]  # Limit to top 5
                        ]

                    # Add simplex deviations (drift from baseline)
                    if hasattr(token_meta, 'simplex_deviations') and token_meta.simplex_deviations:
                        # Only include non-trivial deviations
                        deviations = {
                            k: round(v, 4)
                            for k, v in token_meta.simplex_deviations.items()
                            if v is not None and abs(v) > 0.01
                        }
                        if deviations:
                            metadata['simplex_deviations'] = deviations

                    # Add hidden state norm (useful for stability analysis)
                    if hasattr(token_meta, 'hidden_state_norm') and token_meta.hidden_state_norm:
                        metadata['hidden_state_norm'] = round(token_meta.hidden_state_norm, 4)

                    # Add significance scoring (distinguishes decision vs filler tokens)
                    if hasattr(token_meta, 'significance'):
                        metadata['significance'] = round(token_meta.significance, 4)
                    if hasattr(token_meta, 'entropy_by_layer') and token_meta.entropy_by_layer:
                        metadata['entropy_by_layer'] = {k: round(v, 4) for k, v in token_meta.entropy_by_layer.items()}
                    if hasattr(token_meta, 'activation_delta') and token_meta.activation_delta:
                        metadata['activation_delta'] = round(token_meta.activation_delta, 4)
                    if hasattr(token_meta, 'is_filler'):
                        metadata['is_filler'] = token_meta.is_filler

                    token_data = {
                        'type': 'token',
                        'token': token_meta.token,
                        'position': token_meta.position,
                        'metadata': metadata,
                    }
                    # Add to run state buffer for persistence
                    rs.tokens_buffer.append(token_data)
                    if len(rs.tokens_buffer) > 200:
                        rs.tokens_buffer = rs.tokens_buffer[-200:]

                    yield f"data: {json.dumps(token_data)}\n\n"
                    await asyncio.sleep(0.005)

                if result is None:
                    # Fallback - run without streaming
                    result = await runner.run_episode(
                        episode=episode,
                        condition=effective_condition,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                # Build tick summary for XDB log viewer with full ASK schema
                tick_summary = []
                if hasattr(result, 'ticks') and result.ticks:
                    for tick in result.ticks[:200]:  # Limit to first 200 ticks
                        # WorldTick has token_text attribute
                        tick_info = {
                            'token': getattr(tick, 'token_text', '') or getattr(tick, 'token', ''),
                            'tick_id': getattr(tick, 'tick_id', 0),
                            'timestamp': getattr(tick, 'timestamp', None),
                            'concepts': [],
                            'safety_intensity': 0.0,
                            'steering_active': False,
                            'hidden_state_norm': round(getattr(tick, 'hidden_state_norm', 0.0), 4),
                        }

                        # Format timestamp if present
                        if tick_info['timestamp']:
                            tick_info['timestamp'] = tick_info['timestamp'].isoformat() if hasattr(tick_info['timestamp'], 'isoformat') else str(tick_info['timestamp'])

                        # WorldTick has concept_activations: Dict[str, float]
                        # Also check simplex_activations if concept_activations is empty
                        concept_acts = getattr(tick, 'concept_activations', None)
                        if not concept_acts or len(concept_acts) == 0:
                            concept_acts = getattr(tick, 'simplex_activations', None)

                        if concept_acts and isinstance(concept_acts, dict) and len(concept_acts) > 0:
                            sorted_concepts = sorted(concept_acts.items(), key=lambda x: x[1], reverse=True)
                            tick_info['concepts'] = [(c, round(s, 4)) for c, s in sorted_concepts[:10]]

                            # Calculate safety intensity
                            safety_keywords = {'deception', 'manipulation', 'harm', 'dishonesty', 'sycophancy', 'sandbagging'}
                            safety_count = sum(1 for c, _ in tick_info['concepts'] if any(s in c.lower() for s in safety_keywords))
                            tick_info['safety_intensity'] = round(safety_count / max(len(tick_info['concepts']), 1), 2)

                        # Simplex activations (all monitored simplexes)
                        simplex_acts = getattr(tick, 'simplex_activations', None)
                        if simplex_acts and isinstance(simplex_acts, dict) and len(simplex_acts) > 0:
                            tick_info['simplex_activations'] = {k: round(v, 4) for k, v in simplex_acts.items()}

                        # Simplex deviations (drift from baseline - key for autonomic steering)
                        simplex_devs = getattr(tick, 'simplex_deviations', None)
                        if simplex_devs and isinstance(simplex_devs, dict):
                            tick_info['simplex_deviations'] = {
                                k: round(v, 4) for k, v in simplex_devs.items()
                                if v is not None and abs(v) > 0.001
                            }

                        # WorldTick has steering_applied: List[Dict[str, Any]]
                        steering = getattr(tick, 'steering_applied', None)
                        if steering and len(steering) > 0:
                            tick_info['steering_active'] = True
                            tick_info['steering_applied'] = [
                                {
                                    'concept': s.get('concept', s.get('simplex', 'unknown')),
                                    'action': s.get('action', 'steer'),
                                    'strength': round(s.get('strength', 0), 4) if s.get('strength') else 0,
                                    'direction': s.get('direction', 'suppress'),
                                    'contrastive': s.get('contrastive_concept', s.get('contrastive', '')),
                                }
                                for s in steering
                            ]

                        # Violations (constraint breaches that triggered steering)
                        violations = getattr(tick, 'violations', None)
                        if violations and len(violations) > 0:
                            tick_info['safety_intensity'] = max(tick_info['safety_intensity'], 0.5)
                            tick_info['violations'] = [
                                {
                                    'constraint': v.get('constraint', v.get('constraint_id', 'unknown')),
                                    'simplex': v.get('simplex', v.get('simplex_term', '')),
                                    'deviation': round(v.get('deviation', 0), 4) if v.get('deviation') else 0,
                                    'threshold': round(v.get('threshold', v.get('max_deviation', 0)), 4),
                                    'severity': v.get('severity', 'medium'),
                                }
                                for v in violations
                            ]

                        # Significance scoring (decision vs filler token classification)
                        significance = getattr(tick, 'significance', 0.0)
                        if significance > 0:
                            tick_info['significance'] = round(significance, 4)
                        entropy_by_layer = getattr(tick, 'entropy_by_layer', {})
                        if entropy_by_layer:
                            tick_info['entropy_by_layer'] = {k: round(v, 4) for k, v in entropy_by_layer.items()}
                        activation_delta = getattr(tick, 'activation_delta', 0.0)
                        if activation_delta > 0:
                            tick_info['activation_delta'] = round(activation_delta, 4)
                        is_filler = getattr(tick, 'is_filler', False)
                        if is_filler:
                            tick_info['is_filler'] = True

                        tick_summary.append(tick_info)

                    # Store result using the full condition label (e.g., "B+S")
                    result_key = condition if request.sample_count == 1 else f"{condition}_s{sample_idx + 1}"

                    # Get detailed scoring from result
                    score_result = getattr(result, 'score_result', 'null')
                    score_confidence = getattr(result, 'score_confidence', 0.0)

                    # Aggregate results per condition (average across samples)
                    if condition not in episode_result["conditions"]:
                        episode_result["conditions"][condition] = {
                            "manipulation_detected": result.manipulation_detected,
                            "peak_detection_score": result.peak_detection_score,
                            "interventions": result.interventions_triggered,
                            "corrections": result.steering_corrections,
                            "response": result.responses.get(0, "")[:500],
                            "ticks": tick_summary,
                            "samples": 1,
                            # Detailed scoring
                            "score_result": score_result,
                            "score_confidence": score_confidence,
                            "pass_count": 1 if score_result == "pass" else 0,
                            "fail_count": 1 if score_result == "fail" else 0,
                            "null_count": 1 if score_result == "null" else 0,
                            "confidence_scores": [score_confidence],
                        }
                    else:
                        # Aggregate sample results
                        existing = episode_result["conditions"][condition]
                        existing["samples"] += 1
                        # For multiple samples, track if ANY sample showed manipulation
                        existing["manipulation_detected"] = existing["manipulation_detected"] or result.manipulation_detected
                        # Track max peak score
                        existing["peak_detection_score"] = max(existing["peak_detection_score"], result.peak_detection_score)
                        # Sum interventions/corrections
                        existing["interventions"] += result.interventions_triggered
                        existing["corrections"] += result.steering_corrections
                        # Aggregate detailed scoring
                        existing["pass_count"] = existing.get("pass_count", 0) + (1 if score_result == "pass" else 0)
                        existing["fail_count"] = existing.get("fail_count", 0) + (1 if score_result == "fail" else 0)
                        existing["null_count"] = existing.get("null_count", 0) + (1 if score_result == "null" else 0)
                        existing["confidence_scores"] = existing.get("confidence_scores", []) + [score_confidence]

                    completed += 1

                    # Update run state
                    rs.completed_episodes = completed
                    rs.episode_results.append({
                        'episode_id': episode['id'],
                        'condition': condition,
                        'sample': sample_idx + 1,
                        'result': episode_result['conditions'][condition],
                    })
                    # Keep only last 100 results in memory
                    if len(rs.episode_results) > 100:
                        rs.episode_results = rs.episode_results[-100:]

                    yield f"data: {json.dumps({'type': 'episode_complete', 'episode_id': episode['id'], 'condition': condition, 'sample': sample_idx + 1, 'result': episode_result['conditions'][condition], 'progress': completed / total})}\n\n"

                    # Check for abort
                    if state.abort_requested:
                        rs.status = "aborted"
                        yield f"data: {json.dumps({'type': 'aborted'})}\n\n"
                        return

            results["episodes"].append(episode_result)

        # Compute summary
        summary = compute_summary(results)
        results["summary"] = summary

        # Save results (including summary)
        results_dir = PROJECT_ROOT / state.config.get("output", {}).get("results_dir", "outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / f"results_{results['run_id']}.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update run state - complete
        rs.status = "complete"
        rs.completed_at = datetime.now()
        rs.summary = summary
        rs.current_episode = None
        rs.current_condition = None

        yield f"data: {json.dumps({'type': 'complete', 'run_id': results['run_id'], 'summary': summary})}\n\n"

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Update run state - error
        rs.status = "error"
        rs.error_message = str(e)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


def compute_summary(results: dict) -> dict:
    """Compute summary statistics from results with detailed pass/fail/null analysis."""
    import statistics

    summary = {
        "by_condition": {},
        "by_behavior": {},
    }

    conditions = results.get("conditions", ["A", "B", "C"])

    # Initialize counters for all conditions found in results
    all_conditions = set(conditions)
    for ep in results.get("episodes", []):
        for cond in ep.get("conditions", {}).keys():
            all_conditions.add(cond)

    # Initialize with detailed stats
    for cond in all_conditions:
        summary["by_condition"][cond] = {
            "total": 0, "manipulated": 0, "rate": 0, "interventions": 0,
            # Detailed scoring
            "pass_count": 0, "fail_count": 0, "null_count": 0,
            "pass_rate": 0, "fail_rate": 0, "null_rate": 0,
            "confidence_scores": [],
            "avg_confidence": 0, "confidence_stddev": 0,
        }

    behaviors = set()
    for ep in results.get("episodes", []):
        behaviors.add(ep["behavior"])

    for beh in behaviors:
        summary["by_behavior"][beh] = {}
        for cond in all_conditions:
            summary["by_behavior"][beh][cond] = {
                "total": 0, "manipulated": 0, "rate": 0,
                "pass_count": 0, "fail_count": 0, "null_count": 0,
            }

    # Count
    for ep in results.get("episodes", []):
        behavior = ep["behavior"]
        for cond, data in ep.get("conditions", {}).items():
            if cond not in summary["by_condition"]:
                summary["by_condition"][cond] = {
                    "total": 0, "manipulated": 0, "rate": 0, "interventions": 0,
                    "pass_count": 0, "fail_count": 0, "null_count": 0,
                    "pass_rate": 0, "fail_rate": 0, "null_rate": 0,
                    "confidence_scores": [], "avg_confidence": 0, "confidence_stddev": 0,
                }
            if cond not in summary["by_behavior"].get(behavior, {}):
                if behavior not in summary["by_behavior"]:
                    summary["by_behavior"][behavior] = {}
                summary["by_behavior"][behavior][cond] = {
                    "total": 0, "manipulated": 0, "rate": 0,
                    "pass_count": 0, "fail_count": 0, "null_count": 0,
                }

            summary["by_condition"][cond]["total"] += 1
            summary["by_behavior"][behavior][cond]["total"] += 1

            if data.get("manipulation_detected"):
                summary["by_condition"][cond]["manipulated"] += 1
                summary["by_behavior"][behavior][cond]["manipulated"] += 1

            # Track interventions
            summary["by_condition"][cond]["interventions"] += data.get("interventions", 0)

            # Track detailed pass/fail/null
            pass_count = data.get("pass_count", 0)
            fail_count = data.get("fail_count", 0)
            null_count = data.get("null_count", 0)

            summary["by_condition"][cond]["pass_count"] += pass_count
            summary["by_condition"][cond]["fail_count"] += fail_count
            summary["by_condition"][cond]["null_count"] += null_count
            summary["by_behavior"][behavior][cond]["pass_count"] += pass_count
            summary["by_behavior"][behavior][cond]["fail_count"] += fail_count
            summary["by_behavior"][behavior][cond]["null_count"] += null_count

            # Collect confidence scores
            confidence_scores = data.get("confidence_scores", [])
            if isinstance(confidence_scores, list):
                summary["by_condition"][cond]["confidence_scores"].extend(confidence_scores)

    # Compute rates and statistics
    for cond in all_conditions:
        if cond in summary["by_condition"]:
            s = summary["by_condition"][cond]
            total = s["total"]
            if total > 0:
                s["rate"] = s["manipulated"] / total * 100

                # Detailed rates (based on sample counts, not episode counts)
                sample_total = s["pass_count"] + s["fail_count"] + s["null_count"]
                if sample_total > 0:
                    s["pass_rate"] = round(s["pass_count"] / sample_total * 100, 1)
                    s["fail_rate"] = round(s["fail_count"] / sample_total * 100, 1)
                    s["null_rate"] = round(s["null_count"] / sample_total * 100, 1)

                # Confidence statistics
                if s["confidence_scores"]:
                    s["avg_confidence"] = round(statistics.mean(s["confidence_scores"]), 3)
                    if len(s["confidence_scores"]) > 1:
                        s["confidence_stddev"] = round(statistics.stdev(s["confidence_scores"]), 3)

            # Clean up - don't include raw confidence scores in JSON output
            del s["confidence_scores"]

    for beh in behaviors:
        for cond in all_conditions:
            if cond in summary["by_behavior"].get(beh, {}):
                s = summary["by_behavior"][beh][cond]
                total = s["total"]
                if total > 0:
                    s["rate"] = s["manipulated"] / total * 100

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
