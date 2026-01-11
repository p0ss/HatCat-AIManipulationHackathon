"""
Calibration and Statistical Estimation Routes.

Provides endpoints for lens pack stability metrics and calibration results.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from typing import Optional

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

CALIBRATION_DIR = Path("/home/poss/Documents/Code/AIManipulationHackathon/outputs/calibration")


@router.get("/summary")
async def get_calibration_summary():
    """
    Get the latest calibration validation summary.

    Returns statistical estimation metrics including:
    - Diagonal rank statistics (how well concepts self-identify)
    - Top-k Jaccard stability (structural consistency across probes)
    - Stable/unstable lens counts
    - Per-lens confidence intervals
    """
    summary_file = CALIBRATION_DIR / "dashboard_summary.json"

    if not summary_file.exists():
        return {
            "status": "not_run",
            "message": "Calibration validation has not been run yet. Run: python app/utils/run_calibration_validation.py",
        }

    try:
        with open(summary_file) as f:
            data = json.load(f)

        return {
            "status": "ok",
            "data": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load calibration data: {e}")


@router.get("/lens/{concept_name}")
async def get_lens_stats(concept_name: str):
    """
    Get statistical metrics for a specific lens/concept.

    Returns:
    - Mean rank across probes
    - Rank confidence interval
    - Coefficient of variation (stability measure)
    - Detection rate (how often in top-k)
    """
    summary_file = CALIBRATION_DIR / "dashboard_summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Calibration data not found")

    try:
        with open(summary_file) as f:
            data = json.load(f)

        lens_stats = data.get("lens_stats", {})

        if concept_name not in lens_stats:
            raise HTTPException(status_code=404, detail=f"Lens '{concept_name}' not found in calibration data")

        return {
            "concept": concept_name,
            "stats": lens_stats[concept_name],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load lens data: {e}")


@router.get("/stability")
async def get_stability_overview():
    """
    Get overview of lens pack stability.

    Computes quality metrics from the actual calibration data (cross-activation
    statistics from training), which is more meaningful than single-word probes.
    """
    # Try to load from actual calibration.json (from HatCat lens pack)
    calibration_file = Path("/home/poss/Documents/Code/HatCatDev/lens_packs/gemma-3-4b_first-light-v1-bf16/calibration.json")

    if not calibration_file.exists():
        return {
            "status": "not_run",
            "stable_count": 0,
            "unstable_count": 0,
            "low_crossfire_rate": 0.0,
            "good_gap_rate": 0.0,
        }

    try:
        with open(calibration_file) as f:
            cal = json.load(f)

        calibration = cal.get("calibration", {})
        concepts = list(calibration.values())
        total = len(concepts)

        if total == 0:
            return {"status": "empty", "message": "No calibration data found"}

        # Compute quality metrics from cross-activation calibration
        # These are based on actual training data, not single-word probes
        low_cfr = sum(1 for c in concepts if c.get("cross_fire_rate", 1.0) < 0.1)
        very_low_cfr = sum(1 for c in concepts if c.get("cross_fire_rate", 1.0) < 0.02)
        good_gap = sum(1 for c in concepts if (c.get("self_mean", 0) - c.get("cross_mean", 1)) > 0.2)
        excellent_gap = sum(1 for c in concepts if (c.get("self_mean", 0) - c.get("cross_mean", 1)) > 0.4)

        # Find top over-firers (highest cross_fire_rate)
        sorted_by_cfr = sorted(concepts, key=lambda c: c.get("cross_fire_rate", 0), reverse=True)
        over_firers = [
            f"{c.get('concept', '?')}_L{c.get('layer', '?')}"
            for c in sorted_by_cfr[:10]
            if c.get("cross_fire_rate", 0) > 0.5
        ]

        # Compute average signal gap
        gaps = [c.get("self_mean", 0) - c.get("cross_mean", 1) for c in concepts]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0

        return {
            "status": "ok",
            "source": "calibration.json (training data)",
            "total_lenses": total,
            # Quality metrics
            "low_crossfire_rate": low_cfr / total,  # % with <10% cross-fire
            "very_low_crossfire_rate": very_low_cfr / total,  # % with <2% cross-fire
            "good_gap_rate": good_gap / total,  # % with >0.2 signal gap
            "excellent_gap_rate": excellent_gap / total,  # % with >0.4 signal gap
            "avg_signal_gap": avg_gap,
            # Counts
            "stable_count": low_cfr,  # Low cross-fire = stable
            "unstable_count": total - low_cfr,
            # Over-firers
            "over_firing": over_firers,
            # Metadata
            "timestamp": cal.get("timestamp", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load stability data: {e}")


@router.post("/run")
async def run_calibration(max_concepts: int = 100, top_k: int = 10):
    """
    Trigger a calibration validation run.

    Note: This is a long-running operation (~2-5 minutes for 100 concepts).
    Consider running via CLI instead for production use.
    """
    # For now, just return instructions - running model inference
    # synchronously in an API endpoint is not ideal
    return {
        "status": "use_cli",
        "message": "For best results, run calibration via CLI:",
        "command": f"python app/utils/run_calibration_validation.py --max-concepts {max_concepts} --top-k {top_k}",
    }
