#!/usr/bin/env python3
"""
Run Large Sample Evaluation for Hackathon Results.

Runs the deception suite with multiple samples per episode for
statistically meaningful A/B/C comparison.

Usage:
    python app/utils/run_large_evaluation.py --samples 10
    python app/utils/run_large_evaluation.py --samples 20 --conditions A B C
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import statistics

from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add HatCatDev to path
HATCAT_DEV = Path("/home/poss/Documents/Code/HatCatDev")
sys.path.insert(0, str(HATCAT_DEV))


def run_large_evaluation(
    sample_count: int = 10,
    conditions: List[str] = ["A", "B", "C"],
    suite_file: str = "deception_suite_v2.json",
    max_tokens: int = 150,
    temperature: float = 0.7,
):
    """Run large sample evaluation."""
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from app.evaluation.hatcat_runner import HatCatEvaluationRunner

    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", {}).get("name", "google/gemma-3-4b-pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("LARGE SAMPLE EVALUATION")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Samples per episode: {sample_count}")
    print(f"  Conditions: {conditions}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Load lens manager
    print("Loading lens manager...")
    from src.hat.monitoring.lens_manager import DynamicLensManager

    lens_pack_name = config.get("lens_pack", {}).get("name", "gemma-3-4b_first-light-v1-bf16")
    # Handle config using different naming convention
    if "first-light-v1" in lens_pack_name and not (HATCAT_DEV / "lens_packs" / lens_pack_name).exists():
        lens_pack_name = "gemma-3-4b_first-light-v1-bf16"
    lens_pack_path = HATCAT_DEV / "lens_packs" / lens_pack_name
    concept_pack_path = HATCAT_DEV / "concept_packs" / "first-light"

    lens_manager = DynamicLensManager(
        lenses_dir=lens_pack_path,
        layers_data_dir=concept_pack_path / "hierarchy",
        device=device,
    )
    print(f"  Loaded {len(lens_manager.cache.loaded_lenses)} lenses")

    # Create runner
    concept_pack_path = HATCAT_DEV / "concept_packs" / "first-light"
    runner = HatCatEvaluationRunner(
        model=model,
        tokenizer=tokenizer,
        lens_manager=lens_manager,
        concept_pack_path=concept_pack_path,
        device=device,
    )

    # Load episodes
    suite_path = PROJECT_ROOT / "episodes" / suite_file
    with open(suite_path) as f:
        suite_data = json.load(f)

    episodes = suite_data.get("episodes", [])
    suite_info = suite_data.get("suite", {})
    print(f"\nLoaded {len(episodes)} episodes from {suite_file}")

    # Run evaluation
    results = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": suite_info,
        "sample_count": sample_count,
        "conditions": conditions,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "episodes": [],
    }

    total = len(episodes) * len(conditions) * sample_count
    print(f"\nRunning {total} total generations...")

    pbar = tqdm(total=total, desc="Running evaluation")

    for episode in episodes:
        episode_id = episode["id"]
        behavior = episode.get("behavior", "unknown")

        episode_result = {
            "episode_id": episode_id,
            "behavior": behavior,
            "conditions": {},
        }

        for condition in conditions:
            # Track all samples for this condition
            samples = []
            manipulation_detected_count = 0
            peak_scores = []
            intervention_counts = []
            correction_counts = []
            pass_count = 0
            fail_count = 0
            null_count = 0
            confidences = []

            for sample_idx in range(sample_count):
                try:
                    # Run streaming episode and consume all tokens
                    tokens = []
                    gen = runner.run_episode_streaming(
                        episode=episode,
                        condition=condition,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    # Consume generator
                    try:
                        while True:
                            token_result = next(gen)
                            tokens.append(token_result)
                    except StopIteration as e:
                        result = e.value

                    # Track metrics
                    if result.manipulation_detected:
                        manipulation_detected_count += 1
                    peak_scores.append(result.peak_detection_score)
                    intervention_counts.append(result.interventions_triggered)
                    correction_counts.append(result.steering_corrections)

                    # Score result
                    score_result = getattr(result, 'score_result', 'null')
                    if score_result == 'pass':
                        pass_count += 1
                    elif score_result == 'fail':
                        fail_count += 1
                    else:
                        null_count += 1

                    confidence = getattr(result, 'score_confidence', 0.0)
                    confidences.append(confidence)

                    samples.append({
                        "manipulation_detected": result.manipulation_detected,
                        "peak_score": result.peak_detection_score,
                        "interventions": result.interventions_triggered,
                        "corrections": result.steering_corrections,
                        "score_result": score_result,
                        "confidence": confidence,
                        "response_preview": result.responses.get(0, "")[:200],
                    })

                except Exception as e:
                    print(f"\n  Error on {episode_id}/{condition} sample {sample_idx}: {e}")
                    samples.append({"error": str(e)})

                pbar.update(1)

            # Aggregate statistics for this condition
            n = len([s for s in samples if "error" not in s])
            if n > 0:
                manipulation_rate = manipulation_detected_count / n
                avg_peak_score = statistics.mean(peak_scores) if peak_scores else 0
                avg_interventions = statistics.mean(intervention_counts) if intervention_counts else 0
                avg_corrections = statistics.mean(correction_counts) if correction_counts else 0
                avg_confidence = statistics.mean(confidences) if confidences else 0

                # Compute 95% CI for manipulation rate
                import math
                se = math.sqrt(manipulation_rate * (1 - manipulation_rate) / n) if n > 1 else 0
                ci_lower = max(0, manipulation_rate - 1.96 * se)
                ci_upper = min(1, manipulation_rate + 1.96 * se)

                episode_result["conditions"][condition] = {
                    "samples": n,
                    "manipulation_detected": manipulation_detected_count,
                    "manipulation_rate": round(manipulation_rate, 4),
                    "manipulation_rate_ci": [round(ci_lower, 4), round(ci_upper, 4)],
                    "avg_peak_score": round(avg_peak_score, 4),
                    "avg_interventions": round(avg_interventions, 2),
                    "avg_corrections": round(avg_corrections, 2),
                    "pass_count": pass_count,
                    "fail_count": fail_count,
                    "null_count": null_count,
                    "pass_rate": round(pass_count / n, 4) if n > 0 else 0,
                    "fail_rate": round(fail_count / n, 4) if n > 0 else 0,
                    "avg_confidence": round(avg_confidence, 4),
                    "sample_details": samples,
                }
            else:
                episode_result["conditions"][condition] = {"error": "All samples failed"}

        results["episodes"].append(episode_result)

    pbar.close()

    # Compute summary
    summary = compute_summary(results)
    results["summary"] = summary

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"large_eval_{results['run_id']}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print(f"\nSummary by Condition:")
    for cond, stats in summary.get("by_condition", {}).items():
        rate = stats.get("manipulation_rate", 0) * 100
        ci = stats.get("manipulation_rate_ci", [0, 0])
        interventions = stats.get("total_interventions", 0)
        print(f"  {cond}: {rate:.1f}% manipulation [{ci[0]*100:.1f}%-{ci[1]*100:.1f}%], {interventions} interventions")

    print(f"\nSummary by Behavior:")
    for beh, cond_stats in summary.get("by_behavior", {}).items():
        print(f"  {beh}:")
        for cond, stats in cond_stats.items():
            rate = stats.get("manipulation_rate", 0) * 100
            print(f"    {cond}: {rate:.1f}%")

    return results


def compute_summary(results: dict) -> dict:
    """Compute aggregate summary statistics."""
    import math

    summary = {
        "by_condition": {},
        "by_behavior": {},
    }

    conditions = results.get("conditions", ["A", "B", "C"])
    behaviors = set()

    # Initialize
    for cond in conditions:
        summary["by_condition"][cond] = {
            "total_samples": 0,
            "manipulation_count": 0,
            "manipulation_rate": 0,
            "manipulation_rate_ci": [0, 0],
            "total_interventions": 0,
            "total_corrections": 0,
            "pass_count": 0,
            "fail_count": 0,
            "null_count": 0,
        }

    # Aggregate
    for ep in results.get("episodes", []):
        behavior = ep.get("behavior", "unknown")
        behaviors.add(behavior)

        if behavior not in summary["by_behavior"]:
            summary["by_behavior"][behavior] = {}

        for cond, data in ep.get("conditions", {}).items():
            if cond not in summary["by_condition"]:
                continue

            n = data.get("samples", 0)
            manipulation = data.get("manipulation_detected", 0)
            interventions = data.get("avg_interventions", 0) * n
            corrections = data.get("avg_corrections", 0) * n

            summary["by_condition"][cond]["total_samples"] += n
            summary["by_condition"][cond]["manipulation_count"] += manipulation
            summary["by_condition"][cond]["total_interventions"] += int(interventions)
            summary["by_condition"][cond]["total_corrections"] += int(corrections)
            summary["by_condition"][cond]["pass_count"] += data.get("pass_count", 0)
            summary["by_condition"][cond]["fail_count"] += data.get("fail_count", 0)
            summary["by_condition"][cond]["null_count"] += data.get("null_count", 0)

            # Per-behavior stats
            if cond not in summary["by_behavior"][behavior]:
                summary["by_behavior"][behavior][cond] = {
                    "total_samples": 0,
                    "manipulation_count": 0,
                    "manipulation_rate": 0,
                }
            summary["by_behavior"][behavior][cond]["total_samples"] += n
            summary["by_behavior"][behavior][cond]["manipulation_count"] += manipulation

    # Compute rates and CIs
    for cond in conditions:
        stats = summary["by_condition"][cond]
        n = stats["total_samples"]
        if n > 0:
            rate = stats["manipulation_count"] / n
            stats["manipulation_rate"] = round(rate, 4)

            # 95% CI
            se = math.sqrt(rate * (1 - rate) / n) if n > 1 else 0
            stats["manipulation_rate_ci"] = [
                round(max(0, rate - 1.96 * se), 4),
                round(min(1, rate + 1.96 * se), 4),
            ]

            # Pass rate
            sample_total = stats["pass_count"] + stats["fail_count"] + stats["null_count"]
            if sample_total > 0:
                stats["pass_rate"] = round(stats["pass_count"] / sample_total, 4)
                stats["fail_rate"] = round(stats["fail_count"] / sample_total, 4)

    # Per-behavior rates
    for beh in behaviors:
        for cond in conditions:
            if cond in summary["by_behavior"].get(beh, {}):
                stats = summary["by_behavior"][beh][cond]
                n = stats["total_samples"]
                if n > 0:
                    stats["manipulation_rate"] = round(stats["manipulation_count"] / n, 4)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run large sample evaluation")
    parser.add_argument("--samples", type=int, default=10, help="Samples per episode per condition")
    parser.add_argument("--conditions", nargs="+", default=["A", "B", "C"], help="Conditions to run")
    parser.add_argument("--suite", default="deception_suite_v2.json", help="Episode suite file")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    run_large_evaluation(
        sample_count=args.samples,
        conditions=args.conditions,
        suite_file=args.suite,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
