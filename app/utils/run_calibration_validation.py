#!/usr/bin/env python3
"""
Quick Calibration Validation with Statistical Estimation.

Uses BatchedLensBank for proper 3-layer MLP inference.

Usage:
    python app/utils/run_calibration_validation.py --max-concepts 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict

import numpy as np
from tqdm import tqdm

# Add HatCatDev to path
HATCAT_DEV = Path("/home/poss/Documents/Code/HatCatDev")
sys.path.insert(0, str(HATCAT_DEV))


@dataclass
class ValidationResults:
    """Results from quick validation run."""
    timestamp: str
    lens_pack: str
    concepts_probed: int
    total_lenses: int

    # Core metrics
    avg_diagonal_rank: float = 0.0
    diagonal_in_top_k_rate: float = 0.0

    # Statistical estimation
    topk_jaccard_mean: float = 0.0
    topk_jaccard_std: float = 0.0
    stable_lens_count: int = 0
    unstable_lens_count: int = 0

    # Per-lens statistics
    lens_stats: Dict = field(default_factory=dict)

    # Outliers
    over_firing: List[str] = field(default_factory=list)
    under_firing: List[str] = field(default_factory=list)


def run_validation(
    max_concepts: int = 100,
    layers: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    top_k: int = 10,
):
    """Run quick calibration validation using BatchedLensBank."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.hat.monitoring.lens_batched import BatchedLensBank
    from src.hat.monitoring.lens_types import SimpleMLP, create_lens_from_state_dict

    # Paths
    lens_pack = HATCAT_DEV / "lens_packs/gemma-3-4b_first-light-v1-bf16"
    concept_pack = HATCAT_DEV / "concept_packs/first-light"
    output_dir = output_dir or Path("/home/poss/Documents/Code/AIManipulationHackathon/outputs/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: google/gemma-3-4b-pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-4b-pt",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Determine layers
    if layers is None:
        layers = []
        for layer_dir in lens_pack.glob("layer*"):
            if layer_dir.is_dir():
                try:
                    layer_num = int(layer_dir.name.replace("layer", ""))
                    layers.append(layer_num)
                except ValueError:
                    pass
        layers.sort()

    print(f"Layers: {layers}")

    # Load all lenses into BatchedLensBank
    print("Loading lenses into BatchedLensBank...")
    lens_bank = BatchedLensBank(device=device)
    all_lenses = {}
    lens_layer_map = {}  # concept -> layer

    hidden_dim = 2560  # gemma-3-4b hidden dimension

    for layer in layers:
        layer_dir = lens_pack / f"layer{layer}"
        if not layer_dir.exists():
            continue

        for lens_file in layer_dir.glob("*.pt"):
            concept_name = lens_file.stem.replace('_classifier', '')
            try:
                state_dict = torch.load(lens_file, map_location='cpu', weights_only=True)
                lens = create_lens_from_state_dict(state_dict, hidden_dim=hidden_dim, device='cpu')
                all_lenses[concept_name] = lens
                lens_layer_map[concept_name] = layer
            except Exception as e:
                print(f"  Warning: Failed to load {lens_file}: {e}")

    print(f"  Loaded {len(all_lenses)} lenses")
    lens_bank.add_lenses(all_lenses)
    lens_bank.to(device)
    lens_bank.eval()

    # Load concepts
    print("Loading concepts...")
    concepts = {}
    for layer in layers:
        layer_file = concept_pack / "hierarchy" / f"layer{layer}.json"
        if not layer_file.exists():
            continue

        with open(layer_file) as f:
            layer_data = json.load(f)

        concept_list = layer_data.get('concepts', layer_data if isinstance(layer_data, list) else [])
        for concept in concept_list:
            term = concept.get('sumo_term') or concept.get('term')
            if term and term in all_lenses:
                concepts[term] = {
                    'layer': layer,
                    'prompt': term,  # Fast mode: use term as prompt
                }

    print(f"  Found {len(concepts)} concepts with lenses")

    # Sample concepts
    concept_names = list(concepts.keys())
    if max_concepts and len(concept_names) > max_concepts:
        import random
        concept_names = random.sample(concept_names, max_concepts)

    print(f"\nRunning validation on {len(concept_names)} concepts...")

    # Track statistics
    lens_ranks: Dict[str, List[int]] = defaultdict(list)
    lens_activations: Dict[str, List[float]] = defaultdict(list)
    over_fire_counts: Dict[str, int] = defaultdict(int)
    diagonal_ranks = []
    topk_sets = []

    # Use last layer to match HushedGenerator (hush_integration.py:969)
    # which extracts: outputs.hidden_states[-1][0, -1, :].detach()
    layer_idx = -1  # Last layer

    for concept_name in tqdm(concept_names, desc="Probing concepts"):
        concept_data = concepts[concept_name]
        prompt = concept_data['prompt']

        try:
            # Get activation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[layer_idx]
                activation = hidden_states[0, -1, :].float()

            # Score all lenses
            scores = lens_bank(activation)

            # Sort by score
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Track ranks and activations
            for rank, (lens_concept, score) in enumerate(sorted_scores, 1):
                lens_ranks[lens_concept].append(rank)
                lens_activations[lens_concept].append(score)

                # Track over-firing (in top-k but not the target)
                if rank <= top_k and lens_concept != concept_name:
                    over_fire_counts[lens_concept] += 1

                # Track diagonal rank
                if lens_concept == concept_name:
                    diagonal_ranks.append(rank)

            # Track top-k set for Jaccard
            topk_set = {s[0] for s in sorted_scores[:top_k]}
            topk_sets.append(topk_set)

        except Exception as e:
            print(f"  Error probing {concept_name}: {e}")
            continue

    print("\nComputing statistics...")

    # Compute per-lens statistics with CIs
    lens_stats = {}
    stable_count = 0
    unstable_count = 0

    try:
        from src.map.statistics import compute_calibration_confidence
        HAS_STATS = True
    except ImportError:
        HAS_STATS = False
        print("  Warning: Statistics module not available, skipping CIs")

    for lens_concept in tqdm(list(lens_ranks.keys())[:200], desc="Computing lens stats"):  # Limit for speed
        ranks = lens_ranks[lens_concept]
        activations = lens_activations[lens_concept]

        if len(ranks) < 2:
            continue

        mean_rank = float(np.mean(ranks))
        std_rank = float(np.std(ranks))
        cv = std_rank / mean_rank if mean_rank > 0 else 0.0
        is_stable = cv < 0.5

        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1

        # Bootstrap CIs
        if HAS_STATS and len(ranks) >= 3:
            conf = compute_calibration_confidence(
                ranks=ranks,
                activations=activations,
                top_k=top_k,
                concept=lens_concept,
                layer=lens_layer_map.get(lens_concept, 0),
                n_bootstrap=200,  # Fast
            )
            lens_stats[lens_concept] = {
                'mean_rank': conf.rank_mean,
                'rank_ci': [conf.rank_ci_lower, conf.rank_ci_upper],
                'cv': conf.cv,
                'is_stable': conf.is_stable,
                'detection_rate': conf.detection_rate,
                'detection_rate_ci': [conf.detection_rate_ci_lower, conf.detection_rate_ci_upper],
            }
        else:
            in_topk_rate = sum(1 for r in ranks if r <= top_k) / len(ranks)
            lens_stats[lens_concept] = {
                'mean_rank': mean_rank,
                'rank_ci': [mean_rank - 1.96 * std_rank / np.sqrt(len(ranks)),
                           mean_rank + 1.96 * std_rank / np.sqrt(len(ranks))],
                'cv': cv,
                'is_stable': is_stable,
                'detection_rate': in_topk_rate,
            }

    # Compute Jaccard stability
    jaccard_mean = 0.0
    jaccard_std = 0.0
    if len(topk_sets) >= 2:
        jaccards = []
        sample_size = min(50, len(topk_sets))
        import random
        sampled = random.sample(topk_sets, sample_size)

        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                intersection = len(sampled[i] & sampled[j])
                union = len(sampled[i] | sampled[j])
                if union > 0:
                    jaccards.append(intersection / union)

        if jaccards:
            jaccard_mean = float(np.mean(jaccards))
            jaccard_std = float(np.std(jaccards))

    # Identify outliers
    over_firing = sorted(over_fire_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    over_firing_concepts = [c for c, _ in over_firing]

    # Summary
    avg_diag_rank = float(np.mean(diagonal_ranks)) if diagonal_ranks else 0.0
    diag_in_topk = sum(1 for r in diagonal_ranks if r <= top_k) / len(diagonal_ranks) if diagonal_ranks else 0.0

    results = ValidationResults(
        timestamp=datetime.now(timezone.utc).isoformat(),
        lens_pack=lens_pack.name,
        concepts_probed=len(concept_names),
        total_lenses=len(all_lenses),
        avg_diagonal_rank=avg_diag_rank,
        diagonal_in_top_k_rate=diag_in_topk,
        topk_jaccard_mean=jaccard_mean,
        topk_jaccard_std=jaccard_std,
        stable_lens_count=stable_count,
        unstable_lens_count=unstable_count,
        lens_stats=lens_stats,
        over_firing=over_firing_concepts,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Lens pack: {results.lens_pack}")
    print(f"  Concepts probed: {results.concepts_probed}")
    print(f"  Total lenses: {results.total_lenses}")
    print()
    print(f"  Avg diagonal rank: {results.avg_diagonal_rank:.1f}")
    print(f"  Diagonal in top-{top_k}: {results.diagonal_in_top_k_rate:.1%}")
    print()
    print(f"  Top-k Jaccard: mean={results.topk_jaccard_mean:.3f}, std={results.topk_jaccard_std:.3f}")
    print(f"  Stable lenses: {results.stable_lens_count}, Unstable: {results.unstable_lens_count}")
    print()
    if results.over_firing:
        print(f"  Top over-firing concepts: {', '.join(results.over_firing[:5])}")

    # Save results
    summary_file = output_dir / "dashboard_summary.json"
    with open(summary_file, "w") as f:
        json.dump(asdict(results), f, indent=2)

    print(f"\nSaved to: {summary_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run quick calibration validation")
    parser.add_argument("--max-concepts", type=int, default=100, help="Max concepts to probe")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layers to analyze")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for metrics")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    run_validation(
        max_concepts=args.max_concepts,
        layers=args.layers,
        output_dir=output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
