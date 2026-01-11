#!/usr/bin/env python3
"""
Quick test script for induction concept comparison.

Tests different induction concepts per behavior to find optimal settings.
Runs a single sample per combination to quickly identify issues like model collapse.

Usage:
    python app/utils/test_induction_concepts.py --behavior sandbagging
    python app/utils/test_induction_concepts.py --behavior strategic_deception --strength 0.3
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HATCAT_DEV = Path("/home/poss/Documents/Code/HatCatDev")
sys.path.insert(0, str(HATCAT_DEV))


# Behavior to concept mapping for testing
BEHAVIOR_CONCEPTS = {
    "sandbagging": {
        "generic": ["Deception_L2"],
        "specific": ["DeceptiveSpeech_L4", "UnderSpecificationDeflection_L4"],
    },
    "reward_hacking": {
        "generic": ["Deception_L2", "Manipulation_L2"],
        "specific": ["ExploitationMode_L4"],
    },
    "sycophancy": {
        "generic": ["Deception_L2"],
        "specific": ["DeceptiveSpeech_L4"],  # SycophancySignal not calibrated
    },
    "strategic_deception": {
        "generic": ["Deception_L2"],
        "specific": ["MisleadingAnalogy_L5", "FalseAuthorityClaim_L5"],
    },
    "dark_patterns": {
        "generic": ["Manipulation_L2"],
        "specific": ["DarkPatterns_L5", "ManipulativeCommunication_L4"],
    },
    "persuasive_manipulation": {
        "generic": ["Manipulation_L2"],
        "specific": ["ManipulativeCommunication_L4", "OpenPersuasion_L5"],
    },
}


def load_episode(behavior: str) -> dict:
    """Load episode for given behavior."""
    suite_path = PROJECT_ROOT / "episodes" / "deception_suite_v2.json"
    with open(suite_path) as f:
        suite = json.load(f)

    for ep in suite.get("episodes", []):
        if ep.get("behavior") == behavior:
            return ep

    raise ValueError(f"No episode found for behavior: {behavior}")


def test_induction(
    model,
    tokenizer,
    lens_manager,
    episode: dict,
    induction_concept: str,
    strength: float = 0.5,
    max_tokens: int = 100,
):
    """
    Test a single induction concept on an episode.

    Returns dict with:
    - response: generated text
    - collapsed: bool if response looks like model collapse
    - tokens_generated: count
    """
    import torch
    from src.hat.steering.hooks import (
        load_steering_vectors_from_lens_pack,
        create_steering_hook,
        apply_steering_hooks,
        get_model_layers,
    )

    # Parse concept name and layer
    if "_L" in induction_concept:
        concept_name = induction_concept.rsplit("_L", 1)[0]
        layer_hint = int(induction_concept.rsplit("_L", 1)[1])
    else:
        concept_name = induction_concept
        layer_hint = None

    # Get lens pack path
    lens_pack_path = HATCAT_DEV / "lens_packs" / "gemma-3-4b_first-light-v1-bf16"

    # Load steering vector
    vectors, layers = load_steering_vectors_from_lens_pack(
        lens_pack_path,
        [concept_name],
        positive_only=True,
    )

    if concept_name not in vectors:
        return {"error": f"No vector found for {concept_name}", "response": "", "collapsed": True}

    vector = vectors[concept_name]
    device = next(model.parameters()).device

    # Get model layers
    model_layers = get_model_layers(model)
    n_layers = len(model_layers)

    # Target mid and late layers
    mid_layer = n_layers // 2
    late_layer = (3 * n_layers) // 4

    # Create hooks
    hook_pairs = []
    for layer_idx in [mid_layer, late_layer]:
        hook_fn = create_steering_hook(
            vector,
            strength,  # Positive = amplify
            device,
            layer_idx=layer_idx,
            total_layers=n_layers,
        )
        hook_pairs.append((model_layers[layer_idx], hook_fn))

    # Apply hooks
    handles = apply_steering_hooks(hook_pairs)

    try:
        # Build prompt
        prompt = _build_prompt(episode, tokenizer)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Check for collapse indicators
        collapsed = _check_collapse(response)

        return {
            "response": response,
            "collapsed": collapsed,
            "tokens_generated": len(outputs[0]) - inputs.input_ids.shape[1],
            "concept": induction_concept,
            "strength": strength,
        }

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()


def _build_prompt(episode: dict, tokenizer) -> str:
    """Build prompt from episode turns."""
    messages = []

    if "system_prompt" in episode:
        messages.append({"role": "system", "content": episode["system_prompt"]})

    for turn in episode.get("turns", []):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        messages.append({"role": role, "content": content})

    # Use chat template
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass

    # Fallback
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _check_collapse(response: str) -> bool:
    """Check if response shows signs of model collapse."""
    if not response or len(response.strip()) < 10:
        return True

    # Check for repetition
    words = response.split()
    if len(words) > 10:
        # Check if last 5 words repeat
        if len(set(words[-10:])) < 3:
            return True

    # Check for gibberish (very low unique char ratio)
    if len(response) > 50:
        unique_chars = len(set(response.lower()))
        if unique_chars < 10:
            return True

    return False


def main():
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="Test induction concepts")
    parser.add_argument("--behavior", required=True, choices=list(BEHAVIOR_CONCEPTS.keys()))
    parser.add_argument("--strength", type=float, default=0.5, help="Induction strength")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--test-all", action="store_true", help="Test all concepts for behavior")
    parser.add_argument("--concept", type=str, help="Test specific concept")

    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model", {}).get("name", "google/gemma-3-4b-pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"INDUCTION CONCEPT TEST - {args.behavior}")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Strength: {args.strength}")
    print(f"  Max tokens: {args.max_tokens}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # Load lens manager (for steering vectors)
    print("Loading lens manager...")
    from src.hat.monitoring.lens_manager import DynamicLensManager

    lens_pack_path = HATCAT_DEV / "lens_packs" / "gemma-3-4b_first-light-v1-bf16"
    concept_pack_path = HATCAT_DEV / "concept_packs" / "first-light"

    lens_manager = DynamicLensManager(
        lenses_dir=lens_pack_path,
        layers_data_dir=concept_pack_path / "hierarchy",
        device=device,
    )

    # Load episode
    episode = load_episode(args.behavior)
    print(f"\nEpisode: {episode['id']}")

    # Determine concepts to test
    if args.concept:
        concepts = [args.concept]
    elif args.test_all:
        concepts = BEHAVIOR_CONCEPTS[args.behavior]["generic"] + BEHAVIOR_CONCEPTS[args.behavior]["specific"]
    else:
        # Just test generic vs first specific
        concepts = [
            BEHAVIOR_CONCEPTS[args.behavior]["generic"][0],
            BEHAVIOR_CONCEPTS[args.behavior]["specific"][0],
        ]

    # Run tests
    results = []
    for concept in concepts:
        print(f"\n--- Testing: {concept} ---")

        result = test_induction(
            model=model,
            tokenizer=tokenizer,
            lens_manager=lens_manager,
            episode=episode,
            induction_concept=concept,
            strength=args.strength,
            max_tokens=args.max_tokens,
        )

        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Collapsed: {result['collapsed']}")
            print(f"  Tokens: {result['tokens_generated']}")
            print(f"  Response preview: {result['response'][:200]}...")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "COLLAPSE" if r.get("collapsed") else "OK"
        concept = r.get("concept", "unknown")
        print(f"  {concept}: {status}")

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "induction_tests" / f"{args.behavior}_{datetime.now().strftime('%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
