"""
run_experiment.py

Reads scenarios from data/nq_open/contexts.json, sends each prompt
to the model via Ollama, and saves every response to results/runs/.

Usage:
    python3 src/run_experiment.py --model qwen2.5:3b
    python3 src/run_experiment.py --model mistral:7b

Output:
    results/runs/{model_name}/result_{scenario_id}.json
"""

import json
import os
import time
import argparse
from tqdm import tqdm
import ollama

# ── Config ────────────────────────────────────────────────────────────────────

SCENARIOS_PATH = "data/nq_open/contexts.json"
RESULTS_DIR    = "results/runs"

# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_model_name(model: str) -> str:
    """Turn 'qwen2.5:3b' into 'qwen2.5_3b' for use in file/folder names."""
    return model.replace(":", "_").replace("/", "_")


def already_done(output_dir: str, scenario_id: int) -> bool:
    """Check if this scenario was already run (allows resuming if interrupted)."""
    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    return os.path.exists(path)


def query_model(model: str, prompt: str) -> tuple[str, float]:
    """
    Send a prompt to Ollama and return (response_text, elapsed_seconds).
    Temperature=0 for deterministic, reproducible results.
    """
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    elapsed = time.time() - start
    return response["message"]["content"].strip(), elapsed


def save_result(output_dir: str, scenario_id: int, scenario: dict,
                response: str, elapsed: float):
    """Save one result as a JSON file."""
    result = {
        # --- identity ---
        "scenario_id":  scenario_id,
        "question_id":  scenario["question_id"],
        "question":     scenario["question"],
        "answer":       scenario["answer"],
        "length":       scenario["length"],
        "position":     scenario["position"],
        "gold_index":   scenario["gold_index"],

        # --- inputs ---
        "prompt":       scenario["prompt"],
        "doc_texts":    scenario["doc_texts"],
        "gold_doc":     scenario["gold_doc"],

        # --- output ---
        "response":     response,
        "elapsed_s":    round(elapsed, 2),
    }

    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="qwen2.5:3b",
        help="Ollama model name (e.g. qwen2.5:3b, mistral:7b)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run first N scenarios (useful for dry runs)"
    )
    args = parser.parse_args()

    # Load scenarios
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    if args.limit:
        scenarios = scenarios[:args.limit]
        print(f"Dry run mode: only running first {args.limit} scenarios.")

    # Set up output directory
    model_tag  = sanitize_model_name(args.model)
    output_dir = os.path.join(RESULTS_DIR, model_tag)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nModel:      {args.model}")
    print(f"Scenarios:  {len(scenarios)}")
    print(f"Output dir: {output_dir}\n")

    # Verify model is available before starting
    try:
        ollama.chat(
            model=args.model,
            messages=[{"role": "user", "content": "hi"}],
            options={"temperature": 0}
        )
        print("Model check passed.\n")
    except Exception as e:
        print(f"ERROR: Could not reach model '{args.model}'. Is Ollama running?")
        print(f"Details: {e}")
        return

    # Run
    skipped  = 0
    errors   = 0
    timings  = []

    for i, scenario in enumerate(tqdm(scenarios, desc="Running scenarios")):
        if already_done(output_dir, i):
            skipped += 1
            continue

        try:
            response, elapsed = query_model(args.model, scenario["prompt"])
            save_result(output_dir, i, scenario, response, elapsed)
            timings.append(elapsed)

        except Exception as e:
            errors += 1
            tqdm.write(f"  ERROR on scenario {i}: {e}")
            continue

    # Summary
    print(f"\n--- Done ---")
    print(f"Ran:     {len(scenarios) - skipped - errors}")
    print(f"Skipped: {skipped} (already existed)")
    print(f"Errors:  {errors}")
    if timings:
        avg = sum(timings) / len(timings)
        total = sum(timings)
        print(f"Avg time per call: {avg:.1f}s")
        print(f"Total time:        {total/60:.1f} minutes")


if __name__ == "__main__":
    main()