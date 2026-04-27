"""
run_experiment.py

Reads scenarios from data/nq_open/contexts.json, sends each prompt
to the model and saves every response to results/runs/.

Uses HuggingFace transformers directly — no Ollama needed.
Same CLI interface as before, same output JSON format.

Usage:
    python3 src/run_experiment.py --model qwen2.5:3b
    python3 src/run_experiment.py --model qwen2.5:3b --trial 2
    python3 src/run_experiment.py --model mistral:7b --trial 3
    python3 src/run_experiment.py --model mistral:7b --limit 5  # dry run

Output:
    results/runs/{model_name}/trial_{n}/result_{scenario_id}.json
"""

import json
import os
import time
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

SCENARIOS_PATH = "data/nq_open/contexts.json"
RESULTS_DIR    = "results/runs"
TEMPERATURE    = 0.7
MAX_NEW_TOKENS = 100

# Ollama name -> HuggingFace repo
MODEL_MAP = {
    "qwen2.5:3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b":  "Qwen/Qwen2.5-14B-Instruct",
    "mistral:7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1:8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma2:9b":    "google/gemma-2-9b-it",
    "phi3:mini":    "microsoft/Phi-3-mini-4k-instruct",
    "phi3:medium":  "microsoft/Phi-3-medium-4k-instruct",
}

# Models that need 4-bit quantization to fit on L4 (24GB VRAM)
NEEDS_4BIT = {"qwen2.5:14b", "phi3:medium"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def already_done(output_dir: str, scenario_id: int) -> bool:
    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    return os.path.exists(path)


def migrate_existing_results(model_dir: str):
    """
    If flat result_XXXX.json files exist from old format,
    move them into trial_1/ automatically.
    """
    trial_1_dir = os.path.join(model_dir, "trial_1")
    flat_files = [
        f for f in os.listdir(model_dir)
        if f.startswith("result_") and f.endswith(".json")
    ]
    if flat_files:
        print(f"  Found {len(flat_files)} existing results — moving to trial_1/...")
        os.makedirs(trial_1_dir, exist_ok=True)
        for fname in flat_files:
            os.rename(
                os.path.join(model_dir, fname),
                os.path.join(trial_1_dir, fname)
            )
        print("  Done. Existing results preserved.\n")


def load_model(ollama_name: str):
    """Load model and tokenizer from HuggingFace."""
    if ollama_name not in MODEL_MAP:
        raise ValueError(
            f"Unknown model: {ollama_name}\n"
            f"Available: {list(MODEL_MAP.keys())}"
        )

    hf_name    = MODEL_MAP[ollama_name]
    needs_4bit = ollama_name in NEEDS_4BIT

    print(f"Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    if needs_4bit:
        print("  Applying 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    print(f"  Loaded. Device: {next(model.parameters()).device}\n")
    return tokenizer, model


def query_model(tokenizer, model, prompt: str) -> tuple[str, float]:
    """Run one prompt, return (response, elapsed_seconds)."""
    messages = [{"role": "user", "content": prompt}]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
    except Exception:
        text = f"User: {prompt}\nAssistant:"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, elapsed


def save_result(output_dir: str, scenario_id: int, scenario: dict,
                response: str, elapsed: float, trial: int):
    result = {
        "scenario_id": scenario_id,
        "question_id": scenario["question_id"],
        "question":    scenario["question"],
        "answer":      scenario["answer"],
        "length":      scenario["length"],
        "position":    scenario["position"],
        "gold_index":  scenario["gold_index"],
        "trial":       trial,
        "prompt":      scenario["prompt"],
        "doc_texts":   scenario["doc_texts"],
        "gold_doc":    scenario["gold_doc"],
        "response":    response,
        "elapsed_s":   round(elapsed, 2),
    }
    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:3b",
                        help=f"Model name. Available: {list(MODEL_MAP.keys())}")
    parser.add_argument("--trial", type=int, default=1,
                        help="Trial number (1, 2, or 3)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run first N scenarios (dry run)")
    args = parser.parse_args()

    # Load scenarios
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    if args.limit:
        scenarios = scenarios[:args.limit]
        print(f"Dry run: {args.limit} scenarios only.")

    # Directory setup
    model_tag  = sanitize_model_name(args.model)
    model_dir  = os.path.join(RESULTS_DIR, model_tag)
    os.makedirs(model_dir, exist_ok=True)
    migrate_existing_results(model_dir)

    output_dir = os.path.join(model_dir, f"trial_{args.trial}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nModel:      {args.model} -> {MODEL_MAP[args.model]}")
    print(f"Trial:      {args.trial}")
    print(f"Scenarios:  {len(scenarios)}")
    print(f"Output dir: {output_dir}\n")

    # Load model once, reuse for all scenarios
    tokenizer, model = load_model(args.model)

    # Run
    skipped = 0
    errors  = 0
    timings = []

    for i, scenario in enumerate(tqdm(scenarios, desc="Running")):
        if already_done(output_dir, i):
            skipped += 1
            continue
        try:
            response, elapsed = query_model(tokenizer, model, scenario["prompt"])
            save_result(output_dir, i, scenario, response, elapsed, args.trial)
            timings.append(elapsed)
        except Exception as e:
            errors += 1
            tqdm.write(f"  ERROR on scenario {i}: {e}")

    print(f"\n--- Done ---")
    print(f"Trial:   {args.trial}")
    print(f"Ran:     {len(scenarios) - skipped - errors}")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    if timings:
        print(f"Avg time per call: {sum(timings)/len(timings):.1f}s")
        print(f"Total time: {sum(timings)/60:.1f} minutes")

    # Free GPU memory before next run
    del model, tokenizer
    torch.cuda.empty_cache()
    print("GPU memory cleared.")


if __name__ == "__main__":
    main()