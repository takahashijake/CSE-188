"""
run_experiment_hf_debug.py

Same as run_experiment_hf.py but with detailed debug logging.
"""

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
# !pip install transformers accelerate bitsandbytes datasets tqdm -q


# ── Cell 2: Imports and config ────────────────────────────────────────────────

import json
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_MAP = {
    "qwen2.5:3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b":   "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b":  "Qwen/Qwen2.5-14B-Instruct",
    "mistral:7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3.1:8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

NEEDS_4BIT = {"qwen2.5:14b"}

SCENARIOS_PATH = "data/nq_open/contexts.json"
RESULTS_DIR    = "results/runs"
TEMPERATURE    = 0.7
MAX_NEW_TOKENS = 100


# ── DEBUG LOGGER ──────────────────────────────────────────────────────────────

def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize_model_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def already_done(output_dir: str, scenario_id: int) -> bool:
    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    return os.path.exists(path)


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(ollama_name: str):
    hf_name = MODEL_MAP[ollama_name]

    log(f"Tokenizer loading: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    log("Tokenizer loaded")

    needs_4bit = ollama_name in NEEDS_4BIT

    log("Starting model load (this may take a while)...")

    if needs_4bit:
        log("Using 4-bit quantization")
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

    log("Model weights loaded")

    model.eval()
    device = next(model.parameters()).device
    log(f"Model ready on device: {device}")

    return tokenizer, model


# ── Query function ────────────────────────────────────────────────────────────

def query_model(tokenizer, model, prompt: str, i: int = None):
    if i is not None:
        log(f"Running inference for scenario {i}")

    messages = [{"role": "user", "content": prompt}]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = inputs.to(model.device)
    except Exception:
        log("Chat template failed, using fallback formatting")
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
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if i is not None:
        log(f"Finished scenario {i} in {elapsed:.2f}s")

    return response, elapsed


# ── Save results ──────────────────────────────────────────────────────────────

def save_result(output_dir, scenario_id, scenario, response, elapsed, trial):
    result = {
        "scenario_id": scenario_id,
        "question_id": scenario["question_id"],
        "question": scenario["question"],
        "answer": scenario["answer"],
        "length": scenario["length"],
        "position": scenario["position"],
        "gold_index": scenario["gold_index"],
        "trial": trial,
        "prompt": scenario["prompt"],
        "doc_texts": scenario["doc_texts"],
        "gold_doc": scenario["gold_doc"],
        "response": response,
        "elapsed_s": round(elapsed, 2),
    }

    path = os.path.join(output_dir, f"result_{scenario_id:04d}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_experiment(ollama_name: str, trial: int, limit: int = None):

    log("Loading scenarios...")
    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    log(f"Loaded {len(scenarios)} scenarios")

    if limit:
        scenarios = scenarios[:limit]
        log(f"LIMIT active: running only {limit} scenarios")

    model_tag  = sanitize_model_name(ollama_name)
    model_dir  = os.path.join(RESULTS_DIR, model_tag)
    output_dir = os.path.join(model_dir, f"trial_{trial}")

    log("Creating output directory...")
    os.makedirs(output_dir, exist_ok=True)

    log(f"Model: {ollama_name}")
    log(f"Trial: {trial}")
    log(f"Output: {output_dir}")

    log("Starting model load...")
    tokenizer, model = load_model(ollama_name)
    log("Model fully ready")

    skipped = 0
    errors = 0
    timings = []

    log("Starting inference loop...")

    for i, scenario in enumerate(tqdm(scenarios, desc="Running")):

        if already_done(output_dir, i):
            skipped += 1
            continue

        try:
            response, elapsed = query_model(
                tokenizer, model, scenario["prompt"], i=i
            )

            save_result(output_dir, i, scenario, response, elapsed, trial)
            timings.append(elapsed)

        except Exception as e:
            errors += 1
            log(f"ERROR scenario {i}: {e}")

    log("Run complete")

    print("\n--- SUMMARY ---")
    print(f"Skipped: {skipped}")
    print(f"Errors:  {errors}")
    if timings:
        print(f"Avg time: {sum(timings)/len(timings):.2f}s")
        print(f"Total time: {sum(timings)/60:.2f} min")

    del model, tokenizer
    torch.cuda.empty_cache()
    log("GPU memory cleared")