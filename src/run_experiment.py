"""
run_experiment.py — robust, reproducible multi-model experiment runner

Fixes:
- Saves ALL fields required for classification
- Deterministic inference (no randomness)
- Stable GPU memory handling (L4 safe)
- Consistent chat formatting across models
- Clean resume support
"""

import json
import os
import time
import argparse
import gc
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ── GPU stability fix ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(0)

# ── Paths ────────────────────────────────────────────────────────────────────
SCENARIOS_PATH = "data/nq_open/contexts.json"
RESULTS_DIR = "results/runs"
MAX_NEW_TOKENS = 100

# ── Model map ───────────────────────────────────────────────────────────────
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

# Models that need quantization to fit
NEEDS_4BIT = {"qwen2.5:14b", "phi3:medium"}

# ── Utils ────────────────────────────────────────────────────────────────────

def sanitize(name):
    return name.replace(":", "_").replace("/", "_")


def already_done(outdir, sid):
    return os.path.exists(os.path.join(outdir, f"result_{sid:04d}.json"))


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_input(tokenizer, prompt):
    """Safe chat formatting across models"""
    try:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    except Exception:
        return tokenizer(
            f"User: {prompt}\nAssistant:",
            return_tensors="pt"
        ).input_ids


# ── Load model ───────────────────────────────────────────────────────────────

def load_model(model_key):
    hf_name = MODEL_MAP[model_key]
    use_4bit = model_key in NEEDS_4BIT

    print(f"\nLoading: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        print("Using 4-bit quantization")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "22GiB"},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "22GiB"},
        )

    model.eval()
    print("Loaded on:", next(model.parameters()).device)

    return tokenizer, model


# ── Inference ────────────────────────────────────────────────────────────────

def query(tokenizer, model, prompt):
    inputs = build_input(tokenizer, prompt)

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    start = time.time()

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,        # 🔥 deterministic
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start

    response = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response, elapsed


# ── Save ─────────────────────────────────────────────────────────────────────

def save(outdir, sid, scenario, response, elapsed, trial):
    """Save FULL schema required by classifier"""

    data = {
        # identity
        "scenario_id": sid,
        "question_id": scenario["question_id"],

        # question info
        "question": scenario["question"],
        "answer": scenario["answer"],
        "length": scenario["length"],
        "position": scenario["position"],

        # context
        "prompt": scenario["prompt"],
        "doc_texts": scenario["doc_texts"],
        "gold_doc": scenario["gold_doc"],
        "gold_index": scenario["gold_index"],

        # model output
        "response": response,
        "elapsed_s": round(elapsed, 2),
        "trial": trial,
    }

    path = os.path.join(outdir, f"result_{sid:04d}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(SCENARIOS_PATH) as f:
        scenarios = json.load(f)

    if args.limit:
        scenarios = scenarios[:args.limit]

    model_dir = os.path.join(RESULTS_DIR, sanitize(args.model))
    os.makedirs(model_dir, exist_ok=True)

    outdir = os.path.join(model_dir, f"trial_{args.trial}")
    os.makedirs(outdir, exist_ok=True)

    clear_memory()
    tokenizer, model = load_model(args.model)

    print(f"\nRunning {args.model} | {len(scenarios)} samples\n")

    skipped = 0
    times = []

    for i, sc in enumerate(tqdm(scenarios)):
        if already_done(outdir, i):
            skipped += 1
            continue

        try:
            resp, t = query(tokenizer, model, sc["prompt"])
            save(outdir, i, sc, resp, t, args.trial)
            times.append(t)

        except Exception as e:
            print(f"Error {i}: {e}")

    clear_memory()

    print("\nDONE")
    print("Skipped:", skipped)
    if times:
        print("Avg time:", sum(times)/len(times))


if __name__ == "__main__":
    main()
