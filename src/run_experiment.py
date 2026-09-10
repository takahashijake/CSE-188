"""Run sampled generation trials for the context-position experiment.

Each scenario gets a deterministic per-trial seed. This preserves stochastic
decoding while making interrupted and resumed runs reproducible.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

# Configure CUDA allocation before importing torch.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = REPO_ROOT / "data/nq_open/contexts.json"
RESULTS_DIR = REPO_ROOT / "results/runs"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
RANDOM_SEED = 42

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
    return (outdir / f"result_{sid:04d}.json").exists()


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
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

def load_model(model_key, revision=None):
    hf_name = MODEL_MAP[model_key]
    use_4bit = model_key in NEEDS_4BIT

    print(f"\nLoading: {hf_name}")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, revision=revision)

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
            revision=revision,
            quantization_config=bnb,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "22GiB"},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "22GiB"},
        )

    model.eval()
    print("Loaded on:", next(model.parameters()).device)

    return tokenizer, model


# ── Inference ────────────────────────────────────────────────────────────────

def query(tokenizer, model, prompt, max_new_tokens, temperature):
    inputs = build_input(tokenizer, prompt)

    device = next(model.parameters()).device
    inputs = inputs.to(device)

    start = time.time()

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start

    response = tokenizer.decode(
        output[0][inputs.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return response, elapsed


# ── Save ─────────────────────────────────────────────────────────────────────

def save(outdir, sid, scenario, response, elapsed, trial, run_metadata):
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
        "model_key": run_metadata["model_key"],
        "model_id": run_metadata["model_id"],
        "model_revision": run_metadata.get("resolved_model_revision"),
        "scenario_seed": run_metadata["base_seed"] + trial * 100_000 + sid,
        "generation_config": run_metadata["generation_config"],
    }

    path = outdir / f"result_{sid:04d}.json"
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run_manifest(outdir, metadata):
    """Create a manifest and reject resumes with incompatible settings."""
    manifest_path = outdir / "run_config.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            existing = json.load(f)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in metadata.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Existing run manifest does not match requested settings: {mismatches}"
            )
    elif any(outdir.glob("result_*.json")):
        raise RuntimeError(
            f"Refusing to resume legacy results without a manifest in {outdir}. "
            "Choose a new trial/output directory."
        )
    else:
        with manifest_path.open("w") as f:
            json.dump(metadata, f, indent=2)


def finalize_run_manifest(outdir, metadata, model):
    """Record the resolved Hub commit and prevent revision-mixed resumes."""
    manifest_path = outdir / "run_config.json"
    resolved_revision = getattr(model.config, "_commit_hash", None)
    with manifest_path.open() as f:
        existing = json.load(f)
    previous_revision = existing.get("resolved_model_revision")
    if previous_revision and previous_revision != resolved_revision:
        raise RuntimeError(
            "Resolved model revision changed for an existing trial: "
            f"{previous_revision} != {resolved_revision}"
        )
    metadata["resolved_model_revision"] = resolved_revision
    with manifest_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def package_versions():
    versions = {}
    for package in ("accelerate", "bitsandbytes", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one model/trial over generated context scenarios."
    )
    parser.add_argument(
        "--model", choices=sorted(MODEL_MAP), default="qwen2.5:3b",
        help="Short model key (default: qwen2.5:3b).",
    )
    parser.add_argument("--trial", type=int, default=1, help="Positive trial number.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N scenarios for a smoke test.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--revision", default=None,
        help="Optional Hugging Face branch, tag, or commit hash.",
    )
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trial < 1:
        raise ValueError("--trial must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero for sampling")

    scenarios_path = args.scenarios.resolve()
    with scenarios_path.open() as f:
        scenarios = json.load(f)

    full_scenario_count = len(scenarios)
    if args.limit is not None:
        scenarios = scenarios[:args.limit]

    model_dir = args.results_dir.resolve() / sanitize(args.model)
    outdir = model_dir / f"trial_{args.trial}"
    outdir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "model_key": args.model,
        "model_id": MODEL_MAP[args.model],
        "requested_model_revision": args.revision,
        "trial": args.trial,
        "base_seed": args.seed,
        "generation_config": {
            "do_sample": True,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        "scenarios_sha256": file_sha256(scenarios_path),
        "scenario_count": full_scenario_count,
        "limit": args.limit,
        "package_versions": package_versions(),
    }
    prepare_run_manifest(outdir, run_metadata)

    clear_memory()
    tokenizer, model = load_model(args.model, args.revision)
    finalize_run_manifest(outdir, run_metadata, model)

    print(f"\nRunning {args.model} | {len(scenarios)} samples\n")

    skipped = 0
    failures = []
    times = []

    for i, sc in enumerate(tqdm(scenarios)):
        if already_done(outdir, i):
            skipped += 1
            continue

        try:
            scenario_seed = args.seed + args.trial * 100_000 + i
            set_seed(scenario_seed)
            resp, t = query(
                tokenizer, model, sc["prompt"],
                args.max_new_tokens, args.temperature,
            )
            save(outdir, i, sc, resp, t, args.trial, run_metadata)
            times.append(t)

        except Exception as e:
            failures.append(i)
            print(f"Error {i}: {e}")

    clear_memory()

    print("\nDONE")
    print("Skipped:", skipped)
    print("Failed:", len(failures))
    if times:
        print("Avg time:", sum(times)/len(times))
    if failures:
        raise SystemExit(f"Run incomplete; failed scenario IDs: {failures}")


if __name__ == "__main__":
    main()
