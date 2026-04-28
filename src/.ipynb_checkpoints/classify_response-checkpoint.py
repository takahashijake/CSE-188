"""
classify_response.py

Reads all result JSONs from results/runs/{model}/ and classifies
each response into one of four categories:

correct - answer matches gold
abstained - model said "I don't know" or equivalent
wrong_document - answer matches a distractor doc, not the gold
hallucinated - answer matches nothing in the context

Output:
results/runs/{model}/trial_N/classifications.json (per trial)
results/runs/{model}/classifications_combined.json (averaged across trials)

Usage:
python3 src/classify_response.py --model qwen2.5:3b
python3 src/classify_response.py --model qwen2.5:3b --show-samples
"""

import json
import os
import re
import argparse
from tqdm import tqdm
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = "results/runs"

ABSTAIN_PHRASES = [
    "i dont know",        # after normalization, apostrophe is stripped
    "i do not know",
    "not in the documents",
    "not mentioned",
    "not provided",
    "cannot be found",
    "no information",
    "not stated",
    "doesnt say",         # normalized form of "doesn't say"
    "does not say",
    "not specified",
    "cannot determine",
    "not available",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, replace dashes/hyphens with spaces, strip punctuation."""
    text = text.lower()
    text = re.sub(r"[\u2013\u2014\-]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_abstention(response: str) -> bool:
    """Return True if the response is some form of 'I don't know'."""
    r = normalize(response)
    return any(phrase in r for phrase in ABSTAIN_PHRASES)


def is_correct(response: str, answer: str) -> bool:
    """
    Return True if the gold answer appears in the response,
    OR if token overlap between answer and response is high (>=0.5).
    This handles partial matches like 'Caucasus' vs 'Caucasus Mountains'.
    """
    r = normalize(response)
    a = normalize(answer)

    if a in r:
        return True

    tokens_a = set(a.split())
    tokens_r = set(r.split())
    if not tokens_a:
        return False
    overlap = len(tokens_a & tokens_r) / len(tokens_a)
    return overlap >= 0.5


def token_overlap(text_a: str, text_b: str) -> float:
    """
    Jaccard similarity on word tokens between two strings.
    Used to check if a response content matches a distractor doc.
    """
    tokens_a = set(normalize(text_a).split())
    tokens_b = set(normalize(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def matches_distractor(response: str, doc_texts: list,
                        gold_doc: str, threshold: float = 0.3) -> bool:
    """
    Return True if the response has significant token overlap with
    any non-gold document in the context.
    Threshold of 0.3 means 30% of tokens overlap — conservative.
    """
    non_gold_docs = [d for d in doc_texts if d != gold_doc]
    for doc in non_gold_docs:
        if token_overlap(response, doc) >= threshold:
            return True
    return False


def classify(result: dict) -> str:
    """
    Classify a single result dict into one of four categories.

    Priority order matters:
    1. Abstained - checked first, overrides everything
    2. Correct - checked before wrong/hallucinated
    3. Wrong-doc - response content came from a distractor
    4. Hallucinated - response invented something not in context
    """
    response = result["response"]
    answer = result["answer"]
    doc_texts = result["doc_texts"]
    gold_doc = result["gold_doc"]

    if is_abstention(response):
        return "abstained"

    if is_correct(response, answer):
        return "correct"

    if matches_distractor(response, doc_texts, gold_doc):
        return "wrong_document"

    return "hallucinated"


# ── Trial discovery ───────────────────────────────────────────────────────────

def find_trial_dirs(model_dir: str) -> list:
    """Return sorted list of trial subdirectory paths that contain results."""
    trials = []
    for entry in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, entry)
        if os.path.isdir(full) and entry.startswith("trial_"):
            result_files = [
                f for f in os.listdir(full)
                if f.startswith("result_") and f.endswith(".json")
            ]
            if result_files:
                trials.append(full)
    return trials


# ── Per-trial classification ──────────────────────────────────────────────────

def classify_trial(trial_dir: str, show_samples: bool = False) -> list:
    """Classify all results in one trial directory."""
    result_files = sorted([
        f for f in os.listdir(trial_dir)
        if f.startswith("result_") and f.endswith(".json")
    ])

    trial_name = os.path.basename(trial_dir)
    print(f"\nClassifying {len(result_files)} results for {trial_name}")

    classifications = []
    label_counts = {
        "correct": 0,
        "abstained": 0,
        "wrong_document": 0,
        "hallucinated": 0,
    }
    samples = {k: [] for k in label_counts}

    for fname in tqdm(result_files, desc="Classifying"):
        fpath = os.path.join(trial_dir, fname)
        with open(fpath) as f:
            result = json.load(f)

        label = classify(result)
        label_counts[label] += 1

        entry = {
            "scenario_id": result["scenario_id"],
            "question_id": result["question_id"],
            "question": result["question"],
            "answer": result["answer"],
            "length": result["length"],
            "position": result["position"],
            "response": result["response"],
            "label": label,
            "trial": result.get("trial", 1),
        }
        classifications.append(entry)
        samples[label].append(entry)

    # Save per-trial classifications
    output_path = os.path.join(trial_dir, "classifications.json")
    with open(output_path, "w") as f:
        json.dump(classifications, f, indent=2)

    # Print summary
    total = len(classifications)
    print(f"\n--- Classification Summary ({trial_name}) ---")
    print(f"{'Label':<20} {'Count':>6} {'Pct':>8}")
    print("-" * 36)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"{label:<20} {count:>6} {pct:>7.1f}%")
    print("-" * 36)
    print(f"{'TOTAL':<20} {total:>6}")
    print(f"\nSaved to: {output_path}")

    if show_samples:
        for label, sample_list in samples.items():
            print(f"\n{'='*50}")
            print(f"CATEGORY: {label.upper()} ({len(sample_list)} total)")
            print('='*50)
            for s in sample_list[:3]:
                print(f"  Q: {s['question']}")
                print(f"  Answer: {s['answer']}")
                print(f"  Response: {s['response']}")
                print(f"  Length: {s['length']} | Position: {s['position']}")
                print()

    return classifications


# ── Combined output ───────────────────────────────────────────────────────────

def combine_trials(all_trial_classifications: list, model_dir: str, model: str):
    """
    Merge all trial classifications into one combined file.
    Uses majority voting per condition, stores rates for metrics.py.
    """
    condition_groups = defaultdict(list)

    for trial_classifications in all_trial_classifications:
        for entry in trial_classifications:
            key = (entry["question_id"], entry["length"], entry["position"])
            condition_groups[key].append(entry)

    combined = []
    label_totals = defaultdict(int)

    for key, entries in condition_groups.items():
        q_id, length, position = key
        total_trials = len(entries)

        label_counts = defaultdict(int)
        for e in entries:
            label_counts[e["label"]] += 1

        majority_label = max(label_counts, key=label_counts.get)
        rates = {
            label: count / total_trials
            for label, count in label_counts.items()
        }

        combined.append({
            "question_id": q_id,
            "question": entries[0]["question"],
            "answer": entries[0]["answer"],
            "length": length,
            "position": position,
            "trials": total_trials,
            "majority_label": majority_label,
            "label_counts": dict(label_counts),
            "label_rates": rates,
            "responses": [e["response"] for e in entries],
        })

        label_totals[majority_label] += 1

    output_path = os.path.join(model_dir, "classifications_combined.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    total = len(combined)
    print(f"\n--- Combined Summary ({model}, "
          f"{len(all_trial_classifications)} trials) ---")
    print(f"{'Label':<20} {'Count':>6} {'Pct':>8}")
    print("-" * 36)
    for label, count in sorted(label_totals.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"{label:<20} {count:>6} {pct:>7.1f}%")
    print("-" * 36)
    print(f"{'TOTAL':<20} {total:>6}")
    print(f"\nSaved to: {output_path}")

    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="qwen2.5:3b",
        help="Model name (must match folder in results/runs/)"
    )
    parser.add_argument(
        "--show-samples", action="store_true",
        help="Print sample responses for each category (for manual validation)"
    )
    args = parser.parse_args()

    model_tag = args.model.replace(":", "_").replace("/", "_")
    model_dir = os.path.join(RESULTS_DIR, model_tag)

    if not os.path.exists(model_dir):
        print(f"ERROR: No results found at {model_dir}")
        print("Run run_experiment.py first.")
        return

    trial_dirs = find_trial_dirs(model_dir)

    if not trial_dirs:
        print(f"ERROR: No trial subdirectories found in {model_dir}")
        print("Expected folders named trial_1, trial_2, etc.")
        return

    print(f"Model: {args.model}")
    print(f"Found {len(trial_dirs)} trial(s): "
          f"{[os.path.basename(t) for t in trial_dirs]}")

    all_classifications = []
    for trial_dir in trial_dirs:
        trial_classifications = classify_trial(trial_dir, args.show_samples)
        all_classifications.append(trial_classifications)

    print(f"\n{'='*50}")
    print("Combining trials...")
    combine_trials(all_classifications, model_dir, args.model)

    print(f"\nDone.")
    print(f"Per-trial: {model_dir}/trial_N/classifications.json")
    print(f"Combined: {model_dir}/classifications_combined.json")


if __name__ == "__main__":
    main()