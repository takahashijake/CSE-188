"""Aggregate classified response rates and create portfolio figures.

The script applies the heuristics in ``classify_response.py`` directly to raw
per-scenario JSON files. It records malformed inputs rather than hiding them.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from classify_response import classify


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results/runs"
SUMMARY_DIR = REPO_ROOT / "results/summary"
FIGURES_DIR = REPO_ROOT / "figures"

CATEGORIES = ("correct", "abstained", "hallucinated", "wrong_document")
POSITIONS = ("first", "middle", "last", "absent")
LENGTHS = (1, 5, 10, 20)
MODEL_NAMES = {
    "qwen2.5_3b": "Qwen2.5-3B",
    "qwen2.5_7b": "Qwen2.5-7B",
    "qwen2.5_14b": "Qwen2.5-14B",
    "mistral_7b": "Mistral-7B",
    "llama3.1_8b": "Llama3.1-8B",
    "gemma2_9b": "Gemma2-9B",
    "phi3_mini": "Phi3-Mini",
    "phi3_medium": "Phi3-Medium",
}
MODEL_ORDER = tuple(MODEL_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate raw runs into curated CSV summaries and figures."
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Skip malformed JSON records and disclose them in provenance.json.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def collect_results(results_dir):
    overall = defaultdict(Counter)
    by_condition = defaultdict(Counter)
    source_counts = Counter()
    trial_ids = defaultdict(set)
    invalid = []

    for path in sorted(results_dir.glob("*/trial_*/result_*.json")):
        model_key = path.parents[1].name
        source_counts[model_key] += 1
        try:
            with path.open() as f:
                result = json.load(f)
            label = classify(result)
            length = int(result["length"])
            position = result["position"]
            trial = int(result.get("trial", path.parent.name.split("_")[-1]))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            invalid.append({"path": str(path.relative_to(REPO_ROOT)), "error": str(exc)})
            continue

        overall[model_key][label] += 1
        by_condition[(model_key, length, position)][label] += 1
        trial_ids[model_key].add(trial)

    return overall, by_condition, source_counts, trial_ids, invalid


def rate(count, total):
    return count / total if total else 0.0


def ordered_models(overall):
    known = [key for key in MODEL_ORDER if key in overall]
    return known + sorted(set(overall) - set(known))


def write_summaries(summary_dir, overall, by_condition, source_counts,
                    trial_ids, invalid):
    summary_dir.mkdir(parents=True, exist_ok=True)
    models = ordered_models(overall)

    overall_path = summary_dir / "overall_response_rates.csv"
    fields = ["model", "model_key", "valid_responses", "source_files", "trials"]
    for category in CATEGORIES:
        fields.extend((f"{category}_count", f"{category}_rate"))

    with overall_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model_key in models:
            counts = overall[model_key]
            total = sum(counts.values())
            row = {
                "model": MODEL_NAMES.get(model_key, model_key),
                "model_key": model_key,
                "valid_responses": total,
                "source_files": source_counts[model_key],
                "trials": len(trial_ids[model_key]),
            }
            for category in CATEGORIES:
                row[f"{category}_count"] = counts[category]
                row[f"{category}_rate"] = f"{rate(counts[category], total):.6f}"
            writer.writerow(row)

    condition_path = summary_dir / "response_rates_by_condition.csv"
    condition_fields = ["model", "model_key", "context_length", "gold_position",
                        "valid_responses"]
    for category in CATEGORIES:
        condition_fields.extend((f"{category}_count", f"{category}_rate"))

    with condition_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=condition_fields)
        writer.writeheader()
        for model_key in models:
            for length in LENGTHS:
                for position in POSITIONS:
                    counts = by_condition[(model_key, length, position)]
                    total = sum(counts.values())
                    row = {
                        "model": MODEL_NAMES.get(model_key, model_key),
                        "model_key": model_key,
                        "context_length": length,
                        "gold_position": position,
                        "valid_responses": total,
                    }
                    for category in CATEGORIES:
                        row[f"{category}_count"] = counts[category]
                        row[f"{category}_rate"] = f"{rate(counts[category], total):.6f}"
                    writer.writerow(row)

    provenance = {
        "source": "results/runs/*/trial_*/result_*.json",
        "classification_script": "src/classify_response.py",
        "models": models,
        "source_files": sum(source_counts.values()),
        "valid_responses": sum(sum(c.values()) for c in overall.values()),
        "invalid_responses": len(invalid),
        "invalid_files": invalid,
        "notes": [
            "Rates pool valid responses across trials; they are not majority-vote rates.",
            "The labels are lexical heuristics and are not human factuality judgments.",
        ],
    }
    with (summary_dir / "provenance.json").open("w") as f:
        json.dump(provenance, f, indent=2)

    return models


def create_plots(figures_dir, models, overall, by_condition):
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [MODEL_NAMES.get(key, key) for key in models]
    colors = {
        "correct": "#4CAF50",
        "abstained": "#2196F3",
        "hallucinated": "#F44336",
        "wrong_document": "#FF9800",
    }

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bottoms = np.zeros(len(models))
    for category in CATEGORIES:
        values = []
        for model_key in models:
            counts = overall[model_key]
            values.append(rate(counts[category], sum(counts.values())))
        ax.bar(labels, values, bottom=bottoms, label=category.replace("_", " ").title(),
               color=colors[category])
        bottoms += np.array(values)
    ax.set_title("Response categories by model (all conditions)")
    ax.set_ylabel("Share of valid responses")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.savefig(figures_dir / "response_categories_by_model.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True,
                             sharex=True, sharey=True)
    matrices = []
    for model_key in models:
        matrix = np.array([
            [rate(by_condition[(model_key, length, position)]["hallucinated"],
                  sum(by_condition[(model_key, length, position)].values()))
             for length in LENGTHS]
            for position in POSITIONS
        ])
        matrices.append(matrix)
    vmax = max((matrix.max() for matrix in matrices), default=0.01) or 0.01

    image = None
    for ax, model_key, matrix in zip(axes.flat, models, matrices):
        image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="Reds", aspect="auto")
        ax.set_title(MODEL_NAMES.get(model_key, model_key))
        ax.set_xticks(range(len(LENGTHS)), LENGTHS)
        ax.set_yticks(range(len(POSITIONS)), [p.title() for p in POSITIONS])
        for row in range(len(POSITIONS)):
            for column in range(len(LENGTHS)):
                value = matrix[row, column]
                text_color = "white" if value > vmax * 0.55 else "#333333"
                ax.text(column, row, f"{value:.1%}", ha="center", va="center",
                        color=text_color, fontsize=9)
    fig.suptitle("Heuristic hallucination rate by gold position and document count",
                 fontsize=16)
    fig.supxlabel("Documents in context")
    fig.supylabel("Gold-document position")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
        colorbar.set_label("Hallucination rate")
        colorbar.ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.savefig(figures_dir / "hallucination_heatmap.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    overall, by_condition, source_counts, trial_ids, invalid = collect_results(results_dir)
    if not overall:
        raise SystemExit(f"No valid result JSON files found under {results_dir}")
    if invalid and not args.allow_incomplete:
        examples = "\n".join(f"- {item['path']}: {item['error']}" for item in invalid[:5])
        raise SystemExit(
            f"Found {len(invalid)} malformed result file(s). Re-run the missing cases or "
            f"pass --allow-incomplete to disclose and skip them:\n{examples}"
        )

    models = write_summaries(
        args.summary_dir.resolve(), overall, by_condition,
        source_counts, trial_ids, invalid,
    )
    if not args.no_plots:
        create_plots(args.figures_dir.resolve(), models, overall, by_condition)

    valid = sum(sum(counts.values()) for counts in overall.values())
    print(f"Summarized {valid} valid responses across {len(models)} models.")
    if invalid:
        print(f"Skipped and disclosed {len(invalid)} malformed response(s).")


if __name__ == "__main__":
    main()
