"""
build_contexts.py

Loads Natural Questions, picks 50 clean questions, and builds
context windows for every (length, position) combination.

Output: data/nq_open/contexts.json
    A list of scenario dicts, one per (question, length, position).
"""

import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

NUM_QUESTIONS   = 50        # how many questions to sample
CONTEXT_LENGTHS = [1, 5, 10, 20]   # number of documents in context window
POSITIONS       = ["first", "middle", "last", "absent"]
RANDOM_SEED     = 42
OUTPUT_PATH     = "data/nq_open/contexts.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_in_doc(answer: str, doc: str) -> bool:
    """Check if the answer string appears in the document."""
    return normalize(answer) in normalize(doc)


def insert_at_position(docs: list, gold_doc: str, position: str) -> list:
    """
    Insert gold_doc into docs list at the specified position.
    'absent' means gold_doc is never inserted.
    docs here are the distractors only (gold not included yet).
    """
    if position == "absent":
        return docs[:]                          # gold never appears
    elif position == "first":
        return [gold_doc] + docs
    elif position == "last":
        return docs + [gold_doc]
    elif position == "middle":
        mid = len(docs) // 2
        return docs[:mid] + [gold_doc] + docs[mid:]
    else:
        raise ValueError(f"Unknown position: {position}")


def format_context(docs: list) -> str:
    """Turn a list of document strings into a numbered context block."""
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"Document [{i}]: {doc.strip()}")
    return "\n\n".join(lines)


def build_prompt(context: str, question: str) -> str:
    return (
        "Answer the question using only the documents below.\n"
        "If the answer is not in the documents, say \"I don't know.\"\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer briefly:"
    )

# ── Data loading ──────────────────────────────────────────────────────────────

def load_nq(num_questions: int, seed: int):
    """
    Load Natural Questions (open version) and return a clean subset.
    Each item: {question, answer, gold_doc}
    Filters out questions with missing context or multi-word ambiguous answers.
    """
    print("Loading Natural Questions dataset (this may take a minute)...")
    ds = load_dataset("nq_open", split="validation")  # ~3.6k examples

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)

    clean = []
    for idx in indices:
        item = ds[idx]
        question = item["question"].strip()
        answers  = item["answer"]  # list of valid answers

        # Skip questions with no short answer
        if not answers:
            continue

        # Pick the shortest answer (most unambiguous)
        answer = min(answers, key=len).strip()

        # Skip if answer is too long (likely a phrase, not a fact)
        if len(answer.split()) > 5:
            continue

        # Skip if answer is too short (single letter, etc.)
        if len(answer) < 2:
            continue

        clean.append({
            "question": question,
            "answer":   answer,
        })

        if len(clean) == num_questions:
            break

    print(f"Selected {len(clean)} clean questions.")
    return clean


def load_distractor_pool(questions: list, seed: int):
    """
    Load the full NQ validation set and build a pool of distractor documents.
    A distractor for question Q is any Wikipedia paragraph from another
    question that does NOT contain Q's answer.

    Returns: dict mapping question index -> list of distractor strings
    """
    print("Building distractor pool...")
    ds = load_dataset("nq_open", split="validation")

    # Collect all unique non-empty paragraphs from the dataset
    # NQ open doesn't include full passages, so we synthesize distractors
    # from other questions' answer contexts
    all_paragraphs = []
    for item in ds:
        q = item["question"].strip()
        a = item["answer"]
        if a and q:
            # Use the question + answer as a short "document"
            para = f"{q} {a[0]}" if a else q
            all_paragraphs.append(para.strip())

    random.seed(seed)
    random.shuffle(all_paragraphs)

    # For each question, find distractors that don't contain its answer
    distractor_pool = {}
    for i, q_item in enumerate(tqdm(questions, desc="Building distractors")):
        answer = q_item["answer"]
        distractors = [
            p for p in all_paragraphs
            if not answer_in_doc(answer, p)
        ]
        # Keep top 50 so we have plenty to sample from
        distractor_pool[i] = distractors[:50]

    return distractor_pool

# ── Main build ────────────────────────────────────────────────────────────────

def build_contexts(questions: list, distractor_pool: dict) -> list:
    """
    For each (question, length, position) combination, build a scenario dict.

    Each scenario:
    {
        question_id:  int,
        question:     str,
        answer:       str,
        length:       int,
        position:     str,
        gold_doc:     str,
        prompt:       str,   <- ready to send to LLM
        doc_texts:    list,  <- all docs in order (for classifier later)
        gold_index:   int,   <- which doc index is gold (-1 if absent)
    }
    """
    scenarios = []

    for q_id, q_item in enumerate(tqdm(questions, desc="Building scenarios")):
        question    = q_item["question"]
        answer      = q_item["answer"]
        distractors = distractor_pool[q_id]

        # Gold document: a sentence stating the answer directly
        gold_doc = f"The answer to '{question}' is {answer}."

        for length in CONTEXT_LENGTHS:
            # We need (length - 1) distractors for non-absent, length for absent
            needed = length  # always sample `length` distractors for safety
            if len(distractors) < needed:
                # Skip this length if not enough distractors
                continue

            sampled_distractors = random.sample(distractors, needed)

            for position in POSITIONS:
                if position == "absent":
                    doc_list  = sampled_distractors[:length]
                    gold_index = -1
                else:
                    # Use length-1 distractors + 1 gold = length total
                    distractor_subset = sampled_distractors[:length - 1]
                    doc_list = insert_at_position(
                        distractor_subset, gold_doc, position
                    )
                    gold_index = doc_list.index(gold_doc)

                context = format_context(doc_list)
                prompt  = build_prompt(context, question)

                scenarios.append({
                    "question_id": q_id,
                    "question":    question,
                    "answer":      answer,
                    "length":      length,
                    "position":    position,
                    "gold_doc":    gold_doc,
                    "prompt":      prompt,
                    "doc_texts":   doc_list,
                    "gold_index":  gold_index,
                })

    return scenarios


def main():
    os.makedirs("data/nq_open", exist_ok=True)
    random.seed(RANDOM_SEED)

    questions       = load_nq(NUM_QUESTIONS, RANDOM_SEED)
    distractor_pool = load_distractor_pool(questions, RANDOM_SEED)
    scenarios       = build_contexts(questions, distractor_pool)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nDone. {len(scenarios)} scenarios saved to {OUTPUT_PATH}")
    print(f"Breakdown: {NUM_QUESTIONS} questions × "
          f"{len(CONTEXT_LENGTHS)} lengths × "
          f"{len(POSITIONS)} positions = "
          f"{NUM_QUESTIONS * len(CONTEXT_LENGTHS) * len(POSITIONS)} expected")


if __name__ == "__main__":
    main()