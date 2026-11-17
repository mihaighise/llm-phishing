"""
metrics.py — Step 6 of the LLM phishing project: classification metrics.

WHAT THIS SCRIPT DOES
---------------------
Reads every saved model answer file for the test set, e.g.

    results/raw/llama31_8b_test_all_v2.jsonl

(one per model, downloaded from Google Drive), and computes for each model:

    - accuracy, precision, recall, F1  ("phishing" is the positive class)
    - 95% confidence intervals for accuracy and F1 (bootstrap, see below)
    - parse rate (share of answers that were valid JSON with a valid label)
    - a confusion matrix
    - accuracy per source dataset (Nazario, Nigerian, SpamAssassin, ...)

It also adds the bag-of-words Naive Bayes classifier from shortcut_check.py
as a BASELINE row, because the shortcut check showed that word counting
alone scores ~98.6% on this dataset. Every LLM number should be read next
to it.

TWO SCORING RULES
-----------------
    strict   — an answer that could not be parsed counts as WRONG. This is
               the main result; the rule was fixed before any test run.
    lenient  — if the JSON was broken but the parser could still find the
               label (e.g. Qwen's answers missing only the final "}"), that
               label is used. Reported as a secondary number only.

TWO TEST SETS
-------------
    all      — all 1,000 test emails.
    dedup    — the test emails WITHOUT a near-duplicate in the training data.
               build_dataset.py removes exact duplicates only, so some
               templated scams appear in both training and test with small
               changes. A fine-tuned model could simply remember those, so
               we also report scores without them.

HOW TO RUN (from the repository root)
-------------------------------------
    python3 src/metrics.py

Outputs:
    results/metrics.md    — tables ready to paste into the write-up
    results/metrics.json  — every number, for plots and later analysis

Only the Python standard library is used, so no `pip install` is needed.
"""

# ---------------------------------------------------------------------------
# Imports — standard library, plus our own modules in src/.
# ---------------------------------------------------------------------------
import json       # reads answer files, writes metrics.json
import random     # bootstrap resampling (with a fixed seed)
import re         # splits text into words for near-duplicate detection
from collections import Counter, defaultdict  # counting helpers
from pathlib import Path  # OS-independent file paths

from prompts import PROMPT_VERSION, PHISHING, LEGITIMATE, format_email
from shortcut_check import NaiveBayes, load_split

# Paths are relative to this file, so the script works from any directory.
REPO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_DIR / "results" / "raw"
OUT_JSON = REPO_DIR / "results" / "metrics.json"
OUT_MD = REPO_DIR / "results" / "metrics.md"

# Readable names for the tables. A results file whose model key is not
# listed here is still included, just under its raw key.
MODEL_NAMES = {
    "naive_bayes": "Naive Bayes (bag-of-words baseline)",
    "llama31_8b": "Llama 3.1 8B",
    "qwen35_9b": "Qwen 3.5 9B",
    "mistral_7b": "Mistral 7B",
}

# Source datasets in a fixed, readable order (phishing first).
SOURCE_NAMES = {
    "nazario": "Nazario phishing",
    "nigerian": "Nigerian fraud",
    "spamassassin": "SpamAssassin legitimate",
    "legit_naz5": "Nazario_5 legitimate",
    "legit_nig5": "Nigerian_5 legitimate",
}

# Near-duplicate settings (see near_duplicate_ids below).
SHINGLE_WORDS = 8         # compare emails by overlapping 8-word sequences
DUPLICATE_THRESHOLD = 0.8  # >= 80% shared sequences = near-duplicate

# Bootstrap settings (see bootstrap_ci below).
BOOTSTRAP_ROUNDS = 1000
SEED = 42


# ===========================================================================
# 1. LOADING ANSWERS
# ===========================================================================

def load_answers(test_ids: set) -> dict:
    """Load every test-set answer file for the current prompt version.

    Returns {model_key: {email_id: answer_row}}. Files from an older prompt
    version are ignored, so old and new results are never mixed.
    """
    suffix = f"_test_all_{PROMPT_VERSION}.jsonl"
    answers = {}
    for path in sorted(RAW_DIR.glob(f"*{suffix}")):
        model_key = path.name[: -len(suffix)]  # "llama31_8b_test_all_v2.jsonl" -> "llama31_8b"
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        by_id = {r["id"]: r for r in rows}

        # Sanity checks: a file with missing or extra emails would make the
        # models' scores incomparable, so we warn loudly.
        missing = test_ids - by_id.keys()
        extra = by_id.keys() - test_ids
        if missing or extra or len(by_id) != len(rows):
            print(f"WARNING {path.name}: {len(missing)} test emails missing, "
                  f"{len(extra)} unknown ids, {len(rows) - len(by_id)} duplicates")
        answers[model_key] = by_id

    if not answers:
        print(f"No files matching *{suffix} in {RAW_DIR}. Download the "
              "results from Google Drive first.")
    return answers


def naive_bayes_answers(train: list, test: list) -> dict:
    """Train the bag-of-words baseline on train_full and return its test
    predictions in the same shape as an LLM answer file."""
    model = NaiveBayes().fit([format_email(e) for e in train],
                             [e["label"] for e in train])
    # A word-count model always produces a label, so parse_ok is always True.
    return {e["id"]: {"label": model.predict(format_email(e)), "parse_ok": True}
            for e in test}


# ===========================================================================
# 2. NEAR-DUPLICATES BETWEEN TRAINING AND TEST
# ===========================================================================
#
# How do we decide that two emails are "almost the same"? We cut each body
# into overlapping 8-word sequences ("shingles"):
#
#     "please send your bank details today" with 3-word shingles ->
#     {"please send your", "send your bank", "your bank details", ...}
#
# and compare the two sets with the Jaccard similarity:
#
#     shared shingles / all distinct shingles of both emails
#
# 1.0 = identical text, 0.0 = no 8-word sequence in common. Two copies of a
# scam template with a different name and phone number still share most of
# their 8-word sequences, so they score high.

def _shingles(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    # Very short emails have fewer than 8 words; use the whole text then.
    return {" ".join(words[i:i + SHINGLE_WORDS])
            for i in range(max(1, len(words) - SHINGLE_WORDS + 1))}


def near_duplicate_ids(train: list, test: list) -> set:
    """Ids of test emails whose body is >= 80% similar to ANY training email."""
    train_shingles = [_shingles(e["body"]) for e in train]

    # Index: shingle -> training emails containing it. This lets us compare
    # each test email only with training emails sharing at least one
    # shingle, instead of all ~8,000 of them.
    index = defaultdict(set)
    for i, shingles in enumerate(train_shingles):
        for s in shingles:
            index[s].add(i)

    duplicates = set()
    for e in test:
        shingles = _shingles(e["body"])
        shared = Counter(i for s in shingles for i in index.get(s, ()))
        for i, n_shared in shared.items():
            jaccard = n_shared / len(shingles | train_shingles[i])
            if jaccard >= DUPLICATE_THRESHOLD:
                duplicates.add(e["id"])
                break
    return duplicates


# ===========================================================================
# 3. METRICS
# ===========================================================================

def predicted_label(answer: dict, rule: str):
    """The label we score: None (= unparseable, counts as wrong) under the
    strict rule when parsing failed, otherwise whatever label was found."""
    if rule == "strict" and not answer.get("parse_ok"):
        return None
    return answer.get("label")


def scores(preds: list, truth: list) -> dict:
    """Accuracy, precision, recall and F1 from parallel lists of predicted
    and true labels. None in `preds` = unparseable = wrong."""
    tp = sum(p == PHISHING and t == PHISHING for p, t in zip(preds, truth))
    fp = sum(p == PHISHING and t == LEGITIMATE for p, t in zip(preds, truth))
    fn = sum(p != PHISHING and t == PHISHING for p, t in zip(preds, truth))
    # precision: of the emails called phishing, how many really were?
    # recall:    of the real phishing emails, how many were caught?
    # F1:        their harmonic mean — high only if BOTH are high.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = sum(p == t for p, t in zip(preds, truth)) / len(truth)
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1}


def bootstrap_ci(preds: list, truth: list) -> dict:
    """95% confidence intervals for accuracy and F1.

    With 1,000 test emails, a 1-point difference between two models could
    be luck. The bootstrap estimates how much a score would wobble with a
    different sample of emails: draw 1,000 emails WITH replacement (some
    twice, some not at all), score them, repeat 1,000 times, and take the
    middle 95% of those scores. If two models' intervals overlap a lot,
    don't claim one is better.
    """
    rng = random.Random(SEED)  # fixed seed -> same intervals on every run
    n = len(truth)
    samples = defaultdict(list)
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = [rng.randrange(n) for _ in range(n)]
        s = scores([preds[i] for i in idx], [truth[i] for i in idx])
        samples["accuracy"].append(s["accuracy"])
        samples["f1"].append(s["f1"])
    ci = {}
    for metric, values in samples.items():
        values.sort()
        ci[metric] = [values[int(0.025 * BOOTSTRAP_ROUNDS)],
                      values[int(0.975 * BOOTSTRAP_ROUNDS) - 1]]
    return ci


def evaluate(answers: dict, emails: list, rule: str) -> dict:
    """Every metric for one model, one scoring rule, one set of emails."""
    preds = [predicted_label(answers[e["id"]], rule) for e in emails]
    truth = [e["label"] for e in emails]
    result = scores(preds, truth)
    result["ci95"] = bootstrap_ci(preds, truth)
    result["n"] = len(emails)
    result["parse_rate"] = sum(bool(answers[e["id"]].get("parse_ok"))
                               for e in emails) / len(emails)

    # Confusion matrix: rows = true label, columns = predicted label, with
    # an extra column for answers we could not read.
    matrix = {t: Counter() for t in (PHISHING, LEGITIMATE)}
    for p, t in zip(preds, truth):
        matrix[t][p if p is not None else "unparseable"] += 1
    result["confusion"] = {t: {c: matrix[t][c]
                               for c in (PHISHING, LEGITIMATE, "unparseable")}
                           for t in matrix}
    result["false_alarms"] = matrix[LEGITIMATE][PHISHING]
    result["missed_phishing"] = matrix[PHISHING][LEGITIMATE] + matrix[PHISHING]["unparseable"]

    # Accuracy per source dataset: [correct, total].
    per_source = defaultdict(lambda: [0, 0])
    for p, e in zip(preds, emails):
        per_source[e["source"]][0] += p == e["label"]
        per_source[e["source"]][1] += 1
    result["per_source"] = {s: per_source[s] for s in SOURCE_NAMES if s in per_source}
    return result


# ===========================================================================
# 4. MARKDOWN TABLES
# ===========================================================================

def _name(key: str) -> str:
    # Fine-tuned models are saved as "<base>_ft_<training set>", e.g.
    # "llama31_8b_ft_train_3000" -> "Llama 3.1 8B fine-tuned (train_3000)".
    if "_ft_" in key:
        base, train_set = key.split("_ft_", 1)
        return f"{MODEL_NAMES.get(base, base)} fine-tuned ({train_set})"
    return MODEL_NAMES.get(key, key)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main_table(results: dict, subset: str, rule: str) -> str:
    """One row per model: the headline numbers with 95% intervals."""
    lines = ["| Model | n | Parsed | Accuracy (95% CI) | Precision | Recall "
             "| F1 (95% CI) | False alarms | Missed phishing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, by_subset in results.items():
        r = by_subset[subset][rule]
        acc_lo, acc_hi = r["ci95"]["accuracy"]
        f1_lo, f1_hi = r["ci95"]["f1"]
        lines.append(
            f"| {_name(key)} | {r['n']} | {_pct(r['parse_rate'])} "
            f"| {_pct(r['accuracy'])} ({_pct(acc_lo)}–{_pct(acc_hi)}) "
            f"| {r['precision']:.3f} | {r['recall']:.3f} "
            f"| {r['f1']:.3f} ({f1_lo:.3f}–{f1_hi:.3f}) "
            f"| {r['false_alarms']} | {r['missed_phishing']} |")
    return "\n".join(lines)


def source_table(results: dict, subset: str) -> str:
    """Correct answers per source dataset (strict rule), one column per model."""
    keys = list(results)
    lines = ["| Source | " + " | ".join(_name(k) for k in keys) + " |",
             "|---|" + "---|" * len(keys)]
    any_result = results[keys[0]][subset]["strict"]
    for source in any_result["per_source"]:
        cells = []
        for k in keys:
            correct, total = results[k][subset]["strict"]["per_source"][source]
            cells.append(f"{correct}/{total}")
        lines.append(f"| {SOURCE_NAMES[source]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confusion_tables(results: dict, subset: str) -> str:
    """A small confusion matrix per model (strict rule)."""
    blocks = []
    for key, by_subset in results.items():
        c = by_subset[subset]["strict"]["confusion"]
        blocks.append(
            f"**{_name(key)}**\n\n"
            "| True \\ Predicted | phishing | legitimate | unparseable |\n"
            "|---|---|---|---|\n"
            + "\n".join(f"| {t} | {c[t][PHISHING]} | {c[t][LEGITIMATE]} "
                        f"| {c[t]['unparseable']} |" for t in (PHISHING, LEGITIMATE)))
    return "\n\n".join(blocks)


def build_markdown(results: dict, n_all: int, n_dedup: int) -> str:
    return "\n\n".join([
        f"# Classification metrics (prompt {PROMPT_VERSION}, test set)",
        "Generated by `python3 src/metrics.py` — do not edit by hand. "
        "Phishing is the positive class. **Strict**: unparseable answers count "
        "as wrong (main result). **Lenient**: a label recovered from broken "
        "JSON is used. CI = 95% bootstrap confidence interval.",
        f"## All {n_all} test emails — strict",
        main_table(results, "all", "strict"),
        f"## All {n_all} test emails — lenient",
        main_table(results, "all", "lenient"),
        f"## {n_dedup} test emails without a near-duplicate in training — strict",
        f"Excludes {n_all - n_dedup} test emails whose body is ≥"
        f"{int(DUPLICATE_THRESHOLD * 100)}% similar to a training email "
        f"(Jaccard similarity of {SHINGLE_WORDS}-word sequences).",
        main_table(results, "dedup", "strict"),
        "## Correct answers per source — all test emails, strict",
        source_table(results, "all"),
        "## Confusion matrices — all test emails, strict",
        confusion_tables(results, "all"),
    ]) + "\n"


# ===========================================================================
# 5. MAIN
# ===========================================================================

def main() -> None:
    test = load_split("test")
    train = load_split("train_full")
    test_ids = {e["id"] for e in test}

    answers = load_answers(test_ids)
    if not answers:
        return
    # Baseline first, so it is the top row of every table.
    answers = {"naive_bayes": naive_bayes_answers(train, test), **answers}

    duplicates = near_duplicate_ids(train, test)
    subsets = {"all": test,
               "dedup": [e for e in test if e["id"] not in duplicates]}

    # results[model][subset][rule] = metrics dict
    results = {}
    for key, model_answers in answers.items():
        # Only score emails this model actually answered (see the warning
        # in load_answers if any are missing).
        results[key] = {
            subset: {rule: evaluate(model_answers,
                                    [e for e in emails if e["id"] in model_answers],
                                    rule)
                     for rule in ("strict", "lenient")}
            for subset, emails in subsets.items()
        }

    markdown = build_markdown(results, len(test), len(subsets["dedup"]))
    OUT_MD.write_text(markdown, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "near_duplicate_test_ids": sorted(duplicates),
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")

    print(markdown)
    print(f"Saved {OUT_MD.relative_to(REPO_DIR)} and {OUT_JSON.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
