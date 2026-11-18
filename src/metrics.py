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
    - a learning curve: F1 of the fine-tuned model and of Naive Bayes for
      each training-set size (500, 1,500, 3,000, full)
    - a cross-source test: how well models trained WITHOUT one source
      (Nazario, Nigerian or SpamAssassin) handle that source's test emails
      (see section 6)

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
    results/learning_curve.svg — the learning-curve plot

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

from build_dataset import CROSS_SOURCE_HELD_OUT, CROSS_SOURCE_SIZE
from prompts import PROMPT_VERSION, PHISHING, LEGITIMATE, format_email
from shortcut_check import NaiveBayes, load_split

# Paths are relative to this file, so the script works from any directory.
REPO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_DIR / "results" / "raw"
OUT_JSON = REPO_DIR / "results" / "metrics.json"
OUT_MD = REPO_DIR / "results" / "metrics.md"
OUT_SVG = REPO_DIR / "results" / "learning_curve.svg"

# Nested training sets of the learning curve (see section 5), smallest first.
TRAIN_SETS = ["train_500", "train_1500", "train_3000", "train_full"]

# Cross-source test (see section 6): every held-out source is compared with
# the normal training set of the same size.
CROSS_SOURCE_REFERENCE = f"train_{CROSS_SOURCE_SIZE}"

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
    # Sort by name, but put fine-tuned runs in training-set order (500, 1500,
    # 3000, full) — plain text sorting would put "train_1500" before
    # "train_500". Other training sets (e.g. the cross-source one) come
    # after those.
    def order(path):
        train_set = path.name[: -len(suffix)].partition("_ft_")[2]
        position = (TRAIN_SETS.index(train_set) if train_set in TRAIN_SETS
                    else len(TRAIN_SETS))
        return (path.name.split("_ft_")[0], position, path.name)

    for path in sorted(RAW_DIR.glob(f"*{suffix}"), key=order):
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
    """Train the bag-of-words baseline on `train` and return its test
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


def build_markdown(results: dict, n_all: int, n_dedup: int, curve: dict,
                   cross: dict) -> str:
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
        "## Learning curve — all test emails, strict",
        "F1 by number of training emails. Plot: `results/learning_curve.svg`.",
        learning_curve_table(curve),
        "## Cross-source test — one source left out of training",
        f"Each `train_{CROSS_SOURCE_SIZE}_no_<source>` set has "
        f"{CROSS_SOURCE_SIZE:,} balanced emails and none from `<source>`: "
        "that source is replaced by the other source(s) of the same label "
        f"(see `build_dataset.py`). It is compared with `{CROSS_SOURCE_REFERENCE}`, "
        "which contains every source. **Held out** = the test emails from the "
        "left-out source; **in-source** = all other test emails. Strict "
        "scoring. Zero-shot LLMs never saw any training email, so they are "
        "listed as a reference.",
        *[block for c in cross for block in (
            f"### {SOURCE_NAMES[c['held_out_source']]} held out",
            cross_source_table(c))],
    ]) + "\n"


# ===========================================================================
# 5. LEARNING CURVE
# ===========================================================================
#
# How does the score grow with the number of training emails? The training
# sets are nested (500 ⊂ 1,500 ⊂ 3,000 ⊂ full), so each larger set only ADDS
# emails. We compare two learners on the same sets:
#
#     - Llama 3.1 8B fine-tuned on that set (one Colab run per set; a point
#       appears as soon as its results file is in results/raw/)
#     - Naive Bayes trained on that set (computed here, takes seconds)
#
# and draw the zero-shot LLMs as flat reference lines: they never see any
# training email, so their score does not depend on the training size.
# The metric is F1 on all test emails, strict rule.

def learning_curve(results: dict, test: list) -> dict:
    """Test F1 per training set for the fine-tuned model and Naive Bayes,
    plus the zero-shot reference scores."""
    points = []
    for train_set in TRAIN_SETS:
        train = load_split(train_set)
        nb_answers = naive_bayes_answers(train, test)
        nb = evaluate(nb_answers, test, "strict")
        point = {"train_set": train_set, "n_train": len(train),
                 "naive_bayes": {"f1": nb["f1"], "accuracy": nb["accuracy"],
                                 "ci95": nb["ci95"]["f1"]}}
        # The fine-tuned model is only there if its Colab run is finished.
        ft_key = f"llama31_8b_ft_{train_set}"
        if ft_key in results:
            ft = results[ft_key]["all"]["strict"]
            point["finetuned"] = {"f1": ft["f1"], "accuracy": ft["accuracy"],
                                  "ci95": ft["ci95"]["f1"]}
        points.append(point)

    # Zero-shot models: every results key that is neither the baseline nor a
    # fine-tuned model.
    zero_shot = {key: by_subset["all"]["strict"]["f1"]
                 for key, by_subset in results.items()
                 if key != "naive_bayes" and "_ft_" not in key}
    return {"points": points, "zero_shot": zero_shot}


def learning_curve_table(curve: dict) -> str:
    """Markdown version of the plot (also the accessible 'table view')."""
    lines = ["| Training set | Emails | Naive Bayes F1 | Llama 3.1 8B fine-tuned F1 (95% CI) |",
             "|---|---|---|---|"]
    for p in curve["points"]:
        ft = p.get("finetuned")
        ft_cell = (f"{ft['f1']:.3f} ({ft['ci95'][0]:.3f}–{ft['ci95'][1]:.3f})"
                   if ft else "not run yet")
        lines.append(f"| {p['train_set']} | {p['n_train']:,} "
                     f"| {p['naive_bayes']['f1']:.3f} | {ft_cell} |")
    zero = ", ".join(f"{_name(k)} {f1:.3f}" for k, f1 in curve["zero_shot"].items())
    lines.append("")
    lines.append(f"Zero-shot reference (no training emails): {zero}.")
    return "\n".join(lines)


def learning_curve_svg(curve: dict) -> str:
    """Draw the learning curve as a self-contained SVG file.

    Written by hand instead of with matplotlib so the script keeps needing
    only the standard library. GitHub shows .svg files directly, so the
    README can embed it with ![...](results/learning_curve.svg).

    Layout:
      x axis — number of training emails, LOG scale (500 -> 8,248 would
               squash the small sets into the left edge on a linear scale)
      y axis — test F1, from a round value below the lowest score to 1.0
      lines  — fine-tuned Llama (blue, with 95% CI whiskers) and Naive
               Bayes (orange); zero-shot models as dashed gray lines
      labels — every line is named at its right end (no colour-only
               identification), plus a legend at the top
    """
    import math

    width, height = 760, 440
    left, right, top, bottom = 64, 210, 76, 60  # margins; right holds labels
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = curve["points"]
    sizes = [p["n_train"] for p in points]

    # x: log scale, padded a little so the end markers are not on the edge.
    x_min, x_max = math.log10(min(sizes) * 0.8), math.log10(max(sizes) * 1.25)

    def x(n):
        return left + (math.log10(n) - x_min) / (x_max - x_min) * plot_w

    # y: from the lowest score (rounded down to a multiple of 0.05) to 1.0.
    all_f1 = [p["naive_bayes"]["f1"] for p in points]
    all_f1 += [p["finetuned"]["ci95"][0] for p in points if "finetuned" in p]
    all_f1 += list(curve["zero_shot"].values())
    y_min = math.floor(min(all_f1) / 0.05) * 0.05

    def y(f1):
        return top + (1.0 - f1) / (1.0 - y_min) * plot_h

    out = []  # SVG elements, joined at the end

    # Gridlines + y tick labels every 0.05 (hairline, solid, recessive).
    tick = y_min
    while tick <= 1.0 + 1e-9:
        out.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" '
                   f'y1="{y(tick):.1f}" y2="{y(tick):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" '
                   f'text-anchor="end">{tick:.2f}</text>')
        tick += 0.05
    # x tick labels at the actual training-set sizes.
    for p in points:
        out.append(f'<text class="tick" x="{x(p["n_train"]):.1f}" '
                   f'y="{top + plot_h + 20}" text-anchor="middle">{p["n_train"]:,}</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" '
               f'y1="{top + plot_h}" y2="{top + plot_h}"/>')
    # Axis titles.
    out.append(f'<text class="label" x="{left + plot_w / 2}" y="{height - 14}" '
               'text-anchor="middle">Training emails (log scale)</text>')
    out.append(f'<text class="label" transform="translate(16 {top + plot_h / 2}) '
               'rotate(-90)" text-anchor="middle">Test F1 (strict)</text>')

    # Right-end labels: (wanted y, text). Collected first, then spread out so
    # labels that are close in value do not overlap.
    end_labels = []

    # Zero-shot reference lines.
    for key, f1 in curve["zero_shot"].items():
        out.append(f'<line class="ref" x1="{left}" x2="{left + plot_w}" '
                   f'y1="{y(f1):.1f}" y2="{y(f1):.1f}">'
                   f'<title>{_name(key)} zero-shot: F1 {f1:.3f}</title></line>')
        end_labels.append((y(f1), f"{_name(key)} zero-shot {f1:.3f}", "muted"))

    # The two curves. Naive Bayes is drawn first so the fine-tuned model,
    # the main subject, sits on top where they cross.
    for key, css, name in (("naive_bayes", "nb", "Naive Bayes"),
                           ("finetuned", "ft", "Llama 3.1 8B fine-tuned")):
        series = [(p["n_train"], p[key]) for p in points if key in p]
        if not series:
            continue
        if key == "finetuned":
            # 95% CI whiskers: thin vertical lines behind the markers.
            for n, s in series:
                out.append(f'<line class="whisker {css}" x1="{x(n):.1f}" x2="{x(n):.1f}" '
                           f'y1="{y(s["ci95"][0]):.1f}" y2="{y(s["ci95"][1]):.1f}"/>')
        if len(series) > 1:
            path = " ".join(f"{x(n):.1f},{y(s['f1']):.1f}" for n, s in series)
            out.append(f'<polyline class="line {css}" points="{path}"/>')
        for n, s in series:
            # <title> = tooltip when the SVG is opened on its own.
            out.append(f'<circle class="dot {css}" cx="{x(n):.1f}" cy="{y(s["f1"]):.1f}" r="5">'
                       f'<title>{name}, {n:,} training emails: F1 {s["f1"]:.3f} '
                       f'(95% CI {s["ci95"][0]:.3f}–{s["ci95"][1]:.3f})</title></circle>')
        last = series[-1][1]["f1"]
        end_labels.append((y(last), f"{name} {last:.3f}", "primary"))

    # Spread the labels: sort by position, then push each one down until it
    # is at least 15 px below the previous one.
    end_labels.sort()
    placed = []
    for wanted, text, ink in end_labels:
        pos = max(wanted, placed[-1][0] + 15) if placed else wanted
        placed.append((pos, text, ink))
    for pos, text, ink in placed:
        out.append(f'<text class="end {ink}" x="{left + plot_w + 8}" '
                   f'y="{pos + 4:.1f}">{text}</text>')

    # Legend (top row): series swatches.
    legend = [("ft", "Llama 3.1 8B fine-tuned (with 95% CI)"),
              ("nb", "Naive Bayes, same training emails"),
              ("ref", "Zero-shot LLM (no training)")]
    lx = left
    for css, text in legend:
        if css == "ref":
            out.append(f'<line class="ref" x1="{lx}" x2="{lx + 18}" y1="46" y2="46"/>')
        else:
            out.append(f'<circle class="dot {css}" cx="{lx + 9}" cy="46" r="5"/>')
        out.append(f'<text class="label" x="{lx + 24}" y="50">{text}</text>')
        lx += 24 + 6 * len(text) + 20  # rough text width (~6 px/char), then a gap
    out.append(f'<text class="title" x="{left}" y="22">'
               'Learning curve: test F1 vs. number of training emails</text>')

    # Colours as CSS variables, with a dark-mode set for viewers whose system
    # is in dark mode. Series colours: slot 1 (blue) and 2 (orange) of a
    # colour-blind-safe palette; text never uses the series colours.
    style = """
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#8a8984;
        --grid:#e6e5e1; --ft:#2a78d6; --nb:#eb6834; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e87;
          --grid:#33332f; --ft:#3987e5; --nb:#d95926; }
  }
  .bg { fill: var(--surface); }
  text { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
  .title { font-size: 15px; font-weight: 600; fill: var(--ink); }
  .label, .end.primary { font-size: 12px; fill: var(--ink); }
  .tick { font-size: 11px; fill: var(--ink2); }
  .end.muted { font-size: 12px; fill: var(--ink2); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--ink2); stroke-width: 1; }
  .ref { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 5 4; }
  .line { fill: none; stroke-width: 2; }
  .whisker { stroke-width: 1.5; }
  .dot { stroke: var(--surface); stroke-width: 2; }
  .ft { stroke: var(--ft); } circle.ft { fill: var(--ft); }
  .nb { stroke: var(--nb); } circle.nb { fill: var(--nb); }
  circle.dot { stroke: var(--surface); }
"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            'aria-label="Learning curve: test F1 against number of training emails">\n'
            f"<style>{style}</style>\n"
            f'<rect class="bg" width="{width}" height="{height}"/>\n'
            + "\n".join(out) + "\n</svg>\n")


# ===========================================================================
# 6. CROSS-SOURCE TEST
# ===========================================================================
#
# The shortcut check showed that each source has its own easy-to-spot words
# (Linux mailing lists, "beneficiary ... Lagos" scams). Does a model still
# handle a source it NEVER saw during training? For each left-out source we
# split the test set in two:
#
#     held out  — the test emails from that source (all one label, so the
#                 score is simply accuracy: for a phishing source every
#                 error is a missed phishing email, for a legit source a
#                 false alarm)
#     in-source — all other test emails, from sources seen in training
#
# and compare each learner trained WITH the source (train_1500) against the
# same learner trained WITHOUT it (train_1500_no_<source>). A big drop on
# the held-out part means the model had learned the source, not the label.

def cross_source(answers: dict, test: list) -> list:
    """One result per held-out source: held-out and in-source scores for
    the zero-shot LLMs, Naive Bayes and the fine-tuned Llama."""
    # Naive Bayes trained WITH every source, shared by all held-out sources.
    nb_reference = naive_bayes_answers(load_split(CROSS_SOURCE_REFERENCE), test)
    out = []
    for held_out_source in CROSS_SOURCE_HELD_OUT:
        train_set = f"train_{CROSS_SOURCE_SIZE}_no_{held_out_source}"
        held_out = [e for e in test if e["source"] == held_out_source]
        in_source = [e for e in test if e["source"] != held_out_source]

        def row(name: str, saw_held_out, model_answers: dict) -> dict:
            held = evaluate(model_answers, held_out, "strict")
            other = evaluate(model_answers, in_source, "strict")
            return {"model": name, "trained_on_held_out": saw_held_out,
                    "held_out_correct": held["per_source"][held_out_source][0],
                    "held_out_accuracy": held["accuracy"],
                    "held_out_ci95": held["ci95"]["accuracy"],
                    "in_source_accuracy": other["accuracy"],
                    "in_source_f1": other["f1"]}

        rows = []
        # Zero-shot reference rows.
        for key, model_answers in answers.items():
            if key != "naive_bayes" and "_ft_" not in key:
                rows.append(row(_name(key), None, model_answers))
        # Naive Bayes is cheap, so both versions are always computed here.
        rows.append(row(f"Naive Bayes ({CROSS_SOURCE_REFERENCE})", True, nb_reference))
        rows.append(row(f"Naive Bayes ({train_set})", False,
                        naive_bayes_answers(load_split(train_set), test)))
        # Fine-tuned Llama: each version only once its Colab run is finished.
        for ft_set, saw in ((CROSS_SOURCE_REFERENCE, True), (train_set, False)):
            key = f"llama31_8b_ft_{ft_set}"
            if key in answers:
                rows.append(row(_name(key), saw, answers[key]))
            else:
                rows.append({"model": _name(key), "trained_on_held_out": saw,
                             "not_run": True})
        out.append({"held_out_source": held_out_source, "train_set": train_set,
                    "n_held_out": len(held_out), "n_in_source": len(in_source),
                    "rows": rows})
    return out


def cross_source_table(cross: dict) -> str:
    """Markdown table for ONE held-out source."""
    n = cross["n_held_out"]
    lines = [f"| Model | Trained on {SOURCE_NAMES[cross['held_out_source']]}? "
             f"| Held out: correct of {n} (95% CI) "
             f"| In-source accuracy ({cross['n_in_source']}) | In-source F1 |",
             "|---|---|---|---|---|"]
    for r in cross["rows"]:
        saw = {True: "yes", False: "**no**", None: "— (zero-shot)"}[r["trained_on_held_out"]]
        if r.get("not_run"):
            lines.append(f"| {r['model']} | {saw} | not run yet | | |")
            continue
        lo, hi = r["held_out_ci95"]
        lines.append(
            f"| {r['model']} | {saw} "
            f"| {r['held_out_correct']}/{n} = {_pct(r['held_out_accuracy'])} "
            f"({_pct(lo)}–{_pct(hi)}) "
            f"| {_pct(r['in_source_accuracy'])} | {r['in_source_f1']:.3f} |")
    return "\n".join(lines)


# ===========================================================================
# 7. MAIN
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

    curve = learning_curve(results, test)
    cross = cross_source(answers, test)
    markdown = build_markdown(results, len(test), len(subsets["dedup"]), curve,
                              cross)
    OUT_MD.write_text(markdown, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "prompt_version": PROMPT_VERSION,
        "near_duplicate_test_ids": sorted(duplicates),
        "results": results,
        "learning_curve": curve,
        "cross_source": cross,
    }, indent=2) + "\n", encoding="utf-8")

    print(markdown)
    OUT_SVG.write_text(learning_curve_svg(curve), encoding="utf-8")
    print(f"Saved {OUT_MD.relative_to(REPO_DIR)}, {OUT_JSON.relative_to(REPO_DIR)} "
          f"and {OUT_SVG.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
