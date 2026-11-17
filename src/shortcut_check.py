"""
shortcut_check.py — Step 1 "shortcut check" of the LLM phishing project.

THE QUESTION THIS SCRIPT ANSWERS
--------------------------------
Our LLMs catch almost every phishing email in the test set (99-100% recall).
Is that because they understand phishing, or because the dataset is EASY in
a way that has nothing to do with phishing? The suspected "shortcut" is the
era gap: phishing emails are from 2015-2022, legitimate ones from 1992-2008,
so vocabulary (product names, mailing-list footers, slang) may give the
answer away without any understanding.

THE METHOD
----------
Train the simplest possible text classifier — a bag-of-words Naive Bayes,
the classic 1990s spam filter — and test it on the SAME 1,000 test emails
the LLMs saw. A bag-of-words model only counts words; it knows nothing about
meaning, grammar or intent.

    If this "dumb" model also scores ~99%, the dataset can be solved by word
    counting alone, and the LLM scores must be read with that in mind (the
    write-up reports it as a limitation).

To see WHERE the easy signal lives, we train separate models on:
    full     — exactly the text the LLM sees (From, Subject, links flag, Body)
    body     — the email body only
    subject  — the subject line only
    sender   — the "From:" line only (name + address)
and with two training sizes (500 and all ~8,000 emails), because a shortcut
that is learnable from 500 emails is a very strong one.

Finally we print the words the model relies on most for each class, so we
can see with our own eyes whether they are phishing cues ("verify",
"account") or era/source cues (mailing-list names, old software).

HOW TO RUN (from the repository root, after build_dataset.py)
-------------------------------------------------------------
    python3 src/shortcut_check.py

Only the Python standard library is used, so no `pip install` is needed.
Results are printed and also saved to results/shortcut_check.json.
"""

# ---------------------------------------------------------------------------
# Imports — all from the Python standard library, plus our own prompts.py.
# ---------------------------------------------------------------------------
import json       # reads the JSONL splits and writes the results file
import math       # math.log for Naive Bayes probabilities
import re         # regular expression that splits text into words
from collections import Counter, defaultdict  # word counting
from pathlib import Path  # OS-independent file paths

# format_email() builds the exact text block the LLMs read. Reusing it means
# the "full" feature set is guaranteed to match what the LLMs saw.
from prompts import format_email, PHISHING, LEGITIMATE

# Paths are relative to this file, so the script works from any directory.
REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data" / "processed"
OUT_PATH = REPO_DIR / "results" / "shortcut_check.json"

# The LLM results from the baseline run (README "Baseline results"), shown
# next to Naive Bayes for comparison. Strict scoring: unparseable = wrong.
LLM_RESULTS = {
    "Llama 3.1 8B": {"accuracy": 0.877, "f1": 0.891},
    "Qwen 3.5 9B":  {"accuracy": 0.922, "f1": 0.947},
    "Mistral 7B":   {"accuracy": 0.893, "f1": 0.903},
}


# ===========================================================================
# 1. LOADING DATA AND TURNING EMAILS INTO WORDS
# ===========================================================================

def load_split(name: str) -> list:
    """Read one split file (e.g. "test") into a list of email dicts."""
    with open(DATA_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# A "word" is a run of letters, digits or apostrophes. Punctuation and
# whitespace are separators. Lower-casing means "Account" == "account".
_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list:
    """Split text into lower-case words: "Verify your Account!" ->
    ["verify", "your", "account"]."""
    return _WORD_RE.findall(text.lower())


# Each feature set is a function that picks which part of an email the
# classifier is allowed to read. Comparing them shows where the signal is.
FEATURE_SETS = {
    "full":    format_email,                       # what the LLM sees
    "body":    lambda e: e.get("body") or "",
    "subject": lambda e: e.get("subject") or "",
    "sender":  lambda e: e.get("sender") or "",
}


# ===========================================================================
# 2. NAIVE BAYES — the whole classifier
# ===========================================================================
#
# Idea: for each class (phishing / legitimate), learn how often every word
# appears. To classify a new email, ask "which class makes these words most
# likely?" Formally, pick the class c with the highest
#
#     log P(c) + sum over words w in the email of log P(w | c)
#
# P(c)     = share of training emails in class c (the "prior").
# P(w | c) = how often w appears in class-c emails, out of all words in
#            class c. We add 1 to every count ("Laplace smoothing") so a word
#            never seen in one class does not get probability 0, which would
#            make log() blow up and let one word veto the whole email.
#
# "Naive" because it pretends every word is independent of the others —
# clearly false for language, but it works surprisingly well for spam.

class NaiveBayes:
    def fit(self, texts: list, labels: list) -> "NaiveBayes":
        """Learn word counts per class from training texts."""
        self.classes = sorted(set(labels))
        # word_counts[c][w] = how many times word w appears in class c.
        self.word_counts = {c: Counter() for c in self.classes}
        class_sizes = Counter(labels)
        for text, label in zip(texts, labels):
            self.word_counts[label].update(tokenize(text))

        # The vocabulary is every word seen in training, in any class.
        self.vocab = set()
        for counts in self.word_counts.values():
            self.vocab.update(counts)
        v = len(self.vocab)

        # Precompute log probabilities so predicting is just additions.
        self.log_prior = {c: math.log(class_sizes[c] / len(labels))
                          for c in self.classes}
        self.log_likelihood = {}
        # Probability for a word never seen in class c (only the +1 from
        # smoothing). Needed because a word may be in one class but not the
        # other.
        self.log_unseen = {}
        for c in self.classes:
            total = sum(self.word_counts[c].values())
            denom = total + v  # +v: one smoothing count for each vocab word
            self.log_likelihood[c] = {
                w: math.log((n + 1) / denom)
                for w, n in self.word_counts[c].items()
            }
            self.log_unseen[c] = math.log(1 / denom)
        return self

    def predict(self, text: str) -> str:
        """Return the most likely class for one text."""
        # Words never seen in training carry no information; skip them.
        words = [w for w in tokenize(text) if w in self.vocab]
        best_class, best_score = None, -math.inf
        for c in self.classes:
            ll, unseen = self.log_likelihood[c], self.log_unseen[c]
            score = self.log_prior[c] + sum(ll.get(w, unseen) for w in words)
            if score > best_score:
                best_class, best_score = c, score
        return best_class

    def top_words(self, cls: str, n: int = 25, min_count: int = 10) -> list:
        """The n words that most strongly point to `cls`.

        "Strength" is the log-odds ratio log P(w|cls) - log P(w|other):
        how many times more likely the word is in this class than the other.
        Rare words (fewer than min_count uses in total) are skipped — a word
        seen 3 times can have a huge ratio by pure chance.
        """
        other = next(c for c in self.classes if c != cls)
        scored = []
        for w in self.vocab:
            if self.word_counts[cls][w] + self.word_counts[other][w] < min_count:
                continue
            ratio = (self.log_likelihood[cls].get(w, self.log_unseen[cls])
                     - self.log_likelihood[other].get(w, self.log_unseen[other]))
            scored.append((ratio, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:n]]


# ===========================================================================
# 3. SCORING — same definitions as the LLM baseline numbers
# ===========================================================================

def evaluate(predictions: list, emails: list) -> dict:
    """Accuracy, precision, recall, F1 ("phishing" = positive class) and
    accuracy per source dataset."""
    truth = [e["label"] for e in emails]
    tp = sum(p == PHISHING and t == PHISHING for p, t in zip(predictions, truth))
    fp = sum(p == PHISHING and t == LEGITIMATE for p, t in zip(predictions, truth))
    fn = sum(p != PHISHING and t == PHISHING for p, t in zip(predictions, truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    # Per-source accuracy, e.g. {"spamassassin": "250/254"}.
    per_source = defaultdict(lambda: [0, 0])
    for p, e in zip(predictions, emails):
        per_source[e["source"]][0] += p == e["label"]
        per_source[e["source"]][1] += 1

    return {
        "accuracy": round(sum(p == t for p, t in zip(predictions, truth))
                          / len(truth), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_alarms": fp,      # legitimate emails flagged as phishing
        "missed_phishing": fn,   # phishing emails labelled legitimate
        "per_source": {s: f"{a}/{b}" for s, (a, b) in sorted(per_source.items())},
    }


# ===========================================================================
# 4. MAIN — run every (training size x feature set) combination
# ===========================================================================

def main() -> None:
    test = load_split("test")
    # Smallest and largest of the nested training sets. Both are disjoint
    # from the test set by construction (build_dataset.py carves the test
    # set out first).
    train_sets = {"train_500": load_split("train_500"),
                  "train_full": load_split("train_full")}

    results = {"runs": [], "top_words": {}, "llm_baselines": LLM_RESULTS}

    print(f"{'train set':11} {'features':8}  acc    P      R      F1    "
          f"false alarms  missed")
    for train_name, train in train_sets.items():
        for feat_name, extract in FEATURE_SETS.items():
            model = NaiveBayes().fit([extract(e) for e in train],
                                     [e["label"] for e in train])
            preds = [model.predict(extract(e)) for e in test]
            scores = evaluate(preds, test)
            results["runs"].append({"train_set": train_name,
                                    "train_size": len(train),
                                    "features": feat_name, **scores})
            print(f"{train_name:11} {feat_name:8}  {scores['accuracy']:.3f}  "
                  f"{scores['precision']:.3f}  {scores['recall']:.3f}  "
                  f"{scores['f1']:.3f}  {scores['false_alarms']:12}  "
                  f"{scores['missed_phishing']:6}")

            # The "words it relies on" list is most informative for the
            # biggest model reading the same text as the LLMs.
            if train_name == "train_full" and feat_name == "full":
                for cls in (PHISHING, LEGITIMATE):
                    results["top_words"][cls] = model.top_words(cls)
                full_per_source = scores["per_source"]

    print("\nNaive Bayes (train_full, full text) — correct per source:")
    for source, frac in full_per_source.items():
        print(f"  {source:13} {frac}")

    print("\nLLM baselines for comparison (zero-shot, strict):")
    for name, s in LLM_RESULTS.items():
        print(f"  {name:13} acc={s['accuracy']:.3f}  F1={s['f1']:.3f}")

    for cls, words in results["top_words"].items():
        print(f"\nWords that most strongly signal '{cls}':\n  "
              + ", ".join(words))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {OUT_PATH.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
