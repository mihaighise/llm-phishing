"""
explain_select.py — picks the test emails whose decisions we want explained.

WHY A SEPARATE, LOCAL SCRIPT
----------------------------
The fine-tuned Llama only ever answered {"label": ...}, so it has no written
explanations yet. notebooks/03_explanations.ipynb asks it for some, on Colab.
But explaining all 1,000 test emails is not useful: most are easy for every
model. This script picks the interesting ones and writes their ids to

    results/explanation_emails.json

That small file is committed to GitHub, so Colab uses exactly the same emails.
Choosing them needs only the saved answers in results/raw/, so it runs on your
Mac in a second, with no GPU and no extra packages:

    python3 src/explain_select.py

THE THREE GROUPS
----------------
  fixed          — zero-shot Llama is WRONG, and ALL FOUR fine-tuned Llamas are
                   RIGHT. This is where the report's side-by-side examples
                   come from. Requiring all four (not just one) avoids picking
                   an email that one model got right by luck.
  control        — phishing emails that zero-shot and all fine-tuned models
                   get right. Needed because `fixed` is almost entirely
                   legitimate mail: without controls we could not tell a model
                   that explains well from one that calls everything harmless.
  nazario_missed — Nazario phishing missed by the model that was trained
                   WITHOUT Nazario. These show where fine-tuning fails.

NOTE ON FAIRNESS: the groups are chosen BY looking at who was right. They are
good for examples and for studying explanations, but any percentage computed
on them says nothing about accuracy. Accuracy comes from results/metrics.md.
"""

import json
import random
from pathlib import Path

# The prompt version is part of every answer file's name (see prompts.py).
from prompts import PROMPT_VERSION

# Project folder = the parent of src/, whatever folder you run the script from.
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
OUT_PATH = ROOT / "results" / "explanation_emails.json"

ZERO_SHOT = "llama31_8b"
MAIN_FT = "llama31_8b_ft_train_3000"                   # the model we explain
NO_NAZARIO_FT = "llama31_8b_ft_train_1500_no_nazario"  # the failure examples
ALL_FT = [MAIN_FT, "llama31_8b_ft_train_500", "llama31_8b_ft_train_1500",
          NO_NAZARIO_FT]

# Fixed seed: the same "random" emails every time the script runs.
SEED = 42
CONTROLS_PER_SOURCE = 15   # 15 Nazario + 15 Nigerian = 30 controls

# The two examples proposed for the report. They are always part of the trial.
PREFERRED_IDS = [
    "7ab5587438b151f53be0e97ee196f26b",  # Enron "LNG Update" meeting request
    "9745238908723c12fae597547da45503",  # NTELOS stock alert (not Enron)
]

# The 20-email trial: 10 "fixed" emails + 5 controls from each phishing source.
TRIAL_FIXED = 10
TRIAL_CONTROLS_PER_SOURCE = 5


def read_jsonl(path: Path) -> list:
    """Read a JSONL file: one JSON object per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_answers(model: str) -> dict:
    """One model's saved test answers, as {email id: answer row}."""
    path = RAW_DIR / f"{model}_test_all_{PROMPT_VERSION}.jsonl"
    return {row["id"]: row for row in read_jsonl(path)}


def main() -> None:
    emails = {e["id"]: e for e in read_jsonl(TEST_PATH)}
    zero = read_answers(ZERO_SHOT)
    tuned = {model: read_answers(model) for model in ALL_FT}

    def is_right(answers: dict, email_id: str) -> bool:
        """Did this model give the true label for this email?"""
        return answers[email_id]["label"] == emails[email_id]["label"]

    def all_tuned_right(email_id: str) -> bool:
        return all(is_right(answers, email_id) for answers in tuned.values())

    # --- Group 1: zero-shot wrong, every fine-tuned model right.
    fixed = [i for i in emails
             if not is_right(zero, i) and all_tuned_right(i)]

    # --- Group 2: phishing that everyone gets right (random but repeatable).
    rng = random.Random(SEED)
    controls_by_source = {}
    for source in ("nazario", "nigerian"):
        # sorted() first: the pick must not depend on the file's line order.
        pool = sorted(i for i, e in emails.items()
                      if e["source"] == source
                      and is_right(zero, i) and all_tuned_right(i))
        controls_by_source[source] = rng.sample(pool, CONTROLS_PER_SOURCE)
    control = [i for ids in controls_by_source.values() for i in ids]

    # --- Group 3: Nazario phishing missed by the model that never saw Nazario.
    nazario_missed = [i for i, e in emails.items()
                      if e["source"] == "nazario"
                      and not is_right(tuned[NO_NAZARIO_FT], i)]

    # --- The trial: our two preferred examples + random others.
    for email_id in PREFERRED_IDS:
        if email_id not in fixed:
            raise ValueError(f"Preferred example {email_id} is not in 'fixed'")
    others = sorted(set(fixed) - set(PREFERRED_IDS))
    trial = PREFERRED_IDS + rng.sample(others, TRIAL_FIXED - len(PREFERRED_IDS))
    for ids in controls_by_source.values():
        trial += ids[:TRIAL_CONTROLS_PER_SOURCE]

    def entries(ids: list, group: str, model: str) -> list:
        """What Colab needs to know about each email. `saved_label` is the
        answer this fine-tuned model REALLY gave in the official test run: it
        is the decision we ask the model to explain."""
        return [{"id": i, "group": group,
                 "saved_label": tuned[model][i]["label"],
                 "trial": i in trial and group != "nazario_missed"}
                for i in ids]

    # One list per fine-tuned model to run on Colab.
    selection = {
        "prompt_version": PROMPT_VERSION,
        "models": {
            MAIN_FT: (entries(fixed, "fixed", MAIN_FT)
                      + entries(control, "control", MAIN_FT)),
            NO_NAZARIO_FT: entries(nazario_missed, "nazario_missed",
                                   NO_NAZARIO_FT),
        },
    }
    OUT_PATH.write_text(json.dumps(selection, indent=1) + "\n",
                        encoding="utf-8")

    n_trial = sum(e["trial"] for e in selection["models"][MAIN_FT])
    print(f"fixed: {len(fixed)}, control: {len(control)}, "
          f"nazario_missed: {len(nazario_missed)}, trial: {n_trial}")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
