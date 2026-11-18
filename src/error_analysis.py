"""
error_analysis.py — Step 8 of the LLM phishing project: what goes wrong?

WHAT THIS SCRIPT DOES
---------------------
metrics.py says HOW OFTEN each model is wrong; this script helps explain
WHY. It reads the same answer files (results/raw/*_test_all_v2.jsonl) and
writes results/error_analysis.md with:

    1. Overview      — errors per model, and how many errors the zero-shot
                       models share (an email all three get wrong is
                       probably genuinely confusing, not bad luck).
    2. Fine-tuned    — every test email ANY fine-tuned Llama gets wrong,
                       with a hand-written note on what the email is.
    3. Shared false alarms — the legitimate emails ALL three zero-shot
                       models call phishing, grouped into hand-assigned
                       categories.
    4. Sender effects — false-alarm rates for emails from Enron, slack.net
                       and free-mail addresses: the zero-shot models judge
                       the SENDER heavily, sometimes with invented reasons.
    5. Indicators    — which indicators the zero-shot models cite when they
                       raise a false alarm.
    6. Missed phishing — the phishing emails the zero-shot models and Naive
                       Bayes miss, with hand-written notes.

HAND-WRITTEN PARTS
------------------
The numbers are computed. The NOTES and CATEGORIES dictionaries below are
our own judgement after reading each email (and the models' reasoning).
They are keyed by the first 8 characters of the email id, so you can look
an email up in data/processed/test.jsonl. If a model is re-run and makes
NEW errors, they are listed with "(no note yet)" so nothing is hidden.

HOW TO RUN (from the repository root)
-------------------------------------
    python3 src/error_analysis.py

Only the Python standard library is used, so no `pip install` is needed.
"""

import json
import re
from collections import Counter
from pathlib import Path

from metrics import RAW_DIR, MODEL_NAMES, SOURCE_NAMES, _name, naive_bayes_answers
from prompts import PROMPT_VERSION, PHISHING, LEGITIMATE
from shortcut_check import load_split

REPO_DIR = Path(__file__).resolve().parent.parent
OUT_MD = REPO_DIR / "results" / "error_analysis.md"

ZERO_SHOT = ["llama31_8b", "mistral_7b", "qwen35_9b"]
FINE_TUNED = ["llama31_8b_ft_train_500", "llama31_8b_ft_train_1500",
              "llama31_8b_ft_train_3000"]

# ===========================================================================
# HAND-WRITTEN NOTES (see "HAND-WRITTEN PARTS" above)
# ===========================================================================

# Emails a fine-tuned model gets wrong, or a zero-shot model / Naive Bayes
# misses as phishing: what the email actually is.
NOTES = {
    # --- fine-tuned errors ---
    "af3940b3": "Real PayPal phishing (\"Your PayPal account has been limited\"), "
                "hidden letter by letter inside random Arabic filler text to "
                "fool spam filters. Label is correct.",
    "c6fe173f": "Real Enron IT notice: \"login using your email … and GIS_ID as "
                "password\". Reads exactly like credential phishing.",
    "472e010a": "Marketing newsletter (dELiA*s) the recipient subscribed to; "
                "tracking links with the email address in the URL. Spam-like "
                "but labelled legitimate in SpamAssassin.",
    "74d7826c": "MSN Groups notification: a user post about investing, "
                "surrounded by promotional links.",
    "6cc80071": "Law-firm email: \"attached are clean and blackline versions of "
                "the POA documents\" plus a confidentiality disclaimer — "
                "almost no content besides an attachment reference.",
    # --- zero-shot missed phishing ---
    "b664e42c": "Bank \"eStatement attached\" phishing: the body is harmless; "
                "the attack was in the attachment, which the dataset does not "
                "contain.",
    "599c56b7": "\"Payment advice attached\" phishing — attack in the missing "
                "attachment.",
    "8ab84be1": "Bank \"electronic statement attached\" phishing — attack in "
                "the missing attachment.",
    "a27f5c5c": "Fake IT help-desk \"welcome to the new Outlook\" message; "
                "plausible internal-IT wording.",
    "2f389373": "Fake IT \"web mail upgrade\" phishing; Mistral's answer could "
                "not be parsed.",
    "76405628": "Classic 419 scam (NNPC, Lagos); Qwen's answer could not be "
                "parsed (missing final \"}\"), the label inside was correct.",
    # --- Naive Bayes missed phishing ---
    "50b5f750": "Fake \"urgent computer security update\" in IT-department "
                "wording; few typical phishing words.",
    "59a77c38": "Fake voicemail notification.",
    "afc3f66d": "Fake \"statement data update\" with a PDF to open; broken "
                "English.",
    "dee541bb": "Two lines: \"kindly find attached for payment confirmation\".",
    "d6c96306": "Apple Pay \"account locked\" phishing with every word padded "
                "with \"##number##\" tokens to break word-based filters.",
    "6f59af57": "Fake antivirus \"infected attachment removed\" notice about an "
                "unexpected sign-in.",
    "21ab1d30": "Body is only a timestamp; the subject (\"trouble with your "
                "billing information\") carries the attack.",
    "a43ca180": "Chinese-language fake SF Express invoice.",
    "51e35fd5": "\"Data usage alert\" whose body is random characters "
                "(encoding or obfuscation).",
    "3f655cf8": "Portuguese Santander phishing with the junk string "
                "\"wdabmasel\" injected into every word.",
}

# Categories for the legitimate emails ALL three zero-shot models flag.
CATEGORIES = {
    "Sender looks odd: personal, free-mail or mailing-list relay address whose "
    "name does not match (e.g. tomwhore@slack.net, fork_list@hotmail.com)":
        ["03842c8b", "0730af2d", "add20155", "bad6cc45", "73e76851",
         "b4709f01", "cf581818", "d792c70b", "17191a8b", "82d99f33"],
    "Invented evidence: claims that are simply false (\"ENRON.com instead of "
    "the correct enron.com\", \"Gordon Mohr, a famous hacker\", a real domain "
    "called a misspelling)":
        ["3106fd79", "c585ab93", "25c09b6f", "46598c59"],
    "Marketing / newsletter with tracking links (legitimate, but spam-like)":
        ["472e010a", "7dc3dc8b", "97aaac37", "4290abb9"],
    "Content genuinely resembles phishing: login details, \"click the attached "
    "icon\", a legal threat":
        ["c6fe173f", "4b5dbbf0", "c3728fcb", "d1a25fba"],
    "Informal writing: deliberate misspellings, a one-line note with a bare "
    "link":
        ["a9deb9cc", "432dc597"],
}

# Sender groups whose false-alarm rate we compare (regex on the From line).
SENDER_GROUPS = {
    "Enron (…@ENRON.com)": r"@enron\.com",
    "slack.net (1990s ISP, not the Slack app)": r"@slack\.net",
    "Free-mail (hotmail, aol, yahoo, gmail, …)":
        r"@(hotmail|aol|yahoo|gmail|usa\.net|netcom|msn)\.",
}


# ===========================================================================
# HELPERS
# ===========================================================================

def load_answers(key: str) -> dict:
    path = RAW_DIR / f"{key}_test_all_{PROMPT_VERSION}.jsonl"
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f) if r}


def is_wrong(answer: dict, email: dict) -> bool:
    """Strict rule (as in metrics.py): an unparseable answer is wrong."""
    return not (answer.get("parse_ok") and answer.get("label") == email["label"])


def is_false_alarm(answer: dict, email: dict) -> bool:
    """A legitimate email the model clearly CALLED phishing. Unparseable
    answers are counted separately, as in metrics.py's confusion matrices."""
    return (email["label"] == LEGITIMATE and bool(answer.get("parse_ok"))
            and answer.get("label") == PHISHING)


def note(email_id: str) -> str:
    return NOTES.get(email_id[:8], "(no note yet)")


def snippet(email: dict, n: int = 90) -> str:
    """Subject, cut to n characters, safe inside a Markdown table."""
    text = " ".join(email["subject"].split())
    text = text if len(text) <= n else text[:n - 1] + "…"
    return text.replace("|", "\\|")


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    test = load_split("test")
    by_id = {e["id"]: e for e in test}
    answers = {k: load_answers(k) for k in ZERO_SHOT + FINE_TUNED}
    answers["naive_bayes"] = naive_bayes_answers(load_split("train_full"), test)
    wrong = {k: {i for i, a in ans.items() if is_wrong(a, by_id[i])}
             for k, ans in answers.items()}
    false_alarm = {k: {i for i, a in ans.items() if is_false_alarm(a, by_id[i])}
                   for k, ans in answers.items()}
    legit = [e for e in test if e["label"] == LEGITIMATE]
    out = [f"# Error analysis (prompt {PROMPT_VERSION}, test set)",
           "Generated by `python3 src/error_analysis.py` — do not edit by hand. "
           "Strict scoring (unparseable = wrong). Notes and categories are "
           "hand-written after reading each email; they live in the script."]

    # --- 1. Overview -------------------------------------------------------
    rows = []
    for k in ["naive_bayes"] + ZERO_SHOT + FINE_TUNED:
        unparseable = sum(not answers[k][i].get("parse_ok") for i in wrong[k])
        missed = sum(by_id[i]["label"] == PHISHING and answers[k][i].get("parse_ok")
                     for i in wrong[k])
        rows.append([_name(k), len(wrong[k]), len(false_alarm[k]), missed, unparseable])
    zs_fa = [false_alarm[k] for k in ZERO_SHOT]
    shared_all = set.intersection(*zs_fa)
    shared_two = {i for i in set.union(*zs_fa) if sum(i in s for s in zs_fa) >= 2}
    out += ["## 1. Overview",
            table(["Model", "Errors", "False alarms (legit called phishing)",
                   "Missed phishing (called legitimate)", "Unparseable answers"], rows),
            f"Zero-shot false alarms: {len(set.union(*zs_fa))} different legitimate "
            f"emails are flagged by at least one model, {len(shared_two)} by at "
            f"least two, **{len(shared_all)} by all three** (section 3). Most "
            "false alarms are therefore model-specific, not a fixed set of "
            "'impossible' emails."]

    # --- 2. Fine-tuned errors ----------------------------------------------
    ft_wrong = sorted(set.union(*(wrong[k] for k in FINE_TUNED)))
    rows = []
    for i in ft_wrong:
        e = by_id[i]
        rows.append([f"`{i[:8]}`", SOURCE_NAMES[e["source"]],
                     ", ".join(k.rsplit("_", 1)[1] for k in FINE_TUNED if i in wrong[k]),
                     sum(i in wrong[k] for k in ZERO_SHOT),
                     "yes" if i in wrong["naive_bayes"] else "no",
                     note(i)])
    out += ["## 2. Every error of the fine-tuned models",
            f"{len(ft_wrong)} different emails across the three fine-tuned runs.",
            table(["Email", "Source", "Wrong in fine-tuned run(s)",
                   "Zero-shot models also wrong (of 3)", "Naive Bayes wrong",
                   "What it is"], rows)]

    # --- 3. Shared zero-shot false alarms ----------------------------------
    categorised = {i for ids in CATEGORIES.values() for i in ids}
    rows = []
    for category, ids in CATEGORIES.items():
        present = [i for i in ids if any(x.startswith(i) for x in shared_all)]
        ft_fixed = sum(not any(x.startswith(i) and x in wrong["llama31_8b_ft_train_3000"]
                               for x in shared_all) for i in present)
        rows.append([category, len(present), f"{ft_fixed}/{len(present)}",
                     ", ".join(f"`{i}`" for i in present)])
    missing = sorted(i[:8] for i in shared_all if i[:8] not in categorised)
    if missing:
        rows.append(["(not categorised yet)", len(missing), "", ", ".join(missing)])
    out += [f"## 3. The {len(shared_all)} legitimate emails all three zero-shot "
            "models flag",
            table(["Category", "Emails", "Correct after fine-tuning (train_3000)",
                   "Email ids"], rows)]

    # --- 4. Sender effects -------------------------------------------------
    rows = []
    for group, pattern in SENDER_GROUPS.items():
        ids = [e["id"] for e in legit if re.search(pattern, e["sender"], re.I)]
        cells = [f"{sum(i in false_alarm[k] for i in ids)}/{len(ids)}" for k in ZERO_SHOT]
        rows.append([group, *cells])
    others = [e["id"] for e in legit
              if not any(re.search(p, e["sender"], re.I) for p in SENDER_GROUPS.values())]
    rows.append(["All other legitimate emails",
                 *[f"{sum(i in false_alarm[k] for i in others)}/{len(others)}" for k in ZERO_SHOT]])
    # How often the reasoning uses Enron's history (bankruptcy / scandal).
    enron_ids = [e["id"] for e in legit if re.search(r"@enron\.com", e["sender"], re.I)]
    history = {k: sum(bool(re.search(r"bankrupt|scandal|defunct|infamous|no longer exist",
                                     answers[k][i].get("reasoning") or "", re.I))
                      for i in enron_ids) for k in ZERO_SHOT}
    out += ["## 4. False alarms by sender (legitimate emails)",
            table(["Sender", *[_name(k) for k in ZERO_SHOT]], rows),
            "Emails in the dataset are from 1992–2008, but the models read "
            "them as if they arrived today. Reasoning that mentions Enron's "
            "bankruptcy or scandal: "
            + ", ".join(f"{_name(k)} {history[k]}/{len(enron_ids)}" for k in ZERO_SHOT)
            + ". Example (Llama): *\"pretends to be from Enron, a company that "
              "went bankrupt in 2001\"*."]

    # --- 5. Indicators in false alarms -------------------------------------
    rows = []
    for k in ZERO_SHOT:
        fa = false_alarm[k]
        counts = Counter(ind for i in fa for ind in (answers[k][i].get("indicators") or [])
                         if ind != "none")
        rows.append([_name(k), len(fa),
                     ", ".join(f"{ind} {100 * n / len(fa):.0f}%"
                               for ind, n in counts.most_common(4))])
    out += ["## 5. Indicators cited in false alarms",
            "Share of each model's false alarms that cite the indicator (an "
            "answer can cite several).",
            table(["Model", "False alarms", "Most cited indicators"], rows)]

    # --- 6. Missed phishing ------------------------------------------------
    rows = []
    for k in ZERO_SHOT + ["naive_bayes"]:
        for i in sorted(wrong[k]):
            e = by_id[i]
            if e["label"] != PHISHING:
                continue
            parsed = answers[k][i].get("parse_ok")
            rows.append([_name(k), f"`{i[:8]}`", SOURCE_NAMES[e["source"]],
                         "no" if not parsed else "yes", snippet(e, 60), note(i)])
    out += ["## 6. Missed phishing (zero-shot models and Naive Bayes)",
            table(["Model", "Email", "Source", "Answer parsed", "Subject",
                   "What it is"], rows)]

    OUT_MD.write_text("\n\n".join(out) + "\n", encoding="utf-8")
    print("\n\n".join(out))
    print(f"\nSaved {OUT_MD.relative_to(REPO_DIR)}")


if __name__ == "__main__":
    main()
