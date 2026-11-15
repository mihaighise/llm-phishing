"""
prompts.py — Step 2 of the LLM phishing project.

WHAT THIS FILE IS FOR
---------------------
Every model in the comparison (Llama baseline, Llama fine-tuned, Qwen,
Mistral) must see EXACTLY the same instructions and EXACTLY the same email
text, and must be graded by EXACTLY the same parser. Keeping all of that in
one file guarantees the comparison is fair: if we change the prompt, it
changes for every model at once.

WHAT IS INSIDE
--------------
1. INDICATORS        — the fixed list of phishing signals a model may report.
2. SYSTEM_PROMPT     — the instructions (role, task, required JSON format).
3. format_email()    — turns one email record into the text the model reads.
4. build_messages()  — wraps that text into the chat format models expect.
5. parse_response()  — turns a model's raw text answer back into a clean dict,
                       tolerating the common ways small models break format.
6. format_answer()   — the reverse of parse_response(): turns a dict into the
                       exact JSON text we want. Used later (Step 4/5) to build
                       the fine-tuning targets in the same format.

This file uses only the Python standard library, so it can be imported both on
your Mac and on Colab.
"""

import json  # to read and write the JSON answers
import re    # regular expressions, used to find JSON inside messy output

# ===========================================================================
# 1. THE FIXED INDICATOR LIST
# ===========================================================================
# Models must choose indicators from this list instead of inventing their own
# wording. That makes indicators COUNTABLE: we can later say "Mistral flagged
# impersonation in 62% of phishing emails" and compare models fairly.
#
# Each key is the machine-readable name the model must output; the value is
# the plain-English description shown to the model inside the prompt.
INDICATORS = {
    "urgency": "pressure to act quickly, threats, deadlines, account suspension",
    "impersonation": "pretends to be a known brand, bank, service, or person; "
                     "sender name and address do not match",
    "credential_request": "asks for passwords, login, account, card, or personal details",
    "suspicious_link": "links that are hidden, mismatched, shortened, or lead to "
                       "an unexpected domain; urges you to click",
    "unexpected_attachment": "asks you to open an attachment or download a file "
                             "you did not expect",
    "money_or_reward": "promises money, prizes, refunds, inheritance, or asks for "
                       "payment or a money transfer",
    "poor_grammar": "spelling mistakes, awkward phrasing, or inconsistent formatting",
    "generic_greeting": "impersonal greeting such as 'Dear Customer' or 'Dear Member'",
    "none": "no suspicious characteristics (use only for legitimate emails)",
}

# Version of the prompt + parser. It is included in every results file name,
# so answers produced with an older prompt are never mixed with (or mistaken
# for) answers from a newer one. INCREASE IT whenever SYSTEM_PROMPT,
# format_email() or the indicator list changes.
#   v1 — first version (20-email smoke test, 2026-11-15)
#   v2 — explains the [YEAR] placeholder; Qwen read it as a sign of a
#        template/scam email in the v1 smoke test
PROMPT_VERSION = "v2"

# The two allowed labels. Keeping them in constants avoids typos elsewhere.
PHISHING = "phishing"
LEGITIMATE = "legitimate"


# ===========================================================================
# 2. THE SYSTEM PROMPT
# ===========================================================================
# The bullet list of indicators is generated from INDICATORS above, so the
# prompt can never fall out of sync with the parser.
_indicator_lines = "\n".join(f"- {name}: {desc}" for name, desc in INDICATORS.items())

SYSTEM_PROMPT = f"""You are an email security analyst. Your job is to decide whether an email is a phishing attempt or a legitimate email, and to explain why.

Phishing includes any email that tries to deceive the reader into revealing sensitive information, clicking a malicious link, opening a harmful attachment, or sending money (including advance-fee scams).

Some emails had their link addresses removed during data collection. The line "Contains links: yes" tells you the original email contained links even if you cannot see them.

All years in the emails have been replaced with the placeholder [YEAR] for anonymization. This was done to every email, so [YEAR] is not a sign of phishing.

Choose indicators ONLY from this list:
{_indicator_lines}

Respond with ONLY a JSON object, no other text, in exactly this format:
{{"label": "phishing" or "legitimate", "confidence": a number between 0 and 1, "indicators": [list of indicator names from the list above], "reasoning": "two to four sentences explaining your decision, citing specific evidence from the email"}}"""


# ===========================================================================
# 3. FORMATTING ONE EMAIL
# ===========================================================================

def format_email(email: dict) -> str:
    """Turn one email record (a row from data/processed/*.jsonl) into the text
    block the model reads.

    We show the sender, the subject, whether it had links, and the body —
    the same four things a careful human would look at. We do NOT show the
    `source` or `label` fields, of course: that would give away the answer.
    """
    # Empty fields are shown as "(none)" rather than left blank, so the model
    # knows the field exists but was empty.
    sender = email.get("sender") or "(none)"
    subject = email.get("subject") or "(none)"
    links = "yes" if email.get("has_urls") else "no"
    body = email.get("body") or "(empty)"
    return (
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Contains links: {links}\n"
        f"Body:\n{body}"
    )


# ===========================================================================
# 4. BUILDING THE CHAT MESSAGES
# ===========================================================================

def build_messages(email: dict, use_system_role: bool = True) -> list:
    """Return the conversation to send to a chat model, as a list of
    {"role": ..., "content": ...} dicts.

    Every model family (Llama, Qwen, Mistral) turns this list into its own
    special token format via `tokenizer.apply_chat_template(...)` — we never
    hand-write those special tokens ourselves, because getting them wrong
    silently hurts accuracy.

    use_system_role:
        Most chat templates accept a separate "system" message. A few older
        templates (some Mistral versions) reject it with an error like
        "Conversation roles must alternate user/assistant". For those, pass
        False and the instructions are placed at the top of the user message
        instead — same words, different slot.
    """
    user_text = f"Analyze this email:\n\n{format_email(email)}"
    if use_system_role:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}]


# ===========================================================================
# 5. PARSING THE MODEL'S ANSWER
# ===========================================================================

# Qwen-style models may emit hidden reasoning wrapped in <think>...</think>.
# We disable thinking mode, but strip it anyway in case a model does it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

# Matches markdown code fences like ```json ... ``` that models often add.
_FENCE_RE = re.compile(r"```(?:json)?", re.I)

# Last-resort pattern: find "label": "something" even if the JSON is broken.
_LABEL_RE = re.compile(r'"?label"?\s*[:=]\s*"?([A-Za-z \-]+)"?', re.I)

# Words models use instead of exactly "phishing" / "legitimate".
# Each is mapped to one of our two official labels.
_LABEL_SYNONYMS = {
    "phishing": PHISHING, "phish": PHISHING, "malicious": PHISHING,
    "scam": PHISHING, "fraud": PHISHING, "suspicious": PHISHING,
    "spam": PHISHING,
    "legitimate": LEGITIMATE, "legit": LEGITIMATE, "safe": LEGITIMATE,
    "benign": LEGITIMATE, "ham": LEGITIMATE, "not phishing": LEGITIMATE,
    "not_phishing": LEGITIMATE, "non-phishing": LEGITIMATE,
}


def _normalize_label(value) -> str | None:
    """Map whatever the model wrote as a label onto PHISHING / LEGITIMATE.
    Returns None if it is not recognisable."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    # Check longer phrases like "not phishing" before the bare word "phishing",
    # otherwise "not phishing" would wrongly match "phishing".
    for synonym in sorted(_LABEL_SYNONYMS, key=len, reverse=True):
        if key == synonym or key.startswith(synonym):
            return _LABEL_SYNONYMS[synonym]
    return None


def _first_json_object(text: str) -> str | None:
    """Return the first complete {...} block in `text`, or None.

    We cannot simply use a regex for this, because JSON can contain nested
    braces and braces inside strings. Instead we walk the text character by
    character, counting { and } while ignoring anything inside quotes.
    """
    start = text.find("{")
    while start != -1:
        depth = 0           # how many { are currently open
        in_string = False   # are we inside a "..." string?
        escaped = False     # was the previous character a backslash?
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                # Inside a string, only an unescaped quote ends it.
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]  # found the matching brace
        # Unbalanced from this start; try the next "{" further along.
        start = text.find("{", start + 1)
    return None


def parse_response(text: str) -> dict:
    """Turn a model's raw output into a clean, uniform dict.

    Always returns a dict with these keys, even when parsing fails:
        label        — "phishing", "legitimate", or None if unreadable
        confidence   — float in [0, 1], or None
        indicators   — list of valid indicator names (unknown ones dropped)
        reasoning    — the explanation text ("" if missing)
        parse_ok     — True only if a valid JSON object with a valid label
                       was found
        error        — short description of what went wrong, or None

    Why so forgiving? Small open models often add "Here is my analysis:"
    before the JSON, wrap it in ```json fences, or write "Phishing" with a
    capital P. We still want to grade those answers. But anything we cannot
    read is reported (parse_ok=False) and counted as WRONG in the metrics —
    a model that cannot follow the format should be penalised.
    """
    result = {"label": None, "confidence": None, "indicators": [],
              "reasoning": "", "parse_ok": False, "error": None}
    if not text or not text.strip():
        result["error"] = "empty output"
        return result

    # Remove thinking blocks and code fences before searching for JSON.
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned)

    blob = _first_json_object(cleaned)
    data = None
    if blob is not None:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            result["error"] = f"invalid JSON: {exc.msg}"
    else:
        result["error"] = "no JSON object found"

    if isinstance(data, dict):
        # Lower-case all keys so "Label" or "LABEL" also work.
        data = {str(k).lower(): v for k, v in data.items()}
        result["label"] = _normalize_label(data.get("label"))

        # Confidence: accept numbers or numeric strings, clamp to [0, 1].
        # Some models answer on a 0-100 scale; we rescale those.
        try:
            conf = float(data.get("confidence"))
            if conf > 1:
                conf = conf / 100
            result["confidence"] = min(max(conf, 0.0), 1.0)
        except (TypeError, ValueError):
            pass  # leave as None

        # Indicators: accept a list (or a single string), normalise spacing
        # and case ("Credential Request" -> "credential_request"), and keep
        # only names from our fixed list.
        raw = data.get("indicators") or []
        if isinstance(raw, str):
            raw = [raw]
        cleaned_inds = []
        for item in raw:
            name = re.sub(r"[\s\-]+", "_", str(item).strip().lower())
            if name in INDICATORS and name not in cleaned_inds:
                cleaned_inds.append(name)
        result["indicators"] = cleaned_inds

        result["reasoning"] = str(data.get("reasoning") or "").strip()

        if result["label"] is not None:
            result["parse_ok"] = True
            result["error"] = None
        else:
            result["error"] = f"unrecognised label: {data.get('label')!r}"
        return result

    # JSON failed entirely. Last resort: look for `label: phishing` anywhere
    # in the text. We record the label (useful for analysis) but keep
    # parse_ok=False, so the main metrics still treat it as a format failure.
    match = _LABEL_RE.search(cleaned)
    if match:
        result["label"] = _normalize_label(match.group(1))
    return result


# ===========================================================================
# 6. FORMATTING A TARGET ANSWER (for fine-tuning later)
# ===========================================================================

def format_answer(label: str, indicators: list, reasoning: str,
                  confidence: float = 1.0) -> str:
    """Build the exact JSON string we want a model to output.

    Used in Steps 4-5 to create training targets for the fine-tuned Llama, so
    it learns to answer in precisely the format parse_response() expects.
    Keys are written in the same order as in SYSTEM_PROMPT.
    """
    return json.dumps({
        "label": label,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "reasoning": reasoning,
    }, ensure_ascii=False)
