"""
explain.py — asks the fine-tuned Llama WHY it made its decisions.

Used by notebooks/03_explanations.ipynb on Colab. It needs a CUDA GPU plus
`torch`, `transformers`, `accelerate`, `bitsandbytes` and `peft`, so it is NOT
meant to run on your Mac. It trains nothing: it only loads the adapters that
02_finetune.ipynb saved, and writes to its own results folder.

THE PROBLEM
-----------
We fine-tuned Llama to answer only {"label": "phishing"} or
{"label": "legitimate"}. It has learned to STOP after the label, so simply
asking again gives no explanation. This file gets at the "why" in two ways.

PART A — HIDE-AND-SEE (what does the decision depend on?)
---------------------------------------------------------
We hide one part of the email (sender, subject or body) and measure how the
model's decision changes. If hiding the sender flips the answer, the sender
mattered. This measures the model's real BEHAVIOUR; it does not rely on the
model describing itself, so it is the more trustworthy evidence.

In the first trial, hiding ONE part never changed the fine-tuned model's
answer: the parts that were left were always enough. So we also do the
opposite and show ONLY one part (only_sender, only_subject, only_body), plus
a completely empty email, which shows the model's default answer.

PART B — WRITTEN EXPLANATIONS (what does the model say?)
-------------------------------------------------------
Several ways of getting text, so that we can compare them. The first two were
tried in the first trial and FAILED, so they are no longer run (the code is
kept because their results are part of the story):

  prefill       (failed, 0/20) We write the start of the answer for the model:
                    {"label": "legitimate", "reasoning": "
                and it has to continue from there. The label is the model's
                own, from the official test run.
  followup      (failed, 0/20) We show the model its own answer and ask, as a
                second chat message, to explain it.

  WHY THEY FAILED: in training, every answer was the label and then the end.
  So the model only repeated the label ("legitimate" or {"label": ...}) and
  stopped. base_prefill gets the SAME input but with fine-tuning off, and it
  wrote real text 20/20 times. So our prompt works; the fine-tuning is what
  taught the model not to explain.

These are run:

  starter       A stronger push for the fine-tuned model. We also write the
                first words of the explanation ("I labeled this email as
                legitimate because"), and forbid it to finish during the first
                MIN_NEW_TOKENS tokens. This forces text out, so it may be
                nonsense: read it before trusting it.
  base_starter  CONTROL for `starter`: the same push, fine-tuning off.
  base_prefill  CONTROL without the push. Exactly like `prefill`, but with the
                fine-tuning switched off, so the ORIGINAL Llama argues for the
                label that the fine-tuned one chose.

If a control's text is as good as the fine-tuned model's, then the explanation
shows what any Llama can say in defence of a label, not what fine-tuning
taught it.

HOW TO READ THE WRITTEN EXPLANATIONS
------------------------------------
They are written AFTER the decision is fixed. Such text can be convincing
without being the true cause (a "post-hoc rationalisation"). Always present
them together with Part A, and never as proof of what the model was thinking.
"""

import json
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

# Our own modules in src/ (the notebook adds src/ to the Python path).
# _render_prompt is the SAME function used for the baselines and for training,
# so the model sees exactly the input it was trained and tested on.
from inference import _render_prompt, free_gpu_memory
from prompts import LEGITIMATE, PHISHING, build_messages

# ---------------------------------------------------------------------------
# Text we write on the model's behalf
# ---------------------------------------------------------------------------
# Part A stops right before the label word. The model's next token is then
# the beginning of either "phishing" or "legitimate".
LABEL_PREFIX = '{"label": "'

# Part B (prefill): a finished label plus the opening quote of the reasoning.
REASONING_PREFIX = '{{"label": "{label}", "reasoning": "'

# Part B (followup): the second chat message.
FOLLOWUP_QUESTION = (
    "Explain in two to four sentences why you chose this label, citing "
    "specific evidence from the email. Answer in plain text, not JSON.")

# Part B (starter): the first words of the explanation, written for the model.
# The label is the fine-tuned model's own. The sentence is neutral: it names
# the label but gives no reason, so the reason must come from the model.
STARTER_WORDS = "I labeled this email as {label} because"

# The ways of getting a written explanation, described at the top. `prefill`
# and `followup` still work, but failed in the first trial, so they are not
# run by default any more.
METHODS = ["starter", "base_starter", "base_prefill"]

# Methods where the fine-tuning is switched off (the controls).
CONTROL_METHODS = {"base_prefill", "base_starter"}

# Methods that get the first words and may not end immediately.
STARTER_METHODS = {"starter", "base_starter"}

# For the starter methods: the model may not finish before this many tokens
# (~30 words, about one long sentence).
MIN_NEW_TOKENS = 40

# Long enough for four long sentences (~60-100 tokens in the baselines).
MAX_NEW_TOKENS = 250


# ===========================================================================
# 1. SMALL HELPERS
# ===========================================================================

def load_rows(path: Path) -> list:
    """Read an output file (JSONL), dropping a half-written last line.

    If Colab disconnects in the middle of writing, the last line may be cut
    off. We then rewrite the file without it. Otherwise the next row would be
    appended to the broken line, and both would be lost.
    """
    path = Path(path)
    if not path.exists():
        return []
    good, broken = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            broken = True
    if broken:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in good), encoding="utf-8")
    return good


def adapter_off(model, off: bool):
    """Context manager that switches the fine-tuning off, if `off` is True.

    The LoRA adapter sits ON TOP of the original Llama, so peft can disable it
    temporarily. This gives us the original model without loading it a second
    time:   with adapter_off(model, True): ...
    """
    return model.disable_adapter() if off else nullcontext()


def run_safely(function, items: list) -> list:
    """Call function(items) -> list, splitting the batch if the GPU is full.

    Same idea as inference._generate_safely: after "CUDA out of memory", try
    each half separately, down to a single item.
    """
    try:
        return function(items)
    except torch.cuda.OutOfMemoryError:
        if len(items) == 1:
            raise  # one email alone does not fit: nothing left to try
    # Outside the `except` block on purpose: inside it, Python's traceback
    # would keep the huge GPU tensors alive, so nothing could be freed.
    free_gpu_memory()
    half = len(items) // 2
    print(f"  out of GPU memory with {len(items)} items; "
          f"retrying as {half} + {len(items) - half}")
    return run_safely(function, items[:half]) + run_safely(function, items[half:])


def in_batches(items: list, batch_size: int, length):
    """Yield batches of similar length (less padding = faster)."""
    items = sorted(items, key=length)
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ===========================================================================
# 2. PART A — HIDE-AND-SEE
# ===========================================================================

def make_variants(email: dict) -> dict:
    """All versions of one email that we score: {variant name: email dict}.

    no_...    one part hidden          only_...  everything else hidden
    empty     all parts hidden: the model's default answer. It is nearly the
              same question for every email (only the "Contains links" line
              differs), but costs almost nothing to score.

    That "Contains links: yes/no" line is always shown, in every variant.

    An empty field is shown to the model as "(none)" or "(empty)" by
    prompts.format_email(), so the prompt's layout stays the same.

    no_enron: our training data contains Enron mail only as LEGITIMATE (135
    Enron senders in train_3000, 0 of them phishing). The model may simply
    have learned "Enron = safe". Hiding the sender is not enough to test that,
    because Enron is often named in the body too. So we replace the name
    everywhere with an invented company. Only for emails that mention Enron.
    """
    variants = {
        "full": email,
        "no_sender": {**email, "sender": ""},
        "no_subject": {**email, "subject": ""},
        "no_body": {**email, "body": ""},
        "only_sender": {**email, "subject": "", "body": ""},
        "only_subject": {**email, "sender": "", "body": ""},
        "only_body": {**email, "sender": "", "subject": ""},
        "empty": {**email, "sender": "", "subject": "", "body": ""},
    }
    fields = {k: email.get(k) or "" for k in ("sender", "subject", "body")}
    if any("enron" in text.lower() for text in fields.values()):
        variants["no_enron"] = {
            **email,
            **{k: re.sub("enron", "Norvane", text, flags=re.I)
               for k, text in fields.items()},
        }
    return variants


def label_token_ids(tokenizer) -> tuple:
    """Token ids of the first piece of "phishing" and of "legitimate", as they
    appear right after LABEL_PREFIX.

    A tokenizer may glue characters together differently depending on what
    comes before them, so we do not assume: we tokenize the whole answer and
    check that it really starts with the tokens of LABEL_PREFIX.
    """
    prefix = tokenizer(LABEL_PREFIX, add_special_tokens=False)["input_ids"]
    first = {}
    for label in (PHISHING, LEGITIMATE):
        whole = tokenizer(json.dumps({"label": label}),
                          add_special_tokens=False)["input_ids"]
        if whole[:len(prefix)] != prefix:
            raise ValueError("The tokenizer splits the label differently than "
                             "expected; the scores would be wrong.")
        first[label] = whole[len(prefix)]
    if first[PHISHING] == first[LEGITIMATE]:
        raise ValueError("Both labels start with the same token.")
    return first[PHISHING], first[LEGITIMATE]


@torch.inference_mode()  # no gradients: faster, less memory
def label_scores(model, tokenizer, emails: list) -> list:
    """How strongly does the model lean towards "phishing"? One number each.

    The number is the LOG-ODDS: the model's raw score for "phishing" minus its
    raw score for "legitimate" at the position where the label is written.
        above 0 -> answers phishing      below 0 -> answers legitimate
        far from 0 -> very sure          near 0 -> undecided
    Log-odds are better than a probability here: a very sure model has
    probabilities like 0.9999 and 0.99999, which look the same but are not.
    """
    phish_id, legit_id = label_token_ids(tokenizer)
    texts = [_render_prompt(tokenizer, e, {}) + LABEL_PREFIX for e in emails]
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(model.device)

    # Shorter prompts are padded on the LEFT. Number only the real tokens
    # (0, 1, 2, ...), so padding does not shift an email's positions.
    # generate() does this by itself; a direct call to the model does not.
    positions = (inputs["attention_mask"].cumsum(-1) - 1).clamp(min=0)

    # logits_to_keep=1: we only need the last position. Scores for every
    # position would take several GB for nothing.
    out = model(**inputs, position_ids=positions, logits_to_keep=1,
                use_cache=False)
    last = out.logits[:, -1, :].float()  # full precision before subtracting
    return (last[:, phish_id] - last[:, legit_id]).tolist()


def score_emails(run_name: str, model, tokenizer, emails: list, entries: dict,
                 out_path: Path, batch_size: int = 16) -> list:
    """Score every variant of every email, with fine-tuning on and off.

    entries — {email id: its row from results/explanation_emails.json}
    Rows already in `out_path` are skipped, so a run can be resumed.
    Returns all rows in the file (old + new).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["scorer"], r["variant"]) for r in load_rows(out_path)}

    for scorer in ("finetuned", "base"):
        # One job per (email, variant) that is still missing.
        jobs = [(email, name, changed)
                for email in emails
                for name, changed in make_variants(email).items()
                if (email["id"], scorer, name) not in done]
        print(f"[{run_name}] scoring with {scorer}: {len(jobs)} to run")
        with adapter_off(model, scorer == "base"), \
                open(out_path, "a", encoding="utf-8") as f:
            for batch in in_batches(jobs, batch_size,
                                    length=lambda j: len(j[2]["body"])):
                scores = run_safely(
                    lambda part: label_scores(model, tokenizer,
                                              [j[2] for j in part]), batch)
                for (email, name, _), score in zip(batch, scores):
                    f.write(json.dumps({
                        "id": email["id"],
                        "model": run_name,
                        "group": entries[email["id"]]["group"],
                        "source": email["source"],
                        "true_label": email["label"],
                        "saved_label": entries[email["id"]]["saved_label"],
                        "scorer": scorer,    # "finetuned" or "base"
                        "variant": name,     # which part was hidden
                        "log_odds": score,
                        "label": PHISHING if score > 0 else LEGITIMATE,
                    }, ensure_ascii=False) + "\n")
                f.flush()  # on disk before the next batch
    return load_rows(out_path)


# ===========================================================================
# 3. PART B — WRITTEN EXPLANATIONS
# ===========================================================================

def explanation_prompt(tokenizer, email: dict, label: str, method: str) -> str:
    """The exact text the model has to continue, for one method."""
    if method in ("prefill", "base_prefill"):
        return (_render_prompt(tokenizer, email, {})
                + REASONING_PREFIX.format(label=label))
    if method in STARTER_METHODS:
        return (_render_prompt(tokenizer, email, {})
                + REASONING_PREFIX.format(label=label)
                + STARTER_WORDS.format(label=label))
    if method == "followup":
        messages = build_messages(email) + [
            {"role": "assistant", "content": json.dumps({"label": label})},
            {"role": "user", "content": FOLLOWUP_QUESTION},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    raise ValueError(f"unknown method: {method}")


# Where a prefilled reasoning ends: a quote that is NOT escaped (no backslash
# before it) and is followed by the end of the JSON object, by another JSON
# field, or by nothing. We cannot stop at the first plain quote, because small
# models often forget to escape quotes inside their text.
_END_OF_REASONING = re.compile(r'(?<!\\)"\s*(?:\}|,\s*"|\Z)')


def is_repetitive(text: str) -> bool:
    """True for text that goes round in circles ("legitimate legitimate ...").

    A model that is forced to write can get stuck in a loop. We look at every
    group of three neighbouring words: in normal English nearly all of these
    groups are different, in a loop most of them are the same.
    """
    words = text.lower().split()
    groups = list(zip(words, words[1:], words[2:]))
    if not groups:  # fewer than three words: nothing to compare
        return True
    return len(set(groups)) / len(groups) < 0.7


def clean_explanation(raw: str, method: str, label: str) -> dict:
    """Turn the model's raw continuation into readable text.

    Returns {"reasoning": text, "finished": bool, "usable": bool}.
      finished — the model ended its explanation by itself
                 (for followup we cannot tell, so it is always True)
      usable   — there is real text: not empty, not just the label again and
                 not a loop. This is only a rough filter: it cannot tell
                 whether the text is TRUE, so a person still has to read it.
    """
    text, finished = raw, True
    if method != "followup":
        match = _END_OF_REASONING.search(text)
        finished = match is not None
        if match:
            text = text[:match.start()]
        # Turn JSON escapes back into normal characters: \" -> "  \n -> newline
        text = text.replace('\\"', '"').replace("\\n", "\n")
    text = text.strip()
    # A model that falls back to its training answers {"label": ...} again.
    repeats_label = text.startswith("{") or '"label"' in text
    usable = (len(text) >= 40 and not repeats_label
              and not is_repetitive(text))
    # `usable` was decided on the model's OWN words. Only now do we add the
    # words we wrote for it, so that the sentence reads normally.
    if method in STARTER_METHODS:
        text = STARTER_WORDS.format(label=label) + " " + text
    return {"reasoning": text, "finished": finished, "usable": usable}


class NoEarlyEnd(LogitsProcessor):
    """Forbids the model to finish during its first `min_new_tokens` tokens.

    At every step the model gives each possible next token a score (a logit)
    and takes the highest one. We set the score of the forbidden tokens to
    minus infinity, so they can never be chosen.

    Forbidden are the special tokens (one of them means "I am done") and every
    token containing a quote or a curly bracket, because those would close the
    JSON answer. A side effect: the model cannot quote the email in its first
    words either.
    """

    def __init__(self, tokenizer, prompt_length: int, min_new_tokens: int):
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        forbidden = {i for token, i in tokenizer.get_vocab().items()
                     if any(c in token for c in '"{}')}
        forbidden |= set(tokenizer.get_added_vocab().values())  # special ones
        self.forbidden = torch.tensor(sorted(forbidden))

    def __call__(self, input_ids, scores):
        # input_ids = prompt + what has been written so far. All prompts in a
        # batch have the same padded length, so one number is enough.
        if input_ids.shape[1] - self.prompt_length < self.min_new_tokens:
            scores = scores.clone()  # do not change the model's own tensor
            scores[:, self.forbidden.to(scores.device)] = -float("inf")
        return scores


@torch.inference_mode()
def continue_texts(model, tokenizer, texts: list,
                   min_new_tokens: int = 0) -> list:
    """Let the model continue each text (greedy: same result every time).

    min_new_tokens > 0 forbids it to finish earlier than that.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(model.device)
    processors = LogitsProcessorList()
    if min_new_tokens:
        processors.append(NoEarlyEnd(tokenizer, inputs["input_ids"].shape[1],
                                     min_new_tokens))
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,  # silence warnings
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=processors,
    )
    # Cut off the prompt; keep only what the model added.
    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def explain_emails(run_name: str, model, tokenizer, emails: list,
                   entries: dict, out_path: Path, methods: list = METHODS,
                   batch_size: int = 16) -> list:
    """Write an explanation for every email with every method.

    The label being explained is always `saved_label`: the fine-tuned model's
    own answer in the official test run. Finished rows are skipped, so a run
    can be resumed. Returns all rows in the file (old + new).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["method"]) for r in load_rows(out_path)}

    for method in methods:
        todo = [e for e in emails if (e["id"], method) not in done]
        print(f"[{run_name}] explaining with {method}: {len(todo)} to run")
        min_new = MIN_NEW_TOKENS if method in STARTER_METHODS else 0
        with adapter_off(model, method in CONTROL_METHODS), \
                open(out_path, "a", encoding="utf-8") as f:
            for batch in in_batches(todo, batch_size,
                                    length=lambda e: len(e["body"])):
                prompts = [explanation_prompt(tokenizer, e,
                                              entries[e["id"]]["saved_label"],
                                              method) for e in batch]
                answers = run_safely(
                    lambda part: continue_texts(model, tokenizer, part,
                                                min_new),
                    prompts)
                for email, raw in zip(batch, answers):
                    label = entries[email["id"]]["saved_label"]
                    f.write(json.dumps({
                        "id": email["id"],
                        "model": run_name,
                        "method": method,
                        "group": entries[email["id"]]["group"],
                        "source": email["source"],
                        "true_label": email["label"],
                        "label": label,
                        "raw_output": raw,  # kept, to re-clean it without a GPU
                        **clean_explanation(raw, method, label),
                    }, ensure_ascii=False) + "\n")
                f.flush()
    return load_rows(out_path)
