"""
build_dataset.py — Step 1 of the LLM phishing project.

WHAT THIS SCRIPT DOES
---------------------
Turns the raw Zenodo CSV files (record 8339691) into clean, de-duplicated,
balanced train / validation / test splits that every model in the project
will use. Running it twice with the same seed produces exactly the same
files, so the experiment is reproducible.

PIPELINE (in the order the code runs)
-------------------------------------
1. LOAD     Read the five source CSVs we care about.
             - Phishing:   Nazario.csv (modern phishing, 2015-2022)
                           Nigerian_Fraud.csv ("419" advance-fee scams)
             - Legitimate: the label-0 rows of Nazario_5.csv,
                           Nigerian_5.csv and SpamAssasin.csv
2. CLEAN    Strip leftover HTML/CSS, decode HTML entities, collapse
             whitespace, optionally hide years, and cut very long bodies.
3. DEDUP    Remove exact duplicates (the legit files share many emails),
             and drop any email that appears with BOTH labels (ambiguous).
4. SPLIT    Carve out a fixed test set (1,000) and validation set (400)
             first, then build NESTED training sets from what is left:
             train_500 ⊂ train_1500 ⊂ train_3000 ⊂ train_full.
             "Nested" means each smaller set is the first N emails of the
             bigger one, so the learning-curve experiment only changes ONE
             thing (how much data), never which emails are included.
5. SAVE     Write one JSONL file per split (one JSON object per line) plus
             a stats.json summary, into data/processed/.

HOW TO RUN (from the repository root)
-------------------------------------
    python3 src/build_dataset.py
    python3 src/build_dataset.py --seed 7 --no-mask-years   # variations

Only the Python standard library is used, so no `pip install` is needed.
"""

# ---------------------------------------------------------------------------
# Imports — all from the Python standard library.
# ---------------------------------------------------------------------------
import argparse   # parses command-line options like --seed
import csv        # reads the CSV files
import hashlib    # builds a fingerprint (hash) of each email for de-duplication
import html       # html.unescape turns "&amp;" into "&", "&#39;" into "'", etc.
import json       # writes the output JSONL / JSON files
import random     # shuffles emails reproducibly (with a fixed seed)
import re         # regular expressions, used for text cleaning
import sys        # needed to raise the CSV field size limit
import unicodedata  # normalizes odd Unicode characters (e.g. full-width letters)
from collections import Counter, defaultdict  # handy counting containers
from pathlib import Path  # nicer, OS-independent file paths

# Some email bodies are huge (hundreds of KB). Python's csv module refuses
# fields larger than 128 KB by default, so we raise the limit to the maximum.
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# CONFIGURATION — every number that shapes the dataset lives here, so you can
# change the experiment without hunting through the code.
# ===========================================================================

# Where each source file's emails go, and which label rows we keep from it.
#   file name          -> (short source tag, label value to keep, our label)
# The Zenodo files use "1" = bad (phishing/spam) and "0" = legitimate.
SOURCES = {
    "Nazario.csv":        ("nazario",    "1", "phishing"),
    "Nigerian_Fraud.csv": ("nigerian",   "1", "phishing"),
    # The *_5 files mix phishing with legit mail; we only take their legit
    # half (label 0), because their phishing half duplicates the two files
    # above.
    "Nazario_5.csv":      ("legit_naz5", "0", "legitimate"),
    "Nigerian_5.csv":     ("legit_nig5", "0", "legitimate"),
    # SpamAssassin's label 1 is ordinary spam (not phishing), so we only use
    # its legitimate "ham" emails (label 0).
    "SpamAssasin.csv":    ("spamassassin", "0", "legitimate"),
}

# Size of the held-out evaluation sets, PER CLASS (phishing / legitimate).
# Test = 500 + 500 = 1,000 emails; validation = 200 + 200 = 400 emails.
TEST_PER_CLASS = 500
VAL_PER_CLASS = 200

# Nested training-set sizes (TOTAL emails, half phishing / half legit).
# "full" is added automatically = everything left after test/val, balanced.
TRAIN_SIZES = [500, 1500, 3000]

# Share of phishing emails that come from Nazario (modern credential
# phishing) versus Nigerian Fraud (older advance-fee scams). We favour
# Nazario because it is closer to the phishing people receive today.
# Used for test, validation and the fixed-size training sets.
NAZARIO_SHARE = 0.6

# Bodies longer than this (in characters) are cut. ~4,000 characters is
# roughly 1,000 tokens — long enough to keep the important part of almost
# every email, short enough that 1,000 test emails x 4 models stays fast.
MAX_BODY_CHARS = 4000

# Emails whose cleaned body is shorter than this are dropped — they are
# usually empty shells (attachment-only mails) with nothing to classify.
MIN_BODY_CHARS = 20


# ===========================================================================
# STEP 2 — CLEANING HELPERS
# ===========================================================================

# Regex that matches a CSS rule block such as:
#   @page Section1:first { size:8.5in 11.0in; margin-top:0.75in; }
#   p.MsoNormal { margin:0; }
# i.e. "something that looks like a selector" followed by { property: ... }.
# [^{}\n]{0,200} = up to 200 non-brace characters on one line (the selector)
# \{\s*[\w-]+\s*: = an opening brace followed by "property:" — this is what
#                   makes it CSS. Without it we would also delete real
#                   { ... } blocks of program code from legit developer
#                   mailing-list emails (C, Perl, Tcl patches).
# [^{}]*\}        = the rest of the block up to the closing brace.
CSS_BLOCK_RE = re.compile(r"[^{}\n]{0,200}\{\s*[\w-]+\s*:[^{}]*\}")

# Matches <style>...</style> and <script>...</script> sections, including
# everything inside them. re.S lets "." also match newlines; re.I = any case.
STYLE_SCRIPT_RE = re.compile(r"<(style|script)\b.*?</\1\s*>", re.S | re.I)

# Matches any remaining HTML tag like <div class="x"> or </td> or <br/>.
TAG_RE = re.compile(r"<[^>]+>")

# Matches runs of spaces/tabs (not newlines), so we can squeeze them to one.
SPACES_RE = re.compile(r"[ \t\r\f\v ]+")

# Matches 3 or more newlines in a row (blank-line spam), squeezed to 2.
NEWLINES_RE = re.compile(r"\n{3,}")

# Matches a 4-digit year between 1980 and 2029, as a whole "word"
# (\b = word boundary, so "12005" or "2005abc" are NOT matched).
YEAR_RE = re.compile(r"\b(?:19[89]\d|20[0-2]\d)\b")

# Matches a URL in plain text. Used only to set the has_urls flag.
URL_RE = re.compile(r"https?://|www\.", re.I)


def clean_text(text: str, mask_years: bool) -> str:
    """Return a cleaned, human-readable version of an email body or subject.

    The goal is NOT to remove phishing signals — capital letters, "!!!",
    odd sender names and urgent wording are all kept on purpose, because
    those are exactly what the LLMs should notice. We only remove technical
    noise (HTML/CSS) that wastes tokens and confuses models.
    """
    if not text:
        return ""

    # NFKC normalization converts look-alike Unicode characters into their
    # plain forms (e.g. full-width "Ａ" -> "A", non-breaking space -> space).
    text = unicodedata.normalize("NFKC", text)

    # Remove whole <style>/<script> sections first (their content is code,
    # not text), then any remaining tags.
    text = STYLE_SCRIPT_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)

    # Turn HTML entities into real characters: "&amp;" -> "&", "&nbsp;" -> " ".
    text = html.unescape(text)

    # Remove leftover CSS rules (the "@page Section1:first {...}" junk we saw
    # in some Nazario emails). We loop because a pass can expose new matches.
    for _ in range(3):
        new_text = CSS_BLOCK_RE.sub(" ", text)
        if new_text == text:
            break
        text = new_text

    # Optionally replace years with a placeholder. WHY: our phishing emails
    # are mostly from 2015-2022 and our legit emails from 1992-2008, so a
    # model could "cheat" by noticing the year instead of real phishing cues.
    # Masking years in BOTH classes removes that easy shortcut.
    if mask_years:
        text = YEAR_RE.sub("[YEAR]", text)

    # Tidy whitespace: many spaces/tabs -> one space; strip spaces around
    # line breaks; many blank lines -> one blank line.
    text = SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def fingerprint(subject: str, body: str) -> str:
    """Return a short ID that is identical for duplicate emails.

    We lowercase and remove ALL whitespace before hashing, so two copies of
    the same email that differ only in spacing or capitalization are still
    recognised as duplicates. MD5 is fine here — we need speed and
    consistency, not cryptographic security.
    """
    normalized = re.sub(r"\s+", "", (subject + "||" + body).lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


# ===========================================================================
# STEP 1 + 2 + 3 — LOAD, CLEAN AND DE-DUPLICATE
# ===========================================================================

def load_emails(raw_dir: Path, mask_years: bool):
    """Read every source CSV and return a list of clean, unique emails.

    Each email is a dict with these keys:
        id        — the fingerprint (stable, unique)
        source    — which file it came from (e.g. "nazario")
        label     — "phishing" or "legitimate"
        sender    — the From: field (display name + address)
        subject   — cleaned subject line
        body      — cleaned body (possibly truncated)
        has_urls  — True if the original email contained links
        truncated — True if the body was cut at MAX_BODY_CHARS

    We deliberately do NOT keep the date or receiver fields:
      - date: would leak the era (see the year-masking comment above).
      - receiver: Nazario's phishing was all collected in one mailbox
        (jose@monkey.org), so the receiver alone would reveal the label.
    """
    by_id = {}               # fingerprint -> email dict (first copy we saw)
    labels_seen = defaultdict(set)  # fingerprint -> set of labels it had
    stats = Counter()        # counts for the report at the end

    for file_name, (source, keep_label, our_label) in SOURCES.items():
        path = raw_dir / file_name
        if not path.exists():
            # Fail early with a helpful message instead of a cryptic error.
            sys.exit(f"Missing {path}. Download the Zenodo record 8339691 "
                     f"CSVs into {raw_dir}/ first.")

        # errors="replace" swaps undecodable bytes for "�" instead of
        # crashing — old emails often contain broken encodings.
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                # Keep only the class we want from this file. Some rows have
                # a missing/odd label; those are skipped too.
                if (row.get("label") or "").strip() != keep_label:
                    continue
                stats[f"read:{source}"] += 1

                subject = clean_text(row.get("subject") or "", mask_years)
                body = clean_text(row.get("body") or "", mask_years)

                # Drop empty shells.
                if len(body) < MIN_BODY_CHARS:
                    stats[f"too_short:{source}"] += 1
                    continue

                # The sender is short text; we only normalize whitespace
                # and Unicode — we do NOT mask years or strip it otherwise,
                # because spoofed senders are a key phishing signal.
                sender = SPACES_RE.sub(
                    " ", unicodedata.normalize("NFKC", row.get("sender") or "")
                ).strip()

                # The Zenodo "urls" column is "1" when the original email
                # had links (often the URL text itself was lost when HTML
                # was converted to text). Some legit rows contain "[]"
                # instead of 0/1, so we treat anything other than "1" as 0
                # and also check the body for URL text.
                has_urls = (row.get("urls") or "").strip() == "1" \
                    or bool(URL_RE.search(body))

                # Fingerprint BEFORE truncation, so two copies that differ
                # only far down in a long body are still kept apart.
                fid = fingerprint(subject, body)
                labels_seen[fid].add(our_label)

                truncated = len(body) > MAX_BODY_CHARS
                if truncated:
                    body = body[:MAX_BODY_CHARS].rstrip() + " [...]"

                if fid in by_id:
                    # Exact duplicate (very common: Nazario_5 and Nigerian_5
                    # share most of their legit emails). Keep the first copy.
                    stats[f"duplicate:{source}"] += 1
                    continue

                by_id[fid] = {
                    "id": fid,
                    "source": source,
                    "label": our_label,
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "has_urls": has_urls,
                    "truncated": truncated,
                }

    # An email that appears as BOTH phishing and legitimate is ambiguous
    # ground truth — we cannot fairly grade a model on it, so remove it.
    conflicts = [fid for fid, labs in labels_seen.items() if len(labs) > 1]
    for fid in conflicts:
        by_id.pop(fid, None)
    stats["label_conflicts_removed"] = len(conflicts)

    return list(by_id.values()), stats


# ===========================================================================
# STEP 4 — SPLITTING
# ===========================================================================

def take(pool: list, n: int, what: str) -> list:
    """Remove and return the first n emails from `pool`.

    The pool was shuffled beforehand, so "first n" is a random sample.
    Because we REMOVE them (del pool[:n]), an email can never land in two
    splits — this is what guarantees no test email leaks into training.
    """
    if n > len(pool):
        sys.exit(f"Not enough {what} emails: need {n}, only {len(pool)} left.")
    chosen = pool[:n]
    del pool[:n]
    return chosen


def split_dataset(emails: list, seed: int) -> dict:
    """Split emails into test, val and nested training sets.

    Returns a dict like {"test": [...], "val": [...], "train_500": [...],
    ..., "train_full": [...]}.
    """
    # A dedicated random generator with a fixed seed => same shuffle every
    # run, on every computer. (Using the global `random` could be affected
    # by other code.)
    rng = random.Random(seed)

    # Group emails into three pools: the two phishing sources, and legit.
    # Sorting by id first makes the result independent of file read order.
    pools = {"nazario": [], "nigerian": [], "legitimate": []}
    for e in sorted(emails, key=lambda e: e["id"]):
        key = e["source"] if e["label"] == "phishing" else "legitimate"
        pools[key].append(e)
    for pool in pools.values():
        rng.shuffle(pool)

    def phishing_mix(n_phish: int, what: str) -> list:
        """Take n_phish phishing emails, NAZARIO_SHARE of them from Nazario."""
        n_naz = round(n_phish * NAZARIO_SHARE)
        return (take(pools["nazario"], n_naz, f"{what} Nazario")
                + take(pools["nigerian"], n_phish - n_naz, f"{what} Nigerian"))

    splits = {}

    # --- Held-out sets first, so they are fixed no matter how the training
    # --- sizes are changed later.
    for name, per_class in (("test", TEST_PER_CLASS), ("val", VAL_PER_CLASS)):
        splits[name] = (phishing_mix(per_class, name)
                        + take(pools["legitimate"], per_class, f"{name} legit"))

    # --- Nested training sets.
    # Everything still in the pools is training material. We keep each
    # pool's shuffled order and always take PREFIXES (the first k items),
    # so train_500 is literally contained in train_1500, and so on.
    naz, nig, legit = pools["nazario"], pools["nigerian"], pools["legitimate"]
    for size in TRAIN_SIZES:
        n_phish = size // 2
        n_naz = round(n_phish * NAZARIO_SHARE)
        n_nig = n_phish - n_naz
        if n_naz > len(naz) or n_nig > len(nig) or n_phish > len(legit):
            sys.exit(f"Not enough training emails for train_{size}.")
        splits[f"train_{size}"] = naz[:n_naz] + nig[:n_nig] + legit[:n_phish]

    # train_full = ALL remaining phishing + the same number of legit emails
    # (balanced). Its Nazario share is lower than 60%, because there are far
    # more Nigerian scams than Nazario emails left. It still contains every
    # smaller training set, since those are prefixes of the same lists.
    all_phish = naz + nig
    n_legit = min(len(all_phish), len(legit))
    splits["train_full"] = all_phish[:n_legit] + legit[:n_legit]

    # Shuffle the order INSIDE each split, so phishing and legit emails are
    # interleaved (training on "all phishing, then all legit" is bad
    # practice). This does not change which emails are in each split.
    for name in splits:
        rng.shuffle(splits[name])
    return splits


# ===========================================================================
# STEP 5 — SAVING AND REPORTING
# ===========================================================================

def check_no_leakage(splits: dict) -> None:
    """Stop with an error if any email appears in more than one of
    test / val / train_full. (Smaller train sets are subsets of train_full,
    so checking train_full covers them.)"""
    ids = {name: {e["id"] for e in splits[name]}
           for name in ("test", "val", "train_full")}
    assert not ids["test"] & ids["val"], "test/val overlap!"
    assert not ids["test"] & ids["train_full"], "test/train overlap!"
    assert not ids["val"] & ids["train_full"], "val/train overlap!"
    # And confirm the nesting really holds.
    sizes = TRAIN_SIZES + ["full"]
    for small, big in zip(sizes, sizes[1:]):
        a = {e["id"] for e in splits[f"train_{small}"]}
        b = {e["id"] for e in splits[f"train_{big}"]}
        assert a <= b, f"train_{small} is not inside train_{big}!"


def save(splits: dict, load_stats: Counter, out_dir: Path, args) -> None:
    """Write each split to <out_dir>/<name>.jsonl and a stats.json summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        # Record exactly how the data was built, for the methods section.
        "settings": {
            "seed": args.seed,
            "mask_years": args.mask_years,
            "max_body_chars": MAX_BODY_CHARS,
            "nazario_share": NAZARIO_SHARE,
        },
        "loading": dict(sorted(load_stats.items())),
        "splits": {},
    }

    for name, rows in splits.items():
        # JSONL = one JSON object per line. Easy to stream, easy to load
        # with pandas (pd.read_json(path, lines=True)) or Hugging Face
        # datasets (load_dataset("json", data_files=path)).
        # ensure_ascii=False keeps non-English characters readable.
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary["splits"][name] = {
            "total": len(rows),
            "by_label": dict(Counter(r["label"] for r in rows)),
            "by_source": dict(Counter(r["source"] for r in rows)),
            "truncated": sum(r["truncated"] for r in rows),
        }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print a compact table so you can sanity-check the result at a glance.
    print(f"\nWrote {len(splits)} splits to {out_dir}/")
    print(f"{'split':<12}{'total':>7}{'phish':>7}{'legit':>7}"
          f"{'nazario':>9}{'nigerian':>10}")
    for name, s in summary["splits"].items():
        print(f"{name:<12}{s['total']:>7}"
              f"{s['by_label'].get('phishing', 0):>7}"
              f"{s['by_label'].get('legitimate', 0):>7}"
              f"{s['by_source'].get('nazario', 0):>9}"
              f"{s['by_source'].get('nigerian', 0):>10}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    # Paths default to locations relative to the repository root
    # (this file lives in <repo>/src/, so parents[1] is <repo>).
    repo = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=repo / "data" / "zenodo",
                        help="folder containing the Zenodo CSV files")
    parser.add_argument("--out-dir", type=Path, default=repo / "data" / "processed",
                        help="folder to write the JSONL splits into")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed; same seed = identical splits")
    # BooleanOptionalAction creates both --mask-years and --no-mask-years.
    parser.add_argument("--mask-years", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="replace years 1980-2029 with [YEAR] (default: on)")
    args = parser.parse_args()

    emails, load_stats = load_emails(args.raw_dir, args.mask_years)

    # Report how many unique, usable emails each pool ended up with.
    pool_sizes = Counter(e["source"] if e["label"] == "phishing" else "legitimate"
                         for e in emails)
    print("Unique usable emails:", dict(pool_sizes))

    splits = split_dataset(emails, args.seed)
    check_no_leakage(splits)
    save(splits, load_stats, args.out_dir, args)


# This block runs only when the file is executed directly
# (python3 src/build_dataset.py), not when it is imported by a notebook.
if __name__ == "__main__":
    main()
