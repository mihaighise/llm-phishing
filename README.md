# llm-phishing
LLM-Based Phishing Detection

Compares four LLM configurations on (1) classifying emails as phishing or
legitimate and (2) explaining the reasoning behind each decision:

| Model | Setup |
|---|---|
| Llama 3.1 8B Instruct | Baseline (prompting only) |
| Llama 3.1 8B Instruct | Fine-tuned with QLoRA on our training set |
| Qwen 3.5 9B | Baseline |
| Mistral 7B Instruct | Baseline |

Classification is scored with accuracy, precision, recall and F1. Explanations
are scored manually for correctness, clarity and identification of phishing
indicators (urgency, impersonation, credential requests, suspicious links, …).

## Dataset

Source: [Phishing Email Curated Datasets](https://zenodo.org/records/8339691)
(Champa, Rabbi & Zibran, 2024 — CC-BY-4.0).

- **Phishing:** `Nazario.csv` (modern phishing, 2015–2022) and
  `Nigerian_Fraud.csv` (advance-fee scams).
- **Legitimate:** label-0 rows of `Nazario_5.csv`, `Nigerian_5.csv` and
  `SpamAssasin.csv`.

### Build the splits

The raw data is not committed (it is ~470 MB). Download it, then build:

```bash
mkdir -p data/zenodo
for f in Nazario Nigerian_Fraud Nazario_5 Nigerian_5 SpamAssasin; do
  curl -L -o data/zenodo/$f.csv \
    "https://zenodo.org/api/records/8339691/files/$f.csv/content"
done

python3 src/build_dataset.py            # standard library only, ~20 s
```

This cleans the emails (HTML/CSS removal, whitespace, years masked as
`[YEAR]`), removes duplicates, drops emails with anonymization artifacts
(SpamAssassin's `taint.org` addresses, which appear only in legitimate mail and
were misread as spoofing by the base models), and writes reproducible splits
(seed 42) to `data/processed/`:

| Split | Total | Phishing | Legit | Nazario | Nigerian |
|---|---|---|---|---|---|
| test | 1,000 | 500 | 500 | 300 | 200 |
| val | 400 | 200 | 200 | 120 | 80 |
| train_500 | 500 | 250 | 250 | 150 | 100 |
| train_1500 | 1,500 | 750 | 750 | 450 | 300 |
| train_3000 | 3,000 | 1,500 | 1,500 | 900 | 600 |
| train_full | 8,248 | 4,124 | 4,124 | 1,118 | 3,006 |

Training sets are nested (500 ⊂ 1,500 ⊂ 3,000 ⊂ full) for a learning-curve
experiment on how much fine-tuning data helps. Test and validation emails never
appear in any training set.

Each line of a `.jsonl` file is one email:
`{id, source, label, sender, subject, body, has_urls, truncated}`.

### Known limitations

- **Era gap:** phishing emails are mostly 2015–2022, legitimate ones 1992–2008.
  Dates and receivers are dropped and years are masked to reduce this
  shortcut, but writing style and topics can still reveal the era.
- **Short dates slip through:** only four-digit years are masked, so dates
  like `7/19/01` inside the body remain. In the baseline run, 5–11
  explanations per model (~1%) mention a date or the email's age.
- **Lost URLs:** most Nazario emails had their link text removed during the
  original HTML-to-text conversion; only a `has_urls` flag remains.
- **Limited modern phishing:** ~1,500 unique Nazario emails.
- **Topic/source shortcut:** a bag-of-words classifier reaches 98.6% on the
  test set, because each source has its own vocabulary (Linux mailing-list
  jargon = legitimate, 419-scam wording = phishing). See
  [Shortcut check](#shortcut-check-bag-of-words-baseline).
- **Near-duplicates across splits:** de-duplication is exact-match only, so
  109 test emails are ≥80% identical to a training email (mostly templated
  419 scams). Results should also be reported on the 891 remaining emails.

## Prompt and output format

All models share one prompt and one parser (`src/prompts.py`), so the
comparison is fair. Each email is shown as:

```
From: "USAA" <customer1-mailing@mail.mil>
Subject: COMPLETE YOUR ACCOUNT UPGRADE IMMEDIATELY
Contains links: yes
Body:
Dear USAA Member, We have received multiple account update...
```

and the model must answer with JSON only:

```json
{"label": "phishing", "confidence": 0.92,
 "indicators": ["impersonation", "urgency", "credential_request"],
 "reasoning": "Sender claims to be USAA but uses a mail.mil address..."}
```

Indicators must come from a fixed list so they can be counted across models:
`urgency`, `impersonation`, `credential_request`, `suspicious_link`,
`unexpected_attachment`, `money_or_reward`, `poor_grammar`,
`generic_greeting`, `none`.

The prompt also tells the model that years were replaced with `[YEAR]` on
purpose (in v1, Qwen treated the placeholder as a scam sign).

Settings: zero-shot, greedy decoding (temperature 0), each model's own chat
template, 4-bit NF4 loading for every model, Qwen's thinking mode off.
Answers that cannot be parsed are counted as wrong.

The prompt is versioned (`PROMPT_VERSION` in `src/prompts.py`, currently
**v2**) and the version is part of every results file name. Bump it whenever
the prompt, email formatting or indicator list changes. Only change the prompt
based on **validation** results, never test results.

## Running the baselines (Colab)

1. Accept the license on
   [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   (gated model).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Open
   [`notebooks/01_baselines.ipynb` in Colab](https://colab.research.google.com/github/mihaighise/llm-phishing/blob/main/notebooks/01_baselines.ipynb),
   set *Runtime → Change runtime type → GPU*, and add the token under
   🔑 *Secrets* as `HF_TOKEN`.
4. *Runtime → Run all.*

The notebook rebuilds the dataset from Zenodo (verified byte-identical to the
local build), runs all 1,000 test emails through each model, and saves
answers to `MyDrive/llm-phishing/results/raw/`. For a quick 20-email smoke
test instead (e.g. after changing the prompt), set `SPLIT = "val"` and
`N_EMAILS = 20`.

Tips:
- To get the latest notebook, re-open it from the Colab link above — Colab
  copies the notebook when you open it, and `git pull` in cell 1 only updates
  `src/`.
- If the runtime restarts (crash, disconnect, *Restart session*), run the
  cells again from the top, or at least from the configuration cell; the
  cells that import from `src/` re-add it to the Python path themselves.
- GPU choice: an L4 or A100 (Colab Pro) is recommended for the full test run
  (`BATCH_SIZE = 16`). On those GPUs the setup cell also installs
  `flash-linear-attention`, which makes Qwen 3.5 several times faster; on a
  T4 it is skipped and Qwen takes ~30 s per email (~9 h for 1,000 emails).
- On a small GPU (free T4, 15 GB), batches that don't fit are split
  automatically. If you still get `CUDA out of memory`, use
  *Runtime → Restart session*, set `BATCH_SIZE = 4`, and run all cells again —
  finished emails are not re-run.

### What is saved, and what you need to re-run

| What | Where | Kept after the session? |
|---|---|---|
| Model answers (raw text + parsed fields) | `MyDrive/llm-phishing/results/raw/<model>_<split>_<N>_<prompt version>.jsonl` | Yes (Google Drive) |
| Fine-tuned LoRA adapters (Step 5, ~100–200 MB each) | `MyDrive/llm-phishing/finetune/<run>/adapter/` | Yes (Google Drive) |
| Training checkpoints (every 50 steps; only the best and latest are kept, for resuming) | `MyDrive/llm-phishing/finetune/<run>/checkpoints/` | Yes (Google Drive) — can be deleted once the adapter exists |
| Downloaded model weights (~5–6 GB each) | Colab's temporary disk | No — re-downloaded in 1–3 min when needed |
| Processed dataset | Colab's temporary disk | No — rebuilt from Zenodo in ~30 s |

- Each model only has to be run **once per split**. Re-running with the same
  settings skips emails already in the results file, and an interrupted run
  continues where it stopped.
- Metrics and explanation scoring (Steps 6–7) only read the saved answer
  files, so they need no GPU and can run locally.

## Fine-tuning Llama 3.1 8B (Colab)

[`notebooks/02_finetune.ipynb` in Colab](https://colab.research.google.com/github/mihaighise/llm-phishing/blob/main/notebooks/02_finetune.ipynb)
— same setup as the baselines (GPU runtime, `HF_TOKEN`), then *Run all*.
Training logic is in `src/finetune.py`.

- **Label-only QLoRA:** the model gets exactly the baseline prompt and email
  and learns to answer `{"label": "phishing"}` / `{"label": "legitimate"}`.
  Only the answer tokens are graded (the prompt is masked out of the loss).
  Fine-tuning on explanations is a separate, later run, which also tests
  whether explanations change classification.
- **Settings:** 4-bit Llama (same precision as the baselines), LoRA rank 16
  on all attention and feed-forward layers, learning rate 2e-4 (cosine),
  16 emails per update, 2 epochs. Every 50 steps (~10 min on an L4) and at
  the end, the validation loss is measured and a checkpoint saved; the
  checkpoint with the lowest **validation** loss is kept. The test set is
  used only for the final evaluation. (The first `train_3000` run, before
  commit "Save fine-tuning checkpoints every 50 steps", checked and saved
  only at the end of each epoch.)
  Training emails over 2,048 tokens are skipped (23 of 8,248, 0.3%).
- **One training set per run:** set `TRAIN_SET` (`train_500`, `train_1500`,
  `train_3000`, `train_full`) and *Run all*. Each run has its own Drive
  folder `MyDrive/llm-phishing/finetune/llama31_8b_ft_<train set>/`.
- **Rough L4 times** (2 epochs): `train_500` ~10–20 min, `train_1500`
  ~30–50 min, `train_3000` ~1–1.5 h, `train_full` ~3–5 h, plus ~5 min of
  evaluation, plus ~1 min per 50-step validation check. Because of the
  checkpoints, after a disconnect *Run all* continues where it stopped
  (losing at most ~10 min); a finished adapter is never retrained.
- **After a crash** (e.g. `CUDA out of memory`), use *Runtime → Restart
  session* before *Run all*: the crashed run keeps its GPU memory until the
  session restarts. The notebook checks this and stops with that advice.
- **Results:** `results/raw/llama31_8b_ft_<train set>_{val,test}_all_v2.jsonl`
  in Drive. Download the test file to `results/raw/` and run
  `python3 src/metrics.py`; it appears as "Llama 3.1 8B fine-tuned
  (<train set>)" in every table.

## Baseline results (prompt v2, 1,000 test emails)

Zero-shot, run on an L4 on 2026-11-17. Unparseable answers count as wrong;
"phishing" is the positive class. Full tables — 95% confidence intervals,
lenient scoring, confusion matrices, the near-duplicate-free subset and the
Naive Bayes baseline — are in [`results/metrics.md`](results/metrics.md),
generated by `python3 src/metrics.py`.

| Model | Parsed | Accuracy | Precision | Recall | F1 | False alarms | Missed phishing |
|---|---|---|---|---|---|---|---|
| Llama 3.1 8B | 999 | 87.7% | 0.805 | 0.998 | 0.891 | 121 | 1 |
| Qwen 3.5 9B | 976 | **92.2%** | **0.902** | 0.996 | **0.947** | 54 | 2 |
| Mistral 7B | 998 | 89.3% | 0.831 | 0.990 | 0.903 | 101 | 5 |

Correct answers per source (strict):

| Source | Llama | Qwen | Mistral |
|---|---|---|---|
| Nazario phishing (300) | 299 | 299 | 295 |
| Nigerian fraud (200) | 200 | 199 | 200 |
| SpamAssassin legitimate (254) | 205 | 220 | 198 |
| Nazario_5 legitimate (132) | 94 | 108 | 109 |
| Nigerian_5 legitimate (114) | 79 | 96 | 91 |

Observations:
- **All models over-flag phishing.** Recall is 99–100%; almost every error
  is a legitimate email (most often SpamAssassin mailing-list mail) labelled
  phishing. Reducing these false alarms is what fine-tuning has to improve.
- **Qwen's 24 parse failures** are complete answers missing only the final
  `}` (not cut off by the length limit). The label is readable in all 24 and
  correct in all 24; counting them gives 94.6% accuracy. The strict number
  above is the main result because the rule was fixed before the run — we do
  not change the parser after seeing test results.
- **The labels are easy on this dataset:** a bag-of-words classifier beats
  every LLM (see below), so classification accuracy alone says little about
  understanding phishing.
- Raw answer files are not in git (`results/raw/` is ignored); they live in
  Google Drive.

## Fine-tuned results (label-only QLoRA, `train_3000`)

Llama 3.1 8B fine-tuned on 3,000 emails (2 epochs, best epoch chosen on the
validation set), run on an L4 on 2026-11-17, scored on the same 1,000 test
emails. Strict scoring; full tables in [`results/metrics.md`](results/metrics.md).

| Model | Accuracy (95% CI) | F1 | False alarms | Missed phishing |
|---|---|---|---|---|
| **Llama 3.1 8B fine-tuned (`train_3000`)** | **99.8%** (99.5–100%) | **0.998** | **1** | **1** |
| Naive Bayes (bag-of-words baseline) | 98.6% (97.8–99.3%) | 0.986 | 3 | 11 |
| Qwen 3.5 9B, zero-shot | 92.2% (90.4–93.9%) | 0.947 | 54 | 2 |
| Mistral 7B, zero-shot | 89.3% (87.2–91.2%) | 0.903 | 101 | 5 |
| Llama 3.1 8B, zero-shot | 87.7% (85.7–89.6%) | 0.891 | 121 | 1 |

Observations:
- **False alarms fell from 121 to 1**, the zero-shot models' main weakness;
  all 254 SpamAssassin emails are now correct.
- **It beats Naive Bayes** with non-overlapping confidence intervals, mainly
  by catching 10 of the 11 Nazario phishing emails Naive Bayes misses.
- **Same 99.8% on the 891 near-duplicate-free test emails**, so it is not
  remembering templated scams. All 1,000 answers parsed; validation accuracy
  99.5%.
- **The two errors are understandable:** a legitimate Enron notice ("login
  using your email … and GIS_ID as password") that reads like credential
  phishing, and a phishing email padded with random Arabic words to fool
  filters.
- **Caveat:** test emails come from the same sources as the training
  emails, so part of the gain may still be learning each source's style. A
  cross-source test (train without SpamAssassin, test on it) would show
  whether it generalises. This model outputs only a label — no explanation.

## Shortcut check (bag-of-words baseline)

`src/shortcut_check.py` trains a multinomial Naive Bayes classifier (word
counts only, standard library, ~1 s) and tests it on the same 1,000 test
emails. Results are saved to `results/shortcut_check.json`.

```
python3 src/shortcut_check.py
```

| Classifier | Trained on | Accuracy | F1 |
|---|---|---|---|
| Naive Bayes, full email text (what the LLMs see) | `train_full` (8,248) | **98.6%** | **0.986** |
| Naive Bayes, full email text | `train_500` | 97.8% | 0.978 |
| Naive Bayes, body only | `train_full` | 98.1% | 0.981 |
| Naive Bayes, subject only | `train_full` | 95.7% | 0.957 |
| Naive Bayes, "From:" line only | `train_full` | 92.6% | 0.924 |
| Qwen 3.5 9B, zero-shot | — | 92.2% | 0.947 |
| Mistral 7B, zero-shot | — | 89.3% | 0.903 |
| Llama 3.1 8B, zero-shot | — | 87.7% | 0.891 |

Words the model relies on most:
- **Legitimate:** `ilug, razor, rpm, spamassassin, exmh, perl, redhat,
  linux, kernel, listinfo` — the legitimate emails are largely Linux / tech
  mailing-list traffic.
- **Phishing:** `beneficiary, consignment, abidjan, lagos, barrister,
  deceased, nigeria` — formulaic 419 advance-fee scam wording.

So the shortcut is less about era than about **topic and source**. Excluding
the 109 near-duplicate test emails barely changes this (Naive Bayes 98.5%;
LLMs drop ~1 point).

What it means for the project:
- The **zero-shot** LLM results are still meaningful: the LLMs never saw our
  training data, so they could not learn the shortcut — and their errors are
  exactly the mailing-list mail Naive Bayes finds easy.
- A **fine-tuned** model will likely learn the shortcut too. Naive Bayes is
  therefore reported as a baseline row in every results table: a fine-tuned
  model's accuracy is only impressive relative to 98.6%. (Outcome: the
  `train_3000` model reached 99.8% — see
  [Fine-tuned results](#fine-tuned-results-label-only-qlora-train_3000).)
- **Explanation quality** (Step 7) is where LLMs can show what word counting
  cannot, so it carries more weight in the write-up.

## Repository layout

```
src/
  build_dataset.py   # Step 1: clean, de-duplicate, split (runs locally)
  prompts.py         # shared prompt, email formatting, answer parser
  inference.py       # 4-bit model loading + batched generation (Colab GPU)
  finetune.py        # Step 5: label-only QLoRA training of Llama (Colab GPU)
  shortcut_check.py  # Step 1: bag-of-words (Naive Bayes) baseline (runs locally)
  metrics.py         # Step 6: classification metrics for every results file (runs locally)
notebooks/
  01_baselines.ipynb # Step 3: run Llama / Qwen / Mistral baselines
  02_finetune.ipynb  # Step 5: QLoRA fine-tune Llama, evaluate on val + test
data/                # gitignored: zenodo/ (raw) and processed/ (splits)
results/
  shortcut_check.json  # Naive Bayes scores + top words
  metrics.md         # Step 6 tables (generated — don't edit by hand)
  metrics.json       # every metric + near-duplicate test ids, for plots
  raw/               # gitignored: model answers (copies from Google Drive)
```

## Project checklist

### Step 1 — Dataset ✅
- [x] Choose a phishing dataset (Zenodo curated corpora; rejected the original spam CSV)
- [x] Clean, de-duplicate and split into test / val / nested train sets — `src/build_dataset.py`
- [x] Verify splits are reproducible and leak-free
- [x] Shortcut check: bag-of-words Naive Bayes scores 98.6% — topic/source shortcut documented — `src/shortcut_check.py`

### Step 2 — Prompt and output format ✅
- [x] Fixed indicator list, system prompt, email formatting — `src/prompts.py`
- [x] Robust JSON parser, tested on messy outputs
- [x] Model-running code with 4-bit loading and resumable results — `src/inference.py`

### Step 3 — Baselines (Colab, no training) ✅
- [x] Accept the Llama 3.1 license on Hugging Face and add `HF_TOKEN` to Colab secrets
- [x] Run the 20-email smoke test in `notebooks/01_baselines.ipynb` (prompt v1: all models parsed 20/20; accuracy Llama 18, Qwen 17, Mistral 19)
- [x] Fix issues found in the smoke test: drop `taint.org` emails from the dataset; explain `[YEAR]` in the prompt (v2)
- [x] Re-run the 20-email smoke test with prompt v2 (all models parsed 20/20; accuracy Llama 17, Qwen 20, Mistral 19; no `[YEAR]` or `taint.org` confusion)
- [x] Full run: Llama 3.1 8B, Qwen 3.5 9B, Mistral 7B on all 1,000 test emails (F1: Qwen 0.947, Mistral 0.903, Llama 0.891 — see [Baseline results](#baseline-results-prompt-v2-1000-test-emails))

### Step 4 — Explanation targets for training
- [ ] Generate a reasoning + indicators answer for each training email with a larger model, given the true label
- [ ] Hand-check ~100 generated explanations for correctness
- [ ] Save as chat-format training files (prompt → `format_answer()` JSON)

### Step 5 — Fine-tune Llama 3.1 8B (Colab)
- [x] QLoRA training notebook (`notebooks/02_finetune.ipynb`, logic in `src/finetune.py`) — label-only; tested locally with the real Llama tokenizer and a tiny model
- [x] Train on `train_3000`; pick the best checkpoint on the val set (val accuracy 99.5%)
- [x] Run the fine-tuned model on the same 1,000 test emails (99.8% accuracy, F1 0.998 — see [Fine-tuned results](#fine-tuned-results-label-only-qlora-train_3000))
- [ ] Learning curve: also train on `train_500`, `train_1500` (`train_full` optional; note its different Nazario/Nigerian mix)
- [ ] Optional: cross-source test — train without SpamAssassin, test on it
- [ ] Later: second fine-tune on label + explanations (needs Step 4)

### Step 6 — Classification metrics
- [x] Accuracy, precision, recall, F1 per model, strict and lenient, with 95% bootstrap CIs — `src/metrics.py`
- [x] Confusion matrices and per-source results
- [x] Parse-failure rate per model
- [x] Naive Bayes as a baseline row in every table
- [x] Also report every model on the 891 test emails without near-duplicates in training
- [ ] Learning-curve plot (F1 vs training size, Naive Bayes included) — after Step 5

### Step 7 — Explanation evaluation
- [ ] Sample ~100 test emails; collect every model's explanation
- [ ] Score correctness, clarity, and indicator relevance (1–5) with two raters
- [ ] Compare indicator frequencies across models
- [ ] Optional: LLM-as-judge as a secondary score

### Step 8 — Write-up test
- [ ] Results tables and figures
- [ ] Error analysis with example emails
- [ ] Limitations (era gap, lost URLs, limited modern phishing)

Data preparation runs locally as scripts in `src/`; GPU work runs in Colab
notebooks in `notebooks/` that import from `src/`.
