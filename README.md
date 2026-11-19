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

## Main findings

1. **Zero-shot LLMs catch almost all phishing but raise too many false
   alarms.** Recall is 0.990–0.998, but they flag 54–121 of the 500
   legitimate test emails (11–24%). Qwen 3.5 9B is the best of the three
   (F1 0.947). [Baseline results](#baseline-results-prompt-v2-1000-test-emails)
2. **Fine-tuning fixes the false alarms, with very little data.** QLoRA
   fine-tuned Llama 3.1 8B reaches F1 0.997–0.998 with 1–3 false alarms,
   and 500 training emails are as good as 3,000. It beats the Naive Bayes
   baseline (F1 0.986), and gives the same result on the test emails with no
   near-duplicate in training.
   [Fine-tuned results](#fine-tuned-results-label-only-qlora-train_500-train_1500-train_3000)
3. **But part of that score comes from learning the dataset's sources.**
   Trained without Nazario phishing, the fine-tuned model catches 83.7% of
   it instead of 99.7%. It misses ordinary credential phishing that its
   zero-shot version catches. It still does twice as well as Naive Bayes
   trained without Nazario (41.7%).
   [Cross-source test](#cross-source-test-one-source-left-out-of-training)
4. **Zero-shot models judge the sender and the era, not just the message.**
   Llama flags 34 of 35 Enron emails, citing Enron's 2001 bankruptcy. The
   models also call free-mail senders suspicious and sometimes invent
   evidence. [Error analysis](#error-analysis)
5. **In practice:** fine-tuning gives the best precision, and zero-shot
   models catch new kinds of phishing more reliably. A detector would
   need training data from many phishing types to get both. Explanations
   (the second goal) are not evaluated yet.

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
| train_1500_no_nazario | 1,500 | 750 | 750 | 0 | 750 |
| train_1500_no_nigerian | 1,500 | 750 | 750 | 750 | 0 |
| train_1500_no_spamassassin | 1,500 | 750 | 750 | 450 | 300 |

Training sets are nested (500 ⊂ 1,500 ⊂ 3,000 ⊂ full) for a learning-curve
experiment on how much fine-tuning data helps. Test and validation emails never
appear in any training set.

The `train_1500_no_<source>` sets are for the
[cross-source test](#cross-source-test-one-source-left-out-of-training): each
leaves one source out completely (replaced by the other source(s) of the same
label; `train_1500_no_spamassassin` has no SpamAssassin legit emails). Each
comes with a `val_no_<source>` validation set without that source. Adding them
did not change any other split file.

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
- **Attachments are missing:** some Nazario phishing is just "your statement
  is attached"; the attack was in the attachment, which the dataset does
  not contain. These are the phishing emails the zero-shot models miss.
- **Masking dates does not hide the era:** the models know the world as of
  today, so Llama flags 34 of 35 Enron emails partly because "Enron went
  bankrupt in 2001". See [Error analysis](#error-analysis).
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

## Fine-tuned results (label-only QLoRA, `train_500`, `train_1500`, `train_3000`)

Llama 3.1 8B fine-tuned on 500, 1,500 and 3,000 emails (2 epochs, best
epoch chosen on the validation set), run on an L4 on 2026-11-17/18, scored
on the same 1,000 test emails next to the zero-shot base models and Naive
Bayes. Strict scoring (unparseable answers count as wrong); CI = 95%
bootstrap interval. Full tables in [`results/metrics.md`](results/metrics.md).

| Model | Parsed | Accuracy (95% CI) | Precision | Recall | F1 (95% CI) | False alarms | Missed phishing |
|---|---|---|---|---|---|---|---|
| **Llama 3.1 8B fine-tuned (`train_3000`)** | 100.0% | **99.8%** (99.5–100%) | **0.998** | 0.998 | **0.998** (0.995–1.000) | **1** | 1 |
| Llama 3.1 8B fine-tuned (`train_1500`) | 100.0% | 99.7% (99.3–100%) | 0.996 | 0.998 | 0.997 (0.993–1.000) | 2 | 1 |
| Llama 3.1 8B fine-tuned (`train_500`) | 100.0% | 99.7% (99.3–100%) | 0.994 | **1.000** | 0.997 (0.993–1.000) | 3 | **0** |
| Naive Bayes (bag-of-words baseline) | 100.0% | 98.6% (97.8–99.3%) | 0.994 | 0.978 | 0.986 (0.978–0.993) | 3 | 11 |
| Qwen 3.5 9B, zero-shot | 97.6% | 92.2% (90.4–93.9%) | 0.902 | 0.996 | 0.947 (0.932–0.960) | 54 | 2 |
| Mistral 7B, zero-shot | 99.8% | 89.3% (87.2–91.2%) | 0.831 | 0.990 | 0.903 (0.883–0.921) | 101 | 5 |
| Llama 3.1 8B, zero-shot | 99.9% | 87.7% (85.7–89.6%) | 0.805 | 0.998 | 0.891 (0.872–0.909) | 121 | 1 |

Same models on the **891 test emails without a near-duplicate in training**:

| Model | Accuracy (95% CI) | Precision | Recall | F1 (95% CI) | False alarms | Missed phishing |
|---|---|---|---|---|---|---|
| **Llama 3.1 8B fine-tuned (`train_3000`)** | **99.8%** (99.4–100%) | **0.998** | 0.998 | **0.998** (0.994–1.000) | **1** | 1 |
| Llama 3.1 8B fine-tuned (`train_1500`) | 99.7% (99.2–100%) | 0.995 | 0.998 | 0.996 (0.991–1.000) | 2 | 1 |
| Llama 3.1 8B fine-tuned (`train_500`) | **99.8%** (99.4–100%) | 0.995 | **1.000** | **0.998** (0.994–1.000) | 2 | **0** |
| Naive Bayes (bag-of-words baseline) | 98.5% (97.8–99.2%) | 0.995 | 0.973 | 0.984 (0.975–0.992) | 2 | 11 |
| Qwen 3.5 9B, zero-shot | 91.4% (89.3–93.2%) | 0.885 | 0.995 | 0.937 (0.919–0.953) | 53 | 2 |
| Mistral 7B, zero-shot | 88.3% (86.1–90.2%) | 0.804 | 0.990 | 0.887 (0.863–0.907) | 99 | 4 |
| Llama 3.1 8B, zero-shot | 86.8% (84.6–88.9%) | 0.779 | 0.998 | 0.875 (0.852–0.896) | 116 | 1 |

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
  emails, so part of the gain is learning each source's style. The
  [cross-source test](#cross-source-test-one-source-left-out-of-training)
  confirms this: trained without Nazario, it catches 83.7% of Nazario
  phishing instead of 99.7%. This model outputs only a label — no explanation.
  (These bullets describe `train_3000`; smaller sets are compared below.)

### Effect of training-set size

Each fine-tuned model next to Naive Bayes trained on the **same** emails
(nested sets: 500 ⊂ 1,500 ⊂ 3,000). All 1,000 test emails, strict.

| Training emails | Llama FT F1 (95% CI) | Llama FT errors (false alarms + missed) | Validation correct | Naive Bayes F1 (same emails) |
|---|---|---|---|---|
| 0 (zero-shot) | 0.891 (0.872–0.909) | 122 (121 + 1) | — | — |
| 500 | 0.997 (0.993–1.000) | 3 (3 + 0) | 396/400 | 0.978 |
| 1,500 | 0.997 (0.993–1.000) | 3 (2 + 1) | 397/400 | 0.985 |
| 3,000 | 0.998 (0.995–1.000) | 2 (1 + 1) | 398/400 | 0.988 |

Correct answers per source (test, strict):

| Source | FT `train_500` | FT `train_1500` | FT `train_3000` |
|---|---|---|---|
| Nazario phishing (300) | 300 | 299 | 299 |
| Nigerian fraud (200) | 200 | 200 | 200 |
| SpamAssassin legitimate (254) | 253 | 252 | 254 |
| Nazario_5 legitimate (132) | 131 | 132 | 131 |
| Nigerian_5 legitimate (114) | 113 | 114 | 114 |

- **500 emails are already enough.** The three runs differ by at most one
  error, well inside each other's confidence intervals; more data only
  swaps a false alarm for a missed phishing email or vice versa.
- **Almost all of the gain comes from the first 500 emails:** errors drop
  from 122 (zero-shot) to 3. Fine-tuning mainly teaches the model what
  *legitimate* mail in this dataset looks like.
- **A few hard emails recur:** the Enron notice `c6fe173f` is flagged by
  `train_500` and `train_3000`, and Nazario phishing `af3940b3` is missed
  by `train_1500` and `train_3000`. With 2–3 errors per run the differences
  between sizes are noise; repeating each run with other seeds would be
  needed to rank them.
- **Every fine-tuned run beats Naive Bayes trained on the same emails**,
  mainly because Naive Bayes misses 11 Nazario phishing emails.

### Learning curve

![Learning curve: test F1 vs. number of training emails](results/learning_curve.svg)

Test F1 (strict) of the fine-tuned Llama and of Naive Bayes trained on the
same nested sets (500 ⊂ 1,500 ⊂ 3,000 ⊂ full); dashed lines are the
zero-shot models. Generated by `python3 src/metrics.py`, which adds a
fine-tuned point as soon as `results/raw/llama31_8b_ft_<train set>_test_all_v2.jsonl`
exists — so far `train_500`, `train_1500` and `train_3000` have been run. Naive Bayes is nearly flat
(0.978 → 0.986): the source/topic words it relies on are learnt from 500
emails already.

### Cross-source test: one source left out of training

Does the model learn *phishing*, or just *this dataset's sources*? Each
`train_1500_no_<source>` set leaves one source out of training (and out of
validation); we then test on that source's test emails. Compared with
`train_1500`, only that one source changes. Full tables:
[`results/metrics.md`](results/metrics.md#cross-source-test--one-source-left-out-of-training).

Held-out test emails correct (fine-tuned Llama with it = `train_1500`):

| Left out of training | Naive Bayes with it | Naive Bayes without it | Zero-shot LLMs | Fine-tuned Llama with it | Fine-tuned Llama without it |
|---|---|---|---|---|---|
| Nazario phishing (300) | 98.0% | **41.7%** | 98.3–99.7% | 99.7% | **83.7%** (79.3–87.7%) |
| Nigerian fraud (200) | 100.0% | **16.0%** | 99.5–100% | 100.0% | not run yet |
| SpamAssassin legitimate (254) | 99.6% | 99.6% | 78.0–86.6% | 99.2% | not run yet |

`train_1500_no_nazario` (run on an L4 on 2026-11-18, validated on the 280
`val_no_nazario` emails) on all 1,000 test emails: accuracy 94.9%
(93.5–96.3%), precision 0.996, recall 0.902, F1 0.946 (0.932–0.962), 2 false
alarms, 49 missed phishing. On the 700 other test emails it is as good as
`train_1500` (99.7%).

- **Fine-tuned Llama without Nazario misses 16% of Nazario phishing**
  (49 of 300), against 0.3% for zero-shot Llama. Trained only on 419 scams,
  it learnt a narrower idea of "phishing" and lost part of the general
  knowledge the base model had. The 49 are ordinary credential phishing
  (fake mailbox, password, WeTransfer, bank and PayPal alerts — see
  [Error analysis](#cross-source-model-what-it-misses-without-nazario)), and
  zero-shot Llama catches all 49.
- **It still generalises twice as well as Naive Bayes** (83.7% vs 41.7%):
  reading the email beats word counting on an unseen kind of phishing.
- **It keeps the fine-tuned model's precision:** still only 2 false alarms
  (zero-shot Llama: 121).

- **Leaving out a phishing source breaks word counting.** Trained only on
  419 scams, Naive Bayes misses 58% of modern Nazario phishing (and 84% of
  419 scams when trained only on Nazario). Its in-source accuracy stays at
  ~99%, so the overall accuracy hides this completely.
- **Leaving out a legitimate source does not:** "no scam words" is enough to
  call an email legitimate, whichever mailing list it came from.
- **What this means for the 99.7% figure:** with every source in training,
  part of fine-tuning's gain comes from recognising this dataset's sources.
  On a new kind of phishing, expect something between the zero-shot recall
  (~99.8%) and the cross-source result (~84%), not 99.7% on everything.
  Only one source has been held out so far; `train_1500_no_nigerian` would
  show whether the drop is general.
- **Caveat:** the remaining legitimate sources (mostly Enron and other
  mailing lists such as samba, apache, sourceforge) partly share the style
  of SpamAssassin, so that test is not a fully unseen genre. No SpamAssassin
  test email has a near-duplicate (≥50% shared 8-word sequences) in the
  other legitimate sources.

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
  [Fine-tuned results](#fine-tuned-results-label-only-qlora-train_500-train_1500-train_3000).)
- **Explanation quality** (Step 7) is where LLMs can show what word counting
  cannot, so it carries more weight in the write-up.

## Error analysis

`src/error_analysis.py` lists every error, compares error patterns across
models and writes [`results/error_analysis.md`](results/error_analysis.md).
The categories and notes on individual emails are our own judgement after
reading them (stored in the script).

```
python3 src/error_analysis.py
```

### Errors per model (1,000 test emails, strict)

| Model | Errors | False alarms | Missed phishing | Unparseable |
|---|---|---|---|---|
| Naive Bayes | 14 | 3 | 11 | 0 |
| Llama 3.1 8B | 123 | 121 | 1 | 1 |
| Mistral 7B | 107 | 101 | 4 | 2 |
| Qwen 3.5 9B | 78 | 54 | 0 | 24 |
| Llama fine-tuned (`train_500`) | 3 | 3 | 0 | 0 |
| Llama fine-tuned (`train_1500`) | 3 | 2 | 1 | 0 |
| Llama fine-tuned (`train_3000`) | 2 | 1 | 1 | 0 |

The zero-shot models disagree a lot: 184 legitimate emails are flagged by at
least one, only 24 by all three.

### Every error of the fine-tuned models (5 emails)

| Email | Wrong in run(s) | Zero-shot also wrong | What it is |
|---|---|---|---|
| `472e010a` | 500, 1500 | 3/3 | Subscribed marketing newsletter (dELiA*s) with tracking links |
| `6cc80071` | 500 | 0/3 | Law-firm email: only "attached are the POA documents" + a disclaimer |
| `74d7826c` | 1500 | 2/3 | MSN Groups post about investing, with promotional links |
| `af3940b3` | 1500, 3000 | 1/3 | **Real** PayPal phishing hidden letter by letter in Arabic filler text |
| `c6fe173f` | 500, 3000 | 3/3 | Real Enron IT notice: "login using your email … GIS_ID as password" |

### The 24 legitimate emails all three zero-shot models flag

| Cause | Emails | Fixed by fine-tuning (`train_3000`) |
|---|---|---|
| Sender looks odd (free-mail or list relay address, name ≠ address) | 10 | 10/10 |
| Invented evidence ("ENRON.com instead of enron.com", "a famous hacker") | 4 | 4/4 |
| Marketing newsletter with tracking links | 4 | 4/4 |
| Content genuinely resembles phishing (login details, legal threat) | 4 | 3/4 |
| Informal writing (deliberate misspellings, a bare link) | 2 | 2/2 |

### False alarms by sender (legitimate emails)

| Sender | Llama | Mistral | Qwen |
|---|---|---|---|
| **Enron** | **34/35** | 14/35 | 7/35 |
| slack.net (a 1990s internet provider) | 5/6 | 5/6 | 4/6 |
| Free-mail (hotmail, aol, …) | 12/29 | 11/29 | 9/29 |
| All other senders | 70/430 | 71/430 | 34/430 |

In 12 of the 35 Enron emails, Llama's reasoning mentions Enron's bankruptcy
or scandal — e.g. *"pretends to be from Enron, a company that went bankrupt
in 2001"*.

### Indicators cited in false alarms

| Model | Most cited (share of its false alarms) |
|---|---|
| Llama | suspicious_link 61%, poor_grammar 60%, impersonation 46% |
| Mistral | impersonation 77%, suspicious_link 69%, poor_grammar 39% |
| Qwen | impersonation 85%, suspicious_link 85%, poor_grammar 63% |

### Missed phishing

- **Zero-shot models (1–5 each):** mostly "your eStatement / payment advice
  is attached" — the attack was in the attachment, which the dataset does
  not include — plus plausible IT help-desk emails.
- **Naive Bayes (11):** mostly text deliberately scrambled to beat word
  filters ("##number##" inside every word, letters hidden in Arabic filler,
  random characters), plus Chinese/Portuguese emails and almost-empty
  bodies. The LLMs catch nearly all of these.

### Cross-source model: what it misses without Nazario

Llama fine-tuned on `train_1500_no_nazario` (its only phishing is 419
fraud) misses 49 of the 300 Nazario test emails:

| Kind of phishing | Missed | Zero-shot Llama correct | Fine-tuned `train_1500` correct |
|---|---|---|---|
| Mailbox / password / IT help-desk notice | 19 | 19/19 | 19/19 |
| Brand account alert (USAA, PayPal, Apple, Amazon, other banks, UPS) | 15 | 15/15 | 15/15 |
| Bank statement "attached" (Absa, Nedbank, Standard Bank) | 7 | 7/7 | 7/7 |
| File-sharing or "secure message" link (WeTransfer, encrypted mail) | 7 | 7/7 | 7/7 |
| Text hidden in Arabic filler | 1 | 1/1 | 0/1 |

None of these asks for money the way a 419 scam does: they imitate a
service and want a click or a login. Its only other errors are 2 false
alarms (`74d7826c`, the MSN Groups post `train_1500` also flags, and
`d1a25fba`, a lawyer's copyright-infringement letter).

### Takeaways

- **The zero-shot models judge the sender more than the message.**
  "Impersonation" is cited in 46–85% of their false alarms.
- **Masking dates does not hide the era:** the models read 20-year-old
  emails as if they arrived today (Enron "went bankrupt", slack.net
  "impersonates Slack").
- **Explanations can be confident and false** (invented evidence) —
  relevant for any later explanation evaluation.
- **Fine-tuning removes almost all of these errors**; what remains are
  genuinely hard or borderline emails, not labelling mistakes.
- **Reading beats word counting on obfuscated phishing**, the main kind
  Naive Bayes misses.
- **Fine-tuning narrows what counts as phishing:** without Nazario in
  training, Llama misses 49 credential-phishing emails that its zero-shot
  version catches.

## Repository layout

```
src/
  build_dataset.py   # Step 1: clean, de-duplicate, split (runs locally)
  prompts.py         # shared prompt, email formatting, answer parser
  inference.py       # 4-bit model loading + batched generation (Colab GPU)
  finetune.py        # Step 5: label-only QLoRA training of Llama (Colab GPU)
  shortcut_check.py  # Step 1: bag-of-words (Naive Bayes) baseline (runs locally)
  metrics.py         # Step 6: classification metrics for every results file (runs locally)
  error_analysis.py  # Step 8: error patterns + notes on individual emails (runs locally)
notebooks/
  01_baselines.ipynb # Step 3: run Llama / Qwen / Mistral baselines
  02_finetune.ipynb  # Step 5: QLoRA fine-tune Llama, evaluate on val + test
data/                # gitignored: zenodo/ (raw) and processed/ (splits)
results/
  shortcut_check.json  # Naive Bayes scores + top words
  metrics.md         # Step 6 tables (generated — don't edit by hand)
  metrics.json       # every metric + near-duplicate test ids + learning curve
  learning_curve.svg # Step 6 learning-curve plot (generated)
  error_analysis.md  # Step 8 error analysis (generated)
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
- [x] Run the fine-tuned model on the same 1,000 test emails (99.8% accuracy, F1 0.998 — see [Fine-tuned results](#fine-tuned-results-label-only-qlora-train_500-train_1500-train_3000))
- [x] Learning curve: also train on `train_500`, `train_1500` (F1 0.997 for both — 500 emails are enough; `train_full` skipped)
- [x] Cross-source test: `train_1500_no_nazario` (83.7% of Nazario phishing caught vs 99.7% with it, Naive Bayes 41.7% — see [Cross-source test](#cross-source-test-one-source-left-out-of-training))
- [ ] Optional: `train_1500_no_nigerian` (is the drop general?); `train_1500_no_spamassassin`
- [ ] Later: second fine-tune on label + explanations (needs Step 4)

### Step 6 — Classification metrics
- [x] Accuracy, precision, recall, F1 per model, strict and lenient, with 95% bootstrap CIs — `src/metrics.py`
- [x] Confusion matrices and per-source results
- [x] Parse-failure rate per model
- [x] Naive Bayes as a baseline row in every table
- [x] Also report every model on the 891 test emails without near-duplicates in training
- [x] Learning-curve plot (F1 vs training size, Naive Bayes included) — `results/learning_curve.svg`; fills in as the `train_500` / `train_1500` / `train_full` runs finish

### Step 7 — Explanation evaluation
- [ ] Sample ~100 test emails; collect every model's explanation
- [ ] Score correctness, clarity, and indicator relevance (1–5) with two raters
- [ ] Compare indicator frequencies across models
- [ ] Optional: LLM-as-judge as a secondary score

### Step 8 — Write-up test
- [ ] Results tables and figures
- [x] Error analysis with example emails — `src/error_analysis.py`, [Error analysis](#error-analysis), including the cross-source model's misses
- [ ] Limitations (era gap, lost URLs, limited modern phishing)

Data preparation runs locally as scripts in `src/`; GPU work runs in Colab
notebooks in `notebooks/` that import from `src/`.
