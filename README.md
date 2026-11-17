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
- **Lost URLs:** most Nazario emails had their link text removed during the
  original HTML-to-text conversion; only a `has_urls` flag remains.
- **Limited modern phishing:** ~1,500 unique Nazario emails.

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
| Fine-tuned LoRA adapters (Step 5, ~100–200 MB each) | Google Drive | Yes |
| Downloaded model weights (~5–6 GB each) | Colab's temporary disk | No — re-downloaded in 1–3 min when needed |
| Processed dataset | Colab's temporary disk | No — rebuilt from Zenodo in ~30 s |

- Each model only has to be run **once per split**. Re-running with the same
  settings skips emails already in the results file, and an interrupted run
  continues where it stopped.
- Metrics and explanation scoring (Steps 6–7) only read the saved answer
  files, so they need no GPU and can run locally.

## Repository layout

```
src/
  build_dataset.py   # Step 1: clean, de-duplicate, split (runs locally)
  prompts.py         # shared prompt, email formatting, answer parser
  inference.py       # 4-bit model loading + batched generation (Colab GPU)
notebooks/
  01_baselines.ipynb # Step 3: run Llama / Qwen / Mistral baselines
data/                # gitignored: zenodo/ (raw) and processed/ (splits)
```

## Project checklist

### Step 1 — Dataset ✅
- [x] Choose a phishing dataset (Zenodo curated corpora; rejected the original spam CSV)
- [x] Clean, de-duplicate and split into test / val / nested train sets — `src/build_dataset.py`
- [x] Verify splits are reproducible and leak-free
- [ ] Shortcut check: train a simple bag-of-words classifier; if it scores ~99%, document the era shortcut

### Step 2 — Prompt and output format ✅
- [x] Fixed indicator list, system prompt, email formatting — `src/prompts.py`
- [x] Robust JSON parser, tested on messy outputs
- [x] Model-running code with 4-bit loading and resumable results — `src/inference.py`

### Step 3 — Baselines (Colab, no training)
- [x] Accept the Llama 3.1 license on Hugging Face and add `HF_TOKEN` to Colab secrets
- [x] Run the 20-email smoke test in `notebooks/01_baselines.ipynb` (prompt v1: all models parsed 20/20; accuracy Llama 18, Qwen 17, Mistral 19)
- [x] Fix issues found in the smoke test: drop `taint.org` emails from the dataset; explain `[YEAR]` in the prompt (v2)
- [x] Re-run the 20-email smoke test with prompt v2 (all models parsed 20/20; accuracy Llama 17, Qwen 20, Mistral 19; no `[YEAR]` or `taint.org` confusion)
- [ ] Full run: Llama 3.1 8B, Qwen 3.5 9B, Mistral 7B on all 1,000 test emails

### Step 4 — Explanation targets for training
- [ ] Generate a reasoning + indicators answer for each training email with a larger model, given the true label
- [ ] Hand-check ~100 generated explanations for correctness
- [ ] Save as chat-format training files (prompt → `format_answer()` JSON)

### Step 5 — Fine-tune Llama 3.1 8B (Colab)
- [ ] QLoRA training notebook (`notebooks/02_finetune.ipynb`)
- [ ] Train on `train_3000`; pick the best checkpoint on the val set
- [ ] Learning curve: also train on `train_500`, `train_1500`, `train_full`
- [ ] Run the fine-tuned model(s) on the same 1,000 test emails

### Step 6 — Classification metrics
- [ ] Accuracy, precision, recall, F1 per model (unparseable answers count as wrong)
- [ ] Confusion matrices and per-source results (Nazario vs Nigerian)
- [ ] Parse-failure rate per model
- [ ] Learning-curve plot (F1 vs training size)

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
