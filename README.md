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
`[YEAR]`), removes duplicates, and writes reproducible splits (seed 42) to
`data/processed/`:

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

Settings: zero-shot, greedy decoding (temperature 0), each model's own chat
template, 4-bit NF4 loading for every model, Qwen's thinking mode off.
Answers that cannot be parsed are counted as wrong.

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
local build), runs a 20-email smoke test on the validation set, and saves
answers to `MyDrive/llm-phishing/results/raw/`. For the full run, set
`SPLIT = "test"` and `N_EMAILS = None`. Runs are resumable after a disconnect.

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
- [ ] Accept the Llama 3.1 license on Hugging Face and add `HF_TOKEN` to Colab secrets
- [ ] Run the 20-email smoke test in `notebooks/01_baselines.ipynb`
- [ ] Confirm all three models parse close to 100%; fix the prompt on the **val** set if not
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
