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

## Project plan

1. **Build dataset** — `src/build_dataset.py` (local) ✅
2. **Shared prompt + JSON output format** — `src/prompts.py` ✅
3. **Baseline inference** — Llama, Qwen, Mistral on the test set (Colab):
   `notebooks/01_baselines.ipynb` + `src/inference.py`
4. **Explanation targets** for the training set
5. **QLoRA fine-tune** of Llama 3.1 8B, incl. learning curve (Colab)
6. **Classification metrics** — accuracy, precision, recall, F1
7. **Explanation evaluation** — manual rubric on a ~100-email sample
8. **Write-up**

Data preparation runs locally as scripts in `src/`; GPU work runs in Colab
notebooks in `notebooks/` that import from `src/`.
