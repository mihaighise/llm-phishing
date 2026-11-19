# llm-phishing
LLM-Based Phishing Detection

Can small open LLMs tell phishing emails from legitimate ones, and does
fine-tuning help? We compare:

| Model | Setup |
|---|---|
| Llama 3.1 8B Instruct | Zero-shot (prompting only) |
| Llama 3.1 8B Instruct | Fine-tuned with QLoRA on our training set |
| Qwen 3.5 9B | Zero-shot |
| Mistral 7B Instruct | Zero-shot |
| Naive Bayes (bag of words) | Simple baseline, to show how easy the dataset is |

All models get the same prompt and the same 1,000 test emails (500 phishing,
500 legitimate). They are scored with accuracy, precision, recall and F1
(95% bootstrap confidence intervals). The full write-up is
[`report.pdf`](report.pdf); detailed tables are in
[`results/metrics.md`](results/metrics.md), and the error analysis is in
[`results/error_analysis.md`](results/error_analysis.md).

## Main findings

1. **Zero-shot LLMs catch almost all phishing but raise too many false
   alarms.** Recall is 0.990–0.998, but they flag 54–121 of the 500
   legitimate test emails (11–24%). Qwen 3.5 9B is the best of the three.
2. **Fine-tuning fixes the false alarms, with very little data.** Fine-tuned
   Llama reaches F1 0.997–0.998 with 1–3 false alarms, and 500 training
   emails are as good as 3,000. It beats Naive Bayes (F1 0.986), and gives
   the same result on the 891 test emails with no near-duplicate in training.
3. **But part of that score comes from learning the dataset's sources.**
   Trained without Nazario phishing, the fine-tuned model catches 83.7% of it
   instead of 99.7%. It misses ordinary credential phishing (fake mailbox,
   password, WeTransfer and bank alerts) that its zero-shot version catches.
   It still does twice as well as Naive Bayes trained without Nazario (41.7%).
4. **Zero-shot models judge the sender and the era, not just the message.**
   Llama flags 34 of 35 Enron emails, citing Enron's 2001 bankruptcy. The
   models also call free-mail senders suspicious and sometimes invent
   evidence.
5. **In practice:** fine-tuning gives the best precision, and zero-shot
   models catch new kinds of phishing more reliably. A detector would need
   training data from many phishing types to get both.

### Results (1,000 test emails, strict scoring)

| Model | Accuracy | Precision | Recall | F1 (95% CI) | False alarms | Missed phishing |
|---|---|---|---|---|---|---|
| Llama 3.1 8B fine-tuned (`train_3000`) | **99.8%** | **0.998** | 0.998 | **0.998** (0.995–1.000) | **1** | 1 |
| Llama 3.1 8B fine-tuned (`train_1500`) | 99.7% | 0.996 | 0.998 | 0.997 (0.993–1.000) | 2 | 1 |
| Llama 3.1 8B fine-tuned (`train_500`) | 99.7% | 0.994 | **1.000** | 0.997 (0.993–1.000) | 3 | **0** |
| Naive Bayes (`train_full`) | 98.6% | 0.994 | 0.978 | 0.986 (0.978–0.993) | 3 | 11 |
| Llama 3.1 8B fine-tuned without Nazario | 94.9% | 0.996 | 0.902 | 0.946 (0.932–0.962) | 2 | 49 |
| Qwen 3.5 9B, zero-shot | 92.2% | 0.902 | 0.996 | 0.947 (0.932–0.960) | 54 | 2 |
| Mistral 7B, zero-shot | 89.3% | 0.831 | 0.990 | 0.903 (0.883–0.921) | 101 | 5 |
| Llama 3.1 8B, zero-shot | 87.7% | 0.805 | 0.998 | 0.891 (0.872–0.909) | 121 | 1 |

Strict scoring: an answer that cannot be parsed counts as wrong (24 of
Qwen's answers; with them counted, Qwen reaches 94.6%).

**Cross-source test:** each `train_1500_no_<source>` set leaves one source
out of training and validation. The table shows the share of that source's
test emails each model gets right.

| Left out of training | Naive Bayes with it | Naive Bayes without it | Zero-shot LLMs | Fine-tuned Llama with it | Fine-tuned Llama without it |
|---|---|---|---|---|---|
| Nazario phishing (300) | 98.0% | **41.7%** | 98.3–99.7% | 99.7% | **83.7%** |
| Nigerian fraud (200) | 100.0% | **16.0%** | 99.5–100% | 100.0% | not run |

## Dataset

[Phishing Email Curated Datasets](https://zenodo.org/records/8339691)
(Champa, Rabbi & Zibran, 2024, CC-BY-4.0):
- **Phishing:** Nazario (modern phishing, 2015–2022) and Nigerian Fraud
  (419 advance-fee scams).
- **Legitimate:** SpamAssassin, Nazario_5 and Nigerian_5 (mostly 1992–2008
  mailing-list and Enron mail).

`src/build_dataset.py` cleans the emails (HTML removed, years masked as
`[YEAR]`), removes duplicates and writes reproducible, leak-free splits
(seed 42): test 1,000, validation 400, and nested training sets of 500,
1,500, 3,000 and 8,248 emails, plus the cross-source sets.

**Limitations:**
- **Era gap:** legitimate emails are about 20 years older than the phishing
  emails, and the models notice this even with the years masked.
- **Source shortcut:** each source has its own vocabulary (Linux
  mailing-list words versus 419 wording), so word counting alone reaches
  98.6%.
- **Missing parts:** link URLs and attachments are not in the dataset.
- **Near-duplicates:** 109 test emails are near-duplicates of training
  emails, so results are also reported without them.

## Setup

**Locally** (Python 3, standard library only) to build the data and compute
the metrics:

```bash
mkdir -p data/zenodo
for f in Nazario Nigerian_Fraud Nazario_5 Nigerian_5 SpamAssasin; do
  curl -L -o data/zenodo/$f.csv \
    "https://zenodo.org/api/records/8339691/files/$f.csv/content"
done
python3 src/build_dataset.py     # splits -> data/processed/
python3 src/shortcut_check.py    # Naive Bayes baseline
python3 src/metrics.py           # -> results/metrics.md, metrics.json, learning_curve.svg
python3 src/error_analysis.py    # -> results/error_analysis.md
```

**On Colab** (GPU) to run the models:
1. Accept the license for
   [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   and add a Hugging Face read token as the Colab secret `HF_TOKEN`.
2. Choose an L4 or A100 runtime, then *Run all*:
   - [`01_baselines.ipynb`](https://colab.research.google.com/github/mihaighise/llm-phishing/blob/main/notebooks/01_baselines.ipynb)
     runs the three zero-shot models.
   - [`02_finetune.ipynb`](https://colab.research.google.com/github/mihaighise/llm-phishing/blob/main/notebooks/02_finetune.ipynb)
     fine-tunes Llama on the set chosen in `TRAIN_SET` (about 2 h on an L4
     for 1,500 emails) and tests it.
3. Answers are saved to `MyDrive/llm-phishing/results/raw/`. Copy them to
   `results/raw/` and run `python3 src/metrics.py`.

Runs can be resumed: finished emails are skipped, and training continues
from the last checkpoint (saved every 50 steps).

**Fine-tuning settings:** label-only QLoRA. The model sees the same prompt as
the zero-shot models and learns to answer `{"label": ...}`. It uses 4-bit
Llama, LoRA rank 16 on all attention and feed-forward layers, learning rate
2e-4, 16 emails per update and 2 epochs. The best checkpoint is chosen by
validation loss.

## Repository layout

```
src/
  build_dataset.py   # clean, de-duplicate, split
  prompts.py         # shared prompt, email formatting, answer parser
  inference.py       # 4-bit model loading + batched generation (GPU)
  finetune.py        # QLoRA fine-tuning of Llama (GPU)
  shortcut_check.py  # Naive Bayes baseline
  metrics.py         # all metrics, tables and the learning curve
  error_analysis.py  # error patterns + notes on individual emails
notebooks/
  01_baselines.ipynb # zero-shot runs (Colab)
  02_finetune.ipynb  # fine-tuning + evaluation (Colab)
results/             # generated tables and plots; raw/ (model answers) is gitignored
data/                # gitignored: raw and processed data
```

## Project checklist

### Step 1: Dataset ✅
- [x] Clean, de-duplicate and split the Zenodo data, reproducibly and with no leakage between splits
- [x] Shortcut check: Naive Bayes reaches 98.6%, so the source shortcut is documented

### Step 2: Prompt and inference ✅
- [x] Shared prompt, email formatting and robust answer parser
- [x] 4-bit, batched, resumable model runs

### Step 3: Zero-shot baselines ✅
- [x] Llama 3.1 8B, Qwen 3.5 9B and Mistral 7B on all 1,000 test emails

### Step 4: Fine-tune Llama 3.1 8B ✅
- [x] Label-only QLoRA on `train_500`, `train_1500` and `train_3000` (learning curve)
- [x] Cross-source test: `train_1500_no_nazario`
- [ ] Optional: `train_1500_no_nigerian`, to check whether the drop is general

### Step 5: Metrics and error analysis ✅
- [x] Accuracy, precision, recall and F1 with confidence intervals, strict and lenient, per source, and without near-duplicates
- [x] Learning-curve plot
- [x] Error analysis of every fine-tuned error, shared false alarms, sender effects and missed phishing

### Step 6: Write-up ✅
- [x] Report: results tables and figures, findings, limitations ([`report.pdf`](report.pdf))
