"""
finetune.py — Step 5 of the LLM phishing project: fine-tune Llama 3.1 8B.

Used by notebooks/02_finetune.ipynb on Colab. Needs a CUDA GPU plus `torch`,
`transformers`, `accelerate`, `bitsandbytes` and `peft`, so it is NOT meant
to run on your Mac.

WHAT KIND OF FINE-TUNING
------------------------
LABEL-ONLY classification fine-tuning. For every training email the model
gets EXACTLY the input the baselines got (same system prompt, same email
formatting, same chat template — see inference._render_prompt), and learns
to answer with just

    {"label": "phishing"}      or      {"label": "legitimate"}

The existing parser (prompts.parse_response) accepts this answer, so the
fine-tuned model is scored by the same code as the baselines. (The system
prompt still asks for confidence/indicators/reasoning; the fine-tuned model
learns to skip them. Explanation fine-tuning is a later, separate run.)

WHAT QLoRA MEANS
----------------
Fully re-training 8 billion weights needs ~100 GB of GPU memory. QLoRA
avoids that with two tricks:
  Q  — the original model is loaded in 4-bit and FROZEN (never changed).
  LoRA — small extra matrices ("adapters", ~0.5% of the model's size) are
         added next to the attention and feed-forward layers, and ONLY
         these are trained.
The result is a ~100-200 MB adapter folder that is attached on top of the
original Llama at test time (inference.load_model(..., adapter_path=...)).

ONLY THE ANSWER IS LEARNED ("loss masking")
-------------------------------------------
A training example is prompt + answer, ~800 tokens of which only ~8 are the
answer. If the model were trained on all tokens it would spend nearly all
its effort learning to predict emails. So every prompt token gets the label
-100, which PyTorch's loss function ignores: the model is graded only on
the answer tokens.
"""

import json
from pathlib import Path

import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# Our own modules in src/ (the notebook adds src/ to the Python path).
from inference import load_model, _render_prompt
from prompts import build_messages

# Value that tells the loss function "ignore this token" (PyTorch convention).
IGNORE_INDEX = -100

# Longest training example we accept, in tokens. A typical example is ~850
# tokens; only 23 of the 8,248 emails in train_full (0.3%: 17 legitimate,
# 6 phishing) are longer than 2,048. Those are SKIPPED rather than cut,
# because cutting would remove the answer at the end. Allowing them instead
# (up to ~3,500 tokens) would risk "CUDA out of memory" on an L4 whenever one
# lands in a batch. Only training is affected: all test emails are still
# evaluated, whatever their length.
MAX_LENGTH = 2048

# The layers that get LoRA adapters: all attention projections (q, k, v, o)
# and the feed-forward block (gate, up, down). Adapting all of them works
# better than attention alone, at little extra cost.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


# ===========================================================================
# 1. TURNING EMAILS INTO TRAINING EXAMPLES
# ===========================================================================

def label_only_answer(label: str) -> str:
    """The exact text the model should learn to write for one email."""
    return json.dumps({"label": label})


def build_example(tokenizer, email: dict) -> dict:
    """One training example: token ids, attention mask and loss labels.

    We render the conversation twice with the model's own chat template:
      prompt_text — system + user message + "assistant, your turn" marker.
                    Identical to what the model sees at test time.
      full_text   — the same, plus the assistant's answer and the template's
                    end-of-turn token (so the model also learns to STOP).
    The answer part is whatever full_text adds after prompt_text.
    """
    prompt_text = _render_prompt(tokenizer, email, template_kwargs={})
    # The same conversation the model sees at test time, plus the answer.
    messages = build_messages(email) + [
        {"role": "assistant", "content": label_only_answer(email["label"])}]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Safety check: the full conversation must START with the exact prompt
    # text, otherwise the split into "prompt" and "answer" would be wrong.
    if not full_text.startswith(prompt_text):
        raise ValueError("Chat template renders the prompt differently with "
                         "and without the answer; cannot mask the prompt.")
    answer_text = full_text[len(prompt_text):]

    # Tokenize the two parts separately so we know exactly where the answer
    # starts. add_special_tokens=False: the template already added the
    # begin-of-text token (same as at test time in inference.generate_batch).
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        # -100 for every prompt token: only the answer is graded.
        "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids,
    }


def build_dataset(tokenizer, emails: list) -> list:
    """Training examples for a list of emails, skipping over-long ones."""
    examples, skipped = [], 0
    for email in emails:
        example = build_example(tokenizer, email)
        if len(example["input_ids"]) > MAX_LENGTH:
            skipped += 1
            continue
        examples.append(example)
    lengths = sorted(len(e["input_ids"]) for e in examples)
    print(f"{len(examples)} examples (skipped {skipped} longer than "
          f"{MAX_LENGTH} tokens); tokens per example: median "
          f"{lengths[len(lengths) // 2]}, max {lengths[-1]}")
    return examples


# ===========================================================================
# 2. TRAINING
# ===========================================================================

def train_adapter(train_emails: list, val_emails: list, checkpoint_dir: Path,
                  adapter_dir: Path, epochs: int = 2, batch_size: int = 4,
                  effective_batch_size: int = 16, learning_rate: float = 2e-4,
                  lora_rank: int = 16, seed: int = 42) -> None:
    """QLoRA-train Llama 3.1 8B on `train_emails` and save the best adapter.

    checkpoint_dir — where a checkpoint is saved after every epoch (on
                     Google Drive, so a disconnect loses at most one epoch;
                     re-running continues from the last checkpoint).
    adapter_dir    — where the BEST epoch's adapter is saved at the end.
    epochs         — passes over the training data.
    batch_size     — emails per GPU step (lower it if out of memory).
    effective_batch_size — emails per weight update. With batch_size 4 and
                     16 here, gradients of 4 steps are added up before each
                     update ("gradient accumulation"): the same result as a
                     batch of 16, using the memory of a batch of 4.
    learning_rate  — 2e-4 is the usual QLoRA value.
    lora_rank      — size of the adapter matrices (16 = common default).

    "Best" = lowest loss on the validation answers (eval_loss), checked after
    every epoch. The test set is never used here.
    """
    # Imported here so that importing this module does not require peft.
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Same 4-bit loading as the baselines.
    model, tokenizer = load_model("llama31_8b")
    # For TRAINING, pad on the right: real tokens then start at position 0,
    # exactly as during inference. (Left padding is only needed when
    # generating; load_model sets it for that.)
    tokenizer.padding_side = "right"

    train_data = build_dataset(tokenizer, train_emails)
    val_data = build_dataset(tokenizer, val_emails)

    # Prepare the frozen 4-bit model for training: keeps a few small layers
    # in full precision for stability and turns on gradient checkpointing
    # (re-computes some values instead of storing them: ~2x less memory,
    # ~30% slower).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # The generation cache is useless during training and conflicts with
    # gradient checkpointing (it prints a warning every step otherwise).
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,  # scaling factor; 2x rank is the usual choice
        lora_dropout=0.05,         # randomly drops 5% of adapter inputs: less overfitting
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()  # e.g. "trainable: 42M of 8B (0.5%)"

    accumulation = max(1, effective_batch_size // batch_size)
    steps_per_epoch = max(1, len(train_data) // (batch_size * accumulation))
    use_bf16 = torch.cuda.is_bf16_supported()  # L4/A100 yes, T4 no

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        learning_rate=learning_rate,
        # Learning rate: rises over the first ~5% of steps ("warm-up", avoids
        # a shaky start), then decays smoothly to 0 ("cosine").
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps_per_epoch * epochs)),
        # 8-bit optimizer that can spill to CPU memory: saves GPU memory.
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=not use_bf16,
        # (Recent transformers versions print a FutureWarning that these two
        # are being renamed to "activation_checkpointing"; the old names
        # still work and also work on older versions, so we keep them.)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Evaluate on the validation set and save a checkpoint after every
        # epoch; at the end, reload the epoch with the lowest eval_loss.
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,          # print the training loss every 10 steps
        report_to="none",          # no external experiment-tracking service
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        # Pads each batch to its longest example; labels are padded with
        # -100 so padding is ignored by the loss too.
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=IGNORE_INDEX),
    )

    # Resume after a disconnect: continue from the newest saved checkpoint.
    last = get_last_checkpoint(str(checkpoint_dir)) if Path(checkpoint_dir).exists() else None
    if last:
        print(f"Resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    # trainer.model now holds the best epoch. save_pretrained on a peft
    # model writes ONLY the small adapter, not the 8B base model.
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Keep the loss history next to the adapter, for the write-up.
    (Path(adapter_dir) / "log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Adapter saved to {adapter_dir}")
