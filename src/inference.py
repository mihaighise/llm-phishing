"""
inference.py — runs a model over a list of emails and saves its answers.

Used by the Colab notebooks (Step 3 baselines, Step 5 fine-tuned model). It
needs a CUDA GPU plus the `torch`, `transformers`, `accelerate` and
`bitsandbytes` packages, so it is NOT meant to run on your Mac.

THE MAIN IDEAS
--------------
* Every model is loaded in 4-bit (QLoRA-style "NF4") precision. That makes
  an 8-9B model fit on a free Colab T4 (16 GB), and — more importantly —
  the fine-tuned Llama will also be evaluated in 4-bit, so all four models
  run at the same precision. Fair comparison.
* Greedy decoding (do_sample=False), the equivalent of temperature 0: the
  same email always produces the same answer, so results are reproducible.
* Results are written to a JSONL file ONE BATCH AT A TIME. If Colab
  disconnects halfway, re-running skips the emails already done ("resume").
"""

import gc      # garbage collector, to free GPU memory between models
import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,         # loader for text-only models (Llama, Mistral)
    AutoModelForImageTextToText,  # loader for vision-language models (Qwen 3.5)
    AutoTokenizer,
    BitsAndBytesConfig,
)

# `prompts` lives in the same src/ folder. The notebook adds src/ to the
# Python path, so this import works there.
from prompts import build_messages, parse_response

# ---------------------------------------------------------------------------
# The models in the comparison.
# key   = short name used in result file names
# value = settings needed to load and prompt that model correctly
# ---------------------------------------------------------------------------
MODELS = {
    "llama31_8b": {
        # Gated model: you must accept Meta's license on its Hugging Face page
        # and be logged in with a token, or the download fails with 401/403.
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "loader": AutoModelForCausalLM,
    },
    "qwen35_9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        # Qwen 3.5 is natively multimodal (it can read images), so it uses
        # the image-text loader. We only ever give it text.
        "loader": AutoModelForImageTextToText,
        # Turn OFF Qwen's long "thinking" mode so it answers directly, like
        # the other models. This is passed to its chat template.
        "template_kwargs": {"enable_thinking": False},
    },
    "mistral_7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "loader": AutoModelForCausalLM,
    },
}


def load_model(name: str, adapter_path: str | None = None):
    """Load one model + its tokenizer in 4-bit. Returns (model, tokenizer).

    adapter_path:
        Optional folder with LoRA weights from fine-tuning (Step 5). When
        given, the adapter is attached on top of the 4-bit base model.
    """
    cfg = MODELS[name]

    # bfloat16 is the best number format for computation, but older GPUs
    # (the free T4) do not support it; they use float16 instead.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 4-bit quantization settings — the standard QLoRA recipe:
    #   nf4               = a 4-bit format designed for neural-network weights
    #   double_quant      = also compresses the quantization constants (saves ~0.4 GB)
    #   compute_dtype     = weights are stored in 4-bit but un-packed to this
    #                       format for the actual math
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"])
    # Batched generation needs every input in a batch to be the same length,
    # so shorter ones are padded. For text GENERATION the padding must go on
    # the LEFT, so every prompt ends right where the answer begins.
    tokenizer.padding_side = "left"
    # Llama and Mistral have no dedicated padding token; re-using the
    # end-of-sequence token is the standard workaround.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cfg["loader"].from_pretrained(
        cfg["hf_id"],
        quantization_config=quant,
        dtype=compute_dtype,
        device_map="auto",  # put the model on the GPU automatically
    )

    if adapter_path:
        # peft is the Hugging Face library for LoRA adapters. Imported here
        # so the baseline notebook does not require it.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()  # inference mode: disables dropout etc.
    return model, tokenizer


def unload_model(model) -> None:
    """Free GPU memory so the next model can be loaded in the same session."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _render_prompt(tokenizer, email: dict, template_kwargs: dict) -> str:
    """Turn one email into the model's exact input text using ITS OWN chat
    template (each model family uses different special tokens).

    add_generation_prompt=True appends the "assistant, your turn" marker, so
    the model starts writing its answer immediately.
    """
    try:
        return tokenizer.apply_chat_template(
            build_messages(email, use_system_role=True),
            tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
    except Exception:
        # Some templates reject a separate system message. Fall back to
        # putting the instructions inside the user message (same wording).
        return tokenizer.apply_chat_template(
            build_messages(email, use_system_role=False),
            tokenize=False, add_generation_prompt=True, **template_kwargs,
        )


@torch.inference_mode()  # disables gradient tracking: faster, less memory
def generate_batch(model, tokenizer, emails: list, template_kwargs: dict,
                   max_new_tokens: int = 400) -> list:
    """Run the model on a batch of emails; return their raw text answers."""
    prompts = [_render_prompt(tokenizer, e, template_kwargs) for e in emails]

    # add_special_tokens=False because the chat template already inserted
    # the beginning-of-text token; adding it twice confuses some models.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # cap on answer length (~300 words)
        do_sample=False,                # greedy decoding = temperature 0
        temperature=None, top_p=None, top_k=None,  # silence sampling warnings
        pad_token_id=tokenizer.pad_token_id,
    )

    # `outputs` contains prompt + answer. Cut off the prompt part (every
    # prompt in the batch has the same padded length) and decode the rest.
    answer_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)


def run_model(name: str, model, tokenizer, emails: list, out_path: Path,
              batch_size: int = 8, max_new_tokens: int = 400) -> list:
    """Run `model` over all `emails`, appending results to `out_path` (JSONL).

    Each output line keeps the email id, true label and source, the model's
    raw text (so we can re-parse later without re-running the GPU), and the
    parsed fields from parse_response().

    Emails already present in `out_path` are skipped, so a crashed or
    disconnected run can simply be started again.
    """
    template_kwargs = MODELS.get(name, {}).get("template_kwargs", {})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Resume support: collect ids that were already processed.
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [e for e in emails if e["id"] not in done]
    print(f"[{name}] {len(done)} already done, {len(todo)} to run")

    # Sorting by length puts similar-length emails in the same batch, which
    # means less padding and noticeably faster generation.
    todo.sort(key=lambda e: len(e["body"]))

    start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            answers = generate_batch(model, tokenizer, batch, template_kwargs,
                                     max_new_tokens)
            for email, raw in zip(batch, answers):
                row = {
                    "id": email["id"],
                    "model": name,
                    "true_label": email["label"],
                    "source": email["source"],
                    "raw_output": raw,
                    **parse_response(raw),  # label, confidence, indicators, ...
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # make sure the batch is on disk before continuing

            # Simple progress report with an estimated time remaining.
            n_done = i + len(batch)
            rate = n_done / (time.time() - start)
            eta_min = (len(todo) - n_done) / rate / 60
            print(f"[{name}] {n_done}/{len(todo)}  ({rate:.2f} emails/s, "
                  f"~{eta_min:.0f} min left)", end="\r")
    print()

    # Return every result for this model (old + new) for quick inspection.
    with open(out_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
