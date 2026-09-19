#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

SYSTEM_FALLBACK = (
    "You are Omega Attention v0.1. Output only compact JSON with operation, "
    "target, confidence, reason_code."
)
SYSTEM_FALLBACK_V0_2 = (
    "You are Omega Attention v0.2. Select one attention state-transition "
    "operation for the given cognitive workspace. Return only compact JSON "
    "with operation, target, confidence, and reason_code."
)

KNOWN_PROTOCOLS = {"omega-attention-scenario/v0.1", "omega-attention-workspace/v0.2"}


def validate_jsonl_protocol(path, expected=None):
    """Fail before GPU setup if rows or train/eval files mix incompatible schemas."""
    observed = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            protocol = row.get("input", {}).get("protocol")
            if protocol not in KNOWN_PROTOCOLS:
                raise ValueError(f"{path}:{line_number}: unknown attention protocol {protocol!r}")
            if expected is not None and protocol != expected:
                raise ValueError(f"{path}:{line_number}: {protocol} != expected {expected}")
            if observed is not None and protocol != observed:
                raise ValueError(f"{path}:{line_number}: mixed {observed} and {protocol}")
            observed = protocol
    if observed is None:
        raise ValueError(f"{path}: empty attention dataset")
    return observed


def format_messages(row):
    v0_2 = row.get("input", {}).get("protocol") == "omega-attention-workspace/v0.2"
    fallback = SYSTEM_FALLBACK_V0_2 if v0_2 else SYSTEM_FALLBACK
    messages = row.get("messages")
    if messages:
        system = next((m["content"] for m in messages if m["role"] == "system"), fallback)
        user = next(m["content"] for m in messages if m["role"] == "user")
        assistant = next(m["content"] for m in messages if m["role"] == "assistant")
    else:
        system = fallback
        prefix = "Select the next attention operation for this cognitive workspace:\n" if v0_2 else "Select the next attention operation:\n"
        user = prefix + json.dumps(row["input"], separators=(",", ":"))
        assistant = json.dumps(row["output"], separators=(",", ":"))
    return (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + user
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + assistant
        + "<|im_end|>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
    parser.add_argument("--train", default="training/attention-v0/data/train.jsonl")
    parser.add_argument("--eval", default="training/attention-v0/data/eval.jsonl")
    parser.add_argument("--output", default="training/attention-v0/outputs/omega-attention-v0-lora")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument("--protocol", choices=sorted(KNOWN_PROTOCOLS), help="Require the train/eval workspace protocol to match this version")
    parser.add_argument("--disable-compile", action="store_true", help="Use Unsloth's documented local compile-disable setting for incompatible Triton installs")
    args = parser.parse_args()

    protocol = validate_jsonl_protocol(args.train, args.protocol)
    validate_jsonl_protocol(args.eval, protocol)
    print(f"training attention protocol: {protocol}")
    if args.disable_compile:
        # This local combination also compiles a loss through PyTorch after
        # Unsloth's own compilation is disabled. Both switches are read
        # before model imports and affect only this training process.
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    from datasets import load_dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files={"train": args.train, "eval": args.eval})

    def map_text(row):
        return {"text": format_messages(row)}

    dataset = dataset.map(map_text, remove_columns=dataset["train"].column_names)

    response_marker = "<|im_start|>assistant\n"

    def tokenize(row):
        text = row["text"]
        marker_index = text.index(response_marker) + len(response_marker)
        prefix = text[:marker_index]
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        # Must tokenize with the same `add_special_tokens` behavior as
        # `tokenized` above (the tokenizer's default, not explicitly
        # disabled here) - `mask_until` is a token-count offset into
        # `tokenized["input_ids"]`, so if the tokenizer prepends any special
        # token(s) by default (e.g. BOS) to `tokenized` but not to
        # `prefix_ids`, the count is off by that many tokens and the mask
        # silently falls short, leaving trailing prompt tokens unmasked.
        prefix_ids = tokenizer(
            prefix,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )["input_ids"]
        labels = list(tokenized["input_ids"])
        mask_until = min(len(prefix_ids), len(labels))
        labels[:mask_until] = [-100] * mask_until
        tokenized["labels"] = labels
        return tokenized

    dataset = dataset.map(tokenize, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=training_args,
    )
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
