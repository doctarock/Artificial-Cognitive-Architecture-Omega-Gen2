#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from generate_scenarios import SYSTEM_PROMPT as SYSTEM_PROMPT_V0_1
from generate_scenarios_v0_2 import SYSTEM_PROMPT as SYSTEM_PROMPT_V0_2


def format_prompt(workspace):
    protocol = workspace.get("protocol")
    if protocol == "omega-attention-workspace/v0.2":
        system, user_prefix = SYSTEM_PROMPT_V0_2, "Select the next attention operation for this cognitive workspace:\n"
    elif protocol == "omega-attention-scenario/v0.1":
        system, user_prefix = SYSTEM_PROMPT_V0_1, "Select the next attention operation for this cognitive workspace:\n"
    else:
        raise ValueError(f"unknown attention input protocol: {protocol!r}")
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        + "<|im_start|>user\n" + user_prefix
        + json.dumps(workspace, separators=(",", ":"))
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def predict(model, tokenizer, workspace, max_new_tokens):
    prompt = format_prompt(workspace)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only newly generated token IDs. String-slicing the decoded full
    # prompt was not reliable across tokenizer whitespace/special tokens.
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit")
    parser.add_argument("--adapter", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json")
    source.add_argument("--eval-jsonl")
    parser.add_argument("--predictions-out", help="JSONL result path for --eval-jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Maximum eval rows (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.eval_jsonl and not args.predictions_out:
        parser.error("--eval-jsonl requires --predictions-out")

    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    FastLanguageModel.for_inference(model)

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as handle:
            workspace = json.load(handle)
        print(predict(model, tokenizer, workspace, args.max_new_tokens))
    else:
        output_path = Path(args.predictions_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(args.eval_jsonl, "r", encoding="utf-8") as source_handle, output_path.open("w", encoding="utf-8") as result_handle:
            for line in source_handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["prediction"] = predict(model, tokenizer, row["input"], args.max_new_tokens)
                result_handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"predicted {count} held-out workspaces", flush=True)
                if args.limit and count >= args.limit:
                    break
        print(f"saved {count} predictions to {output_path}")


if __name__ == "__main__":
    main()
