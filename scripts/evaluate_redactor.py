from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from redaction_rlvr.data import load_pii_dataset
from redaction_rlvr.rewards import RedactionReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PII masking model with decomposed rewards.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: mean(row[key] for row in rows) for key in keys}


def main() -> None:
    args = parse_args()
    _, eval_dataset = load_pii_dataset(
        num_train_examples=max(args.num_eval_examples * 2, 4),
        num_eval_examples=args.num_eval_examples,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    reward = RedactionReward()
    rows: list[dict[str, float]] = []

    for example in eval_dataset:
        inputs = tokenizer(example["prompt"], return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0][inputs["input_ids"].shape[-1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        breakdown = reward.breakdown(completion, example["answer"], info=example["info"])
        rows.append(
            {
                "total": breakdown.total,
                "redaction_accuracy": breakdown.redaction_accuracy,
                "output_format": breakdown.output_format,
                "span_count": breakdown.span_count,
                "preservation": breakdown.preservation,
            }
        )

    report: dict[str, Any] = {
        "model": args.model,
        "num_eval_examples": len(rows),
        "metrics": summarize(rows),
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

