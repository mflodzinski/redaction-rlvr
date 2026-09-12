from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from redaction_rlvr.data import load_pii_dataset
from redaction_rlvr.rewards import RedactionReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a PII masking model with RLVR.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="outputs/redaction-rlvr")
    parser.add_argument("--num-train-examples", type=int, default=512)
    parser.add_argument("--num-eval-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--layer-mode", choices=["all", "early", "later"], default="all")
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    return parser.parse_args()


def _get_decoder_layers(model: Any) -> list[Any]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("base_model", "model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    raise ValueError("Could not locate decoder layers for layer freezing.")


def set_trainable_layers(model: Any, mode: str, fraction: float = 0.33) -> list[int]:
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
        return list(range(len(_get_decoder_layers(model))))

    layers = _get_decoder_layers(model)
    train_count = max(1, round(len(layers) * fraction))
    if mode == "early":
        trainable = set(range(train_count))
    elif mode == "later":
        trainable = set(range(len(layers) - train_count, len(layers)))
    else:
        raise ValueError(f"Unknown layer mode: {mode}")

    for param in model.parameters():
        param.requires_grad = False
    for idx, layer in enumerate(layers):
        if idx in trainable:
            for param in layer.parameters():
                param.requires_grad = True

    return sorted(trainable)


def main() -> None:
    args = parse_args()
    train_dataset, eval_dataset = load_pii_dataset(
        num_train_examples=args.num_train_examples,
        num_eval_examples=args.num_eval_examples,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    trainable_layers = set_trainable_layers(model, args.layer_mode, args.layer_fraction)
    reward_model = RedactionReward()

    def reward_func(
        completions: list[str],
        answer: list[str],
        info: list[dict[str, Any]],
        **_: object,
    ) -> list[float]:
        return [
            reward_model(completion, expected_answer, info=metadata)
            for completion, expected_answer, metadata in zip(completions, answer, info)
        ]

    output_dir = Path(args.output_dir) / args.layer_mode
    config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=256,
        logging_steps=5,
        save_steps=50,
    )

    print(f"Training layer mode: {args.layer_mode}; trainable decoder layers: {trainable_layers}")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
