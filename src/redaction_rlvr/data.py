from __future__ import annotations

from typing import Any

from redaction_rlvr.inference import build_redaction_prompt


DATASET_NAME = "AdamLucek/open-pii-masking-en-us-30k"


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    question = str(example["question"])
    answer = str(example["answer"])
    info = example.get("info") or {}
    if not isinstance(info, dict):
        info = {}
    return {
        "prompt": build_redaction_prompt(question),
        "question": question,
        "answer": answer,
        "info": info,
    }


def load_pii_dataset(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    seed: int = 42,
):
    from datasets import load_dataset

    ds_all = load_dataset(DATASET_NAME)
    dataset = ds_all["train"]

    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))

    if num_eval_examples != -1:
        test_size = min(num_eval_examples, max(1, len(dataset) - 1))
    else:
        test_size = 0.2

    split = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"].map(normalize_example), split["test"].map(normalize_example)
