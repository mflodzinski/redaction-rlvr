# RLVR PII Redaction

This repository trains and evaluates an LLM for sensitive-data masking with reinforcement learning from verifiable rewards. It follows the task setup from `ALucek/rl-for-llms`: examples come from `AdamLucek/open-pii-masking-en-us-30k`, the model replaces sensitive spans with the generic `[PII]` token, and completions are expected inside `<masked_output>...</masked_output>` XML tags.

All explanatory text in this repo is original/paraphrased, and no image assets are included.

## What Is Included

- Hugging Face dataset loading for `AdamLucek/open-pii-masking-en-us-30k`.
- Prompting that asks for `[PII]` masking inside `<masked_output>` XML.
- Decomposed verifiable rewards:
  - redaction accuracy,
  - output-format correctness,
  - redacted-span count correctness,
  - preservation of non-sensitive text.
- TRL GRPO training for RLVR fine-tuning.
- A layer-freezing experiment comparing early, later, and all decoder layers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,notebook]"
```

## Train One Model

```bash
python scripts/train_rlvr_redactor.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir outputs/redaction-rlvr \
  --num-train-examples 512 \
  --num-eval-examples 128 \
  --layer-mode all
```

`--layer-mode` can be `all`, `early`, or `later`. Early and later modes freeze most decoder layers and train only the selected slice.

## Compare Layer Choices

```bash
python scripts/run_layer_comparison.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir outputs/layer-comparison \
  --num-train-examples 512 \
  --num-eval-examples 128 \
  --max-steps 100
```

The runner trains three variants and writes metric files:

```text
outputs/layer-comparison/
├── early_metrics.json
├── later_metrics.json
├── all_metrics.json
└── summary.json
```

## Notebooks

- `notebooks/rlvr_redaction.ipynb` explains the dataset, parser contract, and reward functions.
- `notebooks/layer_comparison.ipynb` sets up the early/later/all layer experiment and shows how to inspect results.

## Reward Formula

The composite reward is:

```text
1.0 * redaction_accuracy
+ 0.1 * output_format
+ 0.5 * span_count
+ 0.3 * preservation
```

The exact-match reward remains the main signal. The count reward gives a useful partial signal when the output is close but not exact. The format reward keeps parsing reliable. The preservation reward discourages the model from masking harmless text just to avoid leaks.
