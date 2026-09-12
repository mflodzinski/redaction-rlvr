from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run early/later/all layer RLVR comparison.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", default="outputs/layer-comparison")
    parser.add_argument("--num-train-examples", type=int, default=512)
    parser.add_argument("--num-eval-examples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-fraction", type=float, default=0.33)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reports = []

    for mode in ["early", "later", "all"]:
        subprocess.run(
            [
                "python",
                "scripts/train_rlvr_redactor.py",
                "--model",
                args.model,
                "--output-dir",
                str(root),
                "--num-train-examples",
                str(args.num_train_examples),
                "--num-eval-examples",
                str(args.num_eval_examples),
                "--max-steps",
                str(args.max_steps),
                "--seed",
                str(args.seed),
                "--layer-mode",
                mode,
                "--layer-fraction",
                str(args.layer_fraction),
            ],
            check=True,
        )

        model_dir = root / mode
        report_path = root / f"{mode}_metrics.json"
        subprocess.run(
            [
                "python",
                "scripts/evaluate_redactor.py",
                "--model",
                str(model_dir),
                "--num-eval-examples",
                str(args.num_eval_examples),
                "--seed",
                str(args.seed),
                "--output-json",
                str(report_path),
            ],
            check=True,
        )
        reports.append(json.loads(report_path.read_text(encoding="utf-8")) | {"layer_mode": mode})

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
