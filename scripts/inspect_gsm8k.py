from __future__ import annotations

"""Inspect local GSM8K JSONL examples and prompts.

Examples:

    python scripts/inspect_gsm8k.py --path data/gsm8k/train.jsonl --limit 3
    python scripts/inspect_gsm8k.py --path data/gsm8k/train.jsonl --include-reasoning
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.gsm8k import GSM8KJsonlDataset, make_gsm8k_prompt, parse_model_answer


def parse_args() -> argparse.Namespace:
    """Parse inspection options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--include-reasoning", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Print examples, parsed gold answers, and formatted prompts."""
    args = parse_args()
    dataset = GSM8KJsonlDataset(args.path, limit=args.limit)
    for example in dataset:
        prompt = make_gsm8k_prompt(example.question, include_reasoning=args.include_reasoning)
        print("=" * 80)
        print(f"id: {example.example_id}")
        print(f"question:\n{example.question}")
        print()
        print(f"raw answer:\n{example.answer}")
        print()
        print(f"parsed gold: {example.gold}")
        print(f"parser sanity on raw answer: {parse_model_answer(example.answer)}")
        print()
        print(f"prompt:\n{prompt}")


if __name__ == "__main__":
    main()
