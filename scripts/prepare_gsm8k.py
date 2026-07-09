from __future__ import annotations

"""Prepare small local GSM8K JSONL splits.

Example:

    python scripts/prepare_gsm8k.py --output-dir data/gsm8k --train-limit 500 --test-limit 100

This script needs the optional `datasets` package and internet access the first
time the dataset is downloaded.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    """Parse output path and subset sizes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/gsm8k"))
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=100)
    return parser.parse_args()


def write_split(rows, path: Path, limit: int) -> None:
    """Write a Hugging Face dataset split to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for idx, row in enumerate(rows):
            if idx >= limit:
                break
            out = {
                "id": f"{path.stem}-{idx}",
                "question": row["question"],
                "answer": row["answer"],
            }
            f.write(json.dumps(out) + "\n")


def main() -> None:
    """Download GSM8K and write small JSONL subsets."""
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install datasets") from exc

    dataset = load_dataset("gsm8k", "main")
    write_split(dataset["train"], args.output_dir / "train.jsonl", args.train_limit)
    write_split(dataset["test"], args.output_dir / "test.jsonl", args.test_limit)
    print(f"wrote {args.output_dir / 'train.jsonl'}")
    print(f"wrote {args.output_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
