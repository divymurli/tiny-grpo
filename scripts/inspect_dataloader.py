from __future__ import annotations

"""Inspect what the parity DataLoader emits.

Example ways to run:

    # Inspect three default batches on the best available device.
    python scripts/inspect_dataloader.py

    # Force CPU and print one small batch.
    python scripts/inspect_dataloader.py --batch-size 4 --num-batches 1 --device cpu

    # Change the shuffle seed to verify reproducibility.
    python scripts/inspect_dataloader.py --batch-size 4 --num-batches 2 --seed 123

Example output, with exact digit order depending on the seed:

    device: cpu
    vocab_size: 15
    pad_id=0 bos_id=1 eos_id=2

    batch 1
      prompts.shape: (4, 2)
      digits.shape:  (4,)
      prompts tensor:
    tensor([[ 1,  8],
            [ 1,  9],
            [ 1,  6],
            [ 1,  3]])
      digits tensor:
    tensor([5, 6, 3, 0])
      decoded rows:
        00: ids=[1, 8] tokens=['<bos>', '5'] digit=5 expected=odd
        01: ids=[1, 9] tokens=['<bos>', '6'] digit=6 expected=even
        02: ids=[1, 6] tokens=['<bos>', '3'] digit=3 expected=odd
        03: ids=[1, 3] tokens=['<bos>', '0'] digit=0 expected=even
"""

import argparse
import textwrap
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.data import make_parity_loader
from tiny_grpo.tokenizer import TinyTokenizer
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse options controlling how many dataloader batches to print."""
    parser = argparse.ArgumentParser(
        description="Inspect batches from the parity dataloader.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/inspect_dataloader.py
              python scripts/inspect_dataloader.py --batch-size 4 --num-batches 1 --device cpu
              python scripts/inspect_dataloader.py --batch-size 4 --num-batches 2 --seed 123
            """
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    """Print raw and decoded batches from the parity dataloader.

    This is a debugging script for checking that dataset examples, token ids,
    decoded tokens, labels, and device placement all line up before training.
    """
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    tokenizer = TinyTokenizer()

    loader = make_parity_loader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        dataset_size=args.dataset_size,
        seed=args.seed,
    )

    print(f"device: {device}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"pad_id={tokenizer.pad_id} bos_id={tokenizer.bos_id} eos_id={tokenizer.eos_id}")
    print()

    for batch_idx, batch in enumerate(loader, start=1):
        prompts = batch["prompts"]
        digits = batch["digits"]

        print(f"batch {batch_idx}")
        print(f"  prompts.shape: {tuple(prompts.shape)}")
        print(f"  digits.shape:  {tuple(digits.shape)}")
        print(f"  prompts.device: {prompts.device}")
        print(f"  digits.device:  {digits.device}")
        print(f"  prompts tensor:\n{prompts}")
        print(f"  digits tensor:\n{digits}")
        print("  decoded rows:")

        for row_idx, (prompt_ids, digit) in enumerate(zip(prompts.tolist(), digits.tolist())):
            decoded_tokens = [tokenizer.id_to_token[token_id] for token_id in prompt_ids]
            expected = "even" if int(digit) % 2 == 0 else "odd"
            print(
                f"    {row_idx:02d}: ids={prompt_ids} "
                f"tokens={decoded_tokens} digit={digit} expected={expected}"
            )

        print()

        if batch_idx >= args.num_batches:
            break


if __name__ == "__main__":
    main()
