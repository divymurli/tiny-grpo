from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.data import make_parity_loader
from tiny_grpo.tokenizer import TinyTokenizer
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect batches from the parity dataloader.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
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
