from __future__ import annotations

"""Inspect the parity reward function and grouped reward statistics.

Example ways to run:

    # Inspect the default handcrafted alignment and random reward summary.
    python scripts/inspect_rewards.py

    # Print a tiny CPU-only handcrafted example.
    python scripts/inspect_rewards.py --batch-size 2 --group-size 4 --device cpu

    # Estimate the random reward distribution over more batches.
    python scripts/inspect_rewards.py --num-random-batches 1000

    # See how group reward variance changes with more completions per prompt.
    python scripts/inspect_rewards.py --group-size 8

Example handcrafted output, with exact digits depending on the seed:

    digits:
    tensor([5, 6])

    digits.repeat_interleave(group_size=4):
    tensor([5, 5, 5, 5, 6, 6, 6, 6])

    00 digit=5 expected= odd completion_ids=[13, 2] decoded='odd' reward=1.0
    01 digit=5 expected= odd completion_ids=[14, 2] decoded='even' reward=0.0
    02 digit=5 expected= odd completion_ids=[2, 2] decoded='<empty>' reward=0.0
    03 digit=5 expected= odd completion_ids=[6, 2] decoded='3' reward=0.0

    group reward stats:
      group=00 digit=5 rewards=[1.0, 0.0, 0.0, 0.0] mean=0.250 std=0.433
      group=01 digit=6 rewards=[0.0, 1.0, 0.0, 0.0] mean=0.250 std=0.433

The random-completion section then estimates sparse-reward behavior before a
model has learned anything.
"""

import argparse
import sys
import textwrap
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.data import make_parity_loader
from tiny_grpo.rewards import parity_rewards
from tiny_grpo.tokenizer import TinyTokenizer
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse options for the reward-alignment inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect parity rewards and label alignment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              python scripts/inspect_rewards.py
              python scripts/inspect_rewards.py --batch-size 2 --group-size 4 --device cpu
              python scripts/inspect_rewards.py --num-random-batches 1000
              python scripts/inspect_rewards.py --group-size 8
            """
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--dataset-size", type=int, default=32)
    parser.add_argument("--num-random-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_demo_completions(tokenizer: TinyTokenizer, total: int, device: torch.device) -> torch.Tensor:
    """Create deterministic completions for hand-checking reward behavior.

    The pattern includes a correct-looking odd token, a correct-looking even
    token, an empty completion, and an irrelevant digit token. Repeating this
    pattern makes it easy to see which completions receive reward under each
    prompt digit.
    """
    pattern = [
        [tokenizer.token_to_id["odd"], tokenizer.eos_id],
        [tokenizer.token_to_id["even"], tokenizer.eos_id],
        [tokenizer.eos_id, tokenizer.eos_id],
        [tokenizer.token_to_id["3"], tokenizer.eos_id],
    ]
    rows = [pattern[i % len(pattern)] for i in range(total)]
    return torch.tensor(rows, dtype=torch.long, device=device)


def random_completions(
    tokenizer: TinyTokenizer,
    total: int,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample random completion token ids without using a model.

    This gives a quick baseline for reward sparsity and zero-variance groups
    before the policy learns anything.
    """
    return torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(total, max_new_tokens),
        dtype=torch.long,
        device=device,
    )


def print_alignment(
    tokenizer: TinyTokenizer,
    digits: torch.Tensor,
    completions: torch.Tensor,
    group_size: int,
) -> None:
    """Print row-by-row reward alignment for a handcrafted completion batch.

    The output makes `digits.repeat_interleave(group_size)` explicit, then shows
    how each completion row maps to a digit, expected answer, first content
    token, and scalar reward.
    """
    repeated_digits = digits.repeat_interleave(group_size)
    rewards = parity_rewards(tokenizer, digits, completions, group_size)
    grouped_rewards = rewards.view(digits.shape[0], group_size)
    group_means = grouped_rewards.mean(dim=1)
    group_stds = grouped_rewards.std(dim=1, unbiased=False)

    print("digits:")
    print(digits)
    print()
    print(f"digits.repeat_interleave(group_size={group_size}):")
    print(repeated_digits)
    print()

    for i, completion in enumerate(completions):
        digit = int(repeated_digits[i].item())
        expected = "even" if digit % 2 == 0 else "odd"
        first_token = tokenizer.first_content_token(completion.tolist())
        decoded = tokenizer.decode(completion.tolist()) or "<empty>"
        print(
            f"{i:02d} digit={digit} expected={expected:>4} "
            f"completion_ids={completion.tolist()} decoded={decoded!r} "
            f"first_content={first_token!r} reward={float(rewards[i].item()):.1f}"
        )

    print()
    print("group reward stats:")
    for group_idx, digit in enumerate(digits.tolist()):
        values = grouped_rewards[group_idx].tolist()
        print(
            f"  group={group_idx:02d} digit={int(digit)} rewards={values} "
            f"mean={float(group_means[group_idx].item()):.3f} "
            f"std={float(group_stds[group_idx].item()):.3f}"
        )


def summarize_random_distribution(
    tokenizer: TinyTokenizer,
    loader,
    group_size: int,
    num_batches: int,
    device: torch.device,
) -> None:
    """Estimate reward distribution for random completions.

    Prints mean reward, per-group reward standard deviation, zero-variance group
    count, and first-token frequencies. These are the same kinds of diagnostics
    that matter later for retrieval GRPO.
    """
    reward_counter: Counter[float] = Counter()
    first_token_counter: Counter[str] = Counter()
    total = 0
    total_reward = 0.0
    group_std_sum = 0.0
    group_count = 0
    zero_variance_groups = 0

    for batch_idx, batch in enumerate(loader, start=1):
        digits = batch["digits"]
        completions = random_completions(
            tokenizer=tokenizer,
            total=digits.shape[0] * group_size,
            max_new_tokens=2,
            device=device,
        )
        rewards = parity_rewards(tokenizer, digits, completions, group_size)
        grouped_rewards = rewards.view(digits.shape[0], group_size)
        group_stds = grouped_rewards.std(dim=1, unbiased=False)

        for completion in completions.tolist():
            first = tokenizer.first_content_token(completion)
            first_token_counter[first or "<none>"] += 1

        for reward in rewards.tolist():
            reward_counter[float(reward)] += 1

        total += rewards.numel()
        total_reward += float(rewards.sum().item())
        group_std_sum += float(group_stds.sum().item())
        group_count += group_stds.numel()
        zero_variance_groups += int((group_stds == 0).sum().item())

        if batch_idx >= num_batches:
            break

    print()
    print("random completion reward distribution")
    print(f"  samples: {total}")
    print(f"  mean_reward: {total_reward / max(1, total):.4f}")
    print(f"  mean_group_std: {group_std_sum / max(1, group_count):.4f}")
    print(f"  zero_variance_groups: {zero_variance_groups}/{group_count}")
    print(f"  counts: {dict(sorted(reward_counter.items()))}")
    print()
    print("first content token counts")
    for token, count in first_token_counter.most_common():
        print(f"  {token!r}: {count}")


def main() -> None:
    """Run handcrafted and random reward inspections for the parity task."""
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
    import ipdb
    ipdb.set_trace()

    batch = next(iter(loader))
    digits = batch["digits"]
    demo_completions = make_demo_completions(
        tokenizer=tokenizer,
        total=digits.shape[0] * args.group_size,
        device=device,
    )

    print("handcrafted completion alignment")
    print("=" * 34)
    print_alignment(tokenizer, digits, demo_completions, args.group_size)

    summarize_random_distribution(
        tokenizer=tokenizer,
        loader=loader,
        group_size=args.group_size,
        num_batches=args.num_random_batches,
        device=device,
    )


if __name__ == "__main__":
    main()
