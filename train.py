from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.data import all_digit_prompts, infinite_loader, make_parity_loader
from tiny_grpo.generate import sample_completions
from tiny_grpo.grpo import grpo_loss, group_advantages
from tiny_grpo.model import TinyCausalTransformer
from tiny_grpo.rewards import parity_rewards
from tiny_grpo.tokenizer import TinyTokenizer
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line hyperparameters for the toy GRPO run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--dataset-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model,
    tokenizer: TinyTokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> float:
    """Evaluate the current policy on all ten digit prompts.

    Eval uses one low-temperature completion per digit, so it is closer to a
    greedy accuracy check than to the stochastic training reward. In this toy
    environment the mean reward is also exact parity accuracy.
    """
    prompts, digits = all_digit_prompts(tokenizer, device)
    completions, _, _ = sample_completions(
        model=model,
        prompts=prompts,
        group_size=1,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=0.1,
    )
    rewards = parity_rewards(tokenizer, digits, completions, group_size=1)
    decoded = [tokenizer.decode(row.tolist()) for row in completions]
    pairs = ", ".join(f"{int(d)}->{text or '<empty>'}" for d, text in zip(digits.tolist(), decoded))
    print(f"eval accuracy={rewards.mean().item():.3f} | {pairs}")
    return float(rewards.mean().item())


def main() -> None:
    """Run GRPO training on the synthetic parity task.

    Each training step:
    1. reads a batch of digit prompts,
    2. samples `group_size` completions per prompt,
    3. scores completions with the parity reward,
    4. normalizes rewards within each prompt group,
    5. computes the GRPO loss against a frozen reference model,
    6. updates the trainable policy.
    """
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    tokenizer = TinyTokenizer()

    policy = TinyCausalTransformer(vocab_size=tokenizer.vocab_size).to(device)
    reference = copy.deepcopy(policy).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    loader = make_parity_loader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device,
        dataset_size=args.dataset_size,
        seed=args.seed,
    )
    batches = infinite_loader(loader)

    for step in range(1, args.steps + 1):
        policy.train()
        batch = next(batches)
        prompts = batch["prompts"]
        digits = batch["digits"]

        completions, old_logprobs, completion_mask = sample_completions(
            model=policy,
            prompts=prompts,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            eos_id=tokenizer.eos_id,
            temperature=args.temperature,
        )
        repeated_prompts = prompts.repeat_interleave(args.group_size, dim=0)
        rewards = parity_rewards(tokenizer, digits, completions, args.group_size)
        advantages = group_advantages(rewards, args.batch_size, args.group_size)

        loss, stats = grpo_loss(
            policy=policy,
            reference=reference,
            prompts=repeated_prompts,
            completions=completions,
            old_logprobs=old_logprobs,
            completion_mask=completion_mask,
            advantages=advantages,
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(
                f"step={step:04d} reward={rewards.mean().item():.3f} "
                f"adv_std={advantages.std(unbiased=False).item():.3f} "
                f"loss={stats['loss']:.4f} ref_kl={stats['ref_kl']:.4f} "
                f"clipped_token_frac={stats['clipped_token_frac']:.3f}"
        )
        if step % args.eval_every == 0:
            evaluate(policy, tokenizer, device, args.max_new_tokens)

    evaluate(policy, tokenizer, device, args.max_new_tokens)


if __name__ == "__main__":
    main()
