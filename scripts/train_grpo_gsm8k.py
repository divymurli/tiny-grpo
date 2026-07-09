from __future__ import annotations

"""Run a small GSM8K GRPO post-training smoke test.

Example:

    python scripts/train_grpo_gsm8k.py \
      --train-path data/gsm8k/train.jsonl \
      --val-path data/gsm8k/test.jsonl \
      --model-name Qwen/Qwen2.5-0.5B-Instruct \
      --use-lora \
      --steps 50 \
      --batch-size 2 \
      --group-size 4

This is meant to validate real-LLM GRPO mechanics, not to chase GSM8K SOTA.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.data import infinite_loader
from tiny_grpo.gsm8k import gsm8k_reward_stats, gsm8k_rewards, make_gsm8k_loader
from tiny_grpo.grpo import group_advantages
from tiny_grpo.hf_grpo import hf_grpo_loss, sample_hf_completions
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse GSM8K GRPO training options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=Path("data/gsm8k/train.jsonl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gsm8k-grpo"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--update-passes", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=50)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    return parser.parse_args()


def torch_dtype(name: str):
    """Resolve a dtype CLI value for model loading."""
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_policy_and_reference(args: argparse.Namespace, device: torch.device):
    """Load tokenizer, trainable policy, and frozen reference model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch_dtype(args.dtype)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise SystemExit("Install optional deps first: pip install peft") from exc
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        policy = get_peft_model(policy, config)
        policy.print_trainable_parameters()

    reference = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    return tokenizer, policy, reference


@torch.no_grad()
def evaluate(model, tokenizer, loader, args: argparse.Namespace, device: torch.device) -> float:
    """Evaluate exact-match accuracy with one low-temperature completion per question."""
    model.eval()
    total = 0
    correct = 0.0
    parse_fail = 0
    for batch in loader:
        rollout = sample_hf_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=batch["prompts"],
            group_size=1,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
            top_p=1.0,
            device=device,
        )
        outputs = rollout["outputs"]
        rewards, parsed = gsm8k_rewards(outputs, batch["gold"], group_size=1, device=device)
        correct += float(rewards.sum().item())
        total += len(outputs)
        parse_fail += sum(value is None for value in parsed)
    acc = correct / max(1, total)
    print(f"eval exact_match={acc:.3f} parse_fail={parse_fail}/{total}")
    return acc


def save_checkpoint(policy, tokenizer, output_dir: Path, step: int) -> None:
    """Save policy/tokenizer to a step-specific output directory."""
    path = output_dir / f"step-{step}"
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"saved checkpoint: {path}")


def main() -> None:
    """Train Qwen-style causal LM on GSM8K exact-answer GRPO rewards."""
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    tokenizer, policy, reference = load_policy_and_reference(args, device)
    policy.train()

    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    train_loader = make_gsm8k_loader(
        args.train_path,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        limit=args.train_limit,
        include_reasoning=args.include_reasoning,
    )
    val_loader = make_gsm8k_loader(
        args.val_path,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        limit=args.val_limit,
        include_reasoning=args.include_reasoning,
    )
    batches = infinite_loader(train_loader)

    for step in range(1, args.steps + 1):
        batch = next(batches)
        rollout = sample_hf_completions(
            model=policy,
            tokenizer=tokenizer,
            prompt_texts=batch["prompts"],
            group_size=args.group_size,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        outputs = rollout["outputs"]
        rewards, parsed = gsm8k_rewards(outputs, batch["gold"], args.group_size, device)
        advantages = group_advantages(rewards, len(batch["prompts"]), args.group_size)

        last_stats = {}
        for _ in range(args.update_passes):
            loss, last_stats = hf_grpo_loss(
                policy=policy,
                reference=reference,
                prompt_ids=rollout["prompt_ids"],
                prompt_mask=rollout["prompt_mask"],
                completion_ids=rollout["completion_ids"],
                old_logprobs=rollout["old_logprobs"],
                completion_mask=rollout["completion_mask"],
                advantages=advantages,
                clip_eps=args.clip_eps,
                kl_beta=args.kl_beta,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        reward_stats = gsm8k_reward_stats(rewards, parsed, len(batch["prompts"]), args.group_size)
        if step == 1 or step % 5 == 0:
            print(
                f"step={step:04d} reward={reward_stats['reward']:.3f} "
                f"group_std={reward_stats['mean_group_std']:.3f} "
                f"zero_var={reward_stats['zero_var_frac']:.3f} "
                f"parse_fail={reward_stats['parse_fail_frac']:.3f} "
                f"loss={last_stats['loss']:.4f} ref_kl={last_stats['ref_kl']:.4f} "
                f"clipped_token_frac={last_stats['clipped_token_frac']:.3f}"
            )
        if step % args.eval_every == 0:
            evaluate(policy, tokenizer, val_loader, args, device)
            save_checkpoint(policy, tokenizer, args.output_dir, step)

    evaluate(policy, tokenizer, val_loader, args, device)
    save_checkpoint(policy, tokenizer, args.output_dir, args.steps)


if __name__ == "__main__":
    main()
