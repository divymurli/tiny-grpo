from __future__ import annotations

"""Evaluate a base model or saved adapter/checkpoint on local GSM8K JSONL."""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tiny_grpo.gsm8k import gsm8k_rewards, make_gsm8k_loader
from tiny_grpo.hf_grpo import sample_hf_completions
from tiny_grpo.utils import pick_device, seed_everything


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--print-failures", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device):
    """Load tokenizer and model, optionally attaching a PEFT adapter."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir or args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    if args.adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Install optional deps first: pip install peft") from exc
        model = PeftModel.from_pretrained(model, args.adapter_dir).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def main() -> None:
    """Run exact-match evaluation and print a few failures."""
    args = parse_args()
    seed_everything(args.seed)
    device = pick_device(args.device)
    tokenizer, model = load_model(args, device)
    loader = make_gsm8k_loader(
        args.path,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        limit=args.limit,
        include_reasoning=args.include_reasoning,
    )

    total = 0
    correct = 0.0
    parse_fail = 0
    failures_printed = 0
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

        for question, gold, output, pred, reward in zip(
            batch["questions"],
            batch["gold"],
            outputs,
            parsed,
            rewards.tolist(),
        ):
            if reward == 0.0 and failures_printed < args.print_failures:
                failures_printed += 1
                print("=" * 80)
                print(f"question: {question}")
                print(f"gold: {gold} pred: {pred}")
                print(f"output: {output!r}")

    print("=" * 80)
    print(f"exact_match={correct / max(1, total):.3f} parse_fail={parse_fail}/{total}")


if __name__ == "__main__":
    main()
