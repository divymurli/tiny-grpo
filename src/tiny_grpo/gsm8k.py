from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class GSM8KExample:
    """One GSM8K example loaded from local JSONL."""

    example_id: str
    question: str
    answer: str
    gold: str


class GSM8KJsonlDataset(Dataset[GSM8KExample]):
    """Read GSM8K-style examples from a JSONL file.

    Expected input rows look like:

    `{"id": "train-0", "question": "...", "answer": "... #### 42"}`

    The `gold` field is parsed once at load time from the answer string.
    """

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        """Load up to `limit` examples from `path`."""
        self.path = Path(path)
        self.examples = list(load_gsm8k_jsonl(self.path, limit=limit))

    def __len__(self) -> int:
        """Return the number of loaded examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> GSM8KExample:
        """Return one loaded example."""
        return self.examples[index]


class GSM8KCollator:
    """Collate GSM8K examples into prompt strings and gold answers."""

    def __init__(self, include_reasoning: bool = False) -> None:
        """Configure prompt style.

        `include_reasoning=False` asks for only the final number, which keeps the
        first GRPO smoke test focused on parsing and reward plumbing.
        """
        self.include_reasoning = include_reasoning

    def __call__(self, examples: list[GSM8KExample]) -> dict[str, list[str]]:
        """Return a batch dictionary consumed by GSM8K train/eval scripts."""
        questions = [example.question for example in examples]
        return {
            "ids": [example.example_id for example in examples],
            "questions": questions,
            "answers": [example.answer for example in examples],
            "gold": [example.gold for example in examples],
            "prompts": [
                make_gsm8k_prompt(question, include_reasoning=self.include_reasoning)
                for question in questions
            ],
        }


def load_gsm8k_jsonl(path: Path, limit: int | None = None) -> Iterable[GSM8KExample]:
    """Yield parsed GSM8K examples from a local JSONL file."""
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            row: dict[str, Any] = json.loads(line)
            answer = str(row["answer"])
            yield GSM8KExample(
                example_id=str(row.get("id", idx)),
                question=str(row["question"]),
                answer=answer,
                gold=parse_gsm8k_gold_answer(answer),
            )


def make_gsm8k_loader(
    path: str | Path,
    batch_size: int,
    shuffle: bool,
    seed: int,
    limit: int | None = None,
    include_reasoning: bool = False,
) -> DataLoader:
    """Build a DataLoader for a GSM8K JSONL split."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        GSM8KJsonlDataset(path, limit=limit),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=GSM8KCollator(include_reasoning=include_reasoning),
        generator=generator,
    )


def make_gsm8k_prompt(question: str, include_reasoning: bool = False) -> str:
    """Format a GSM8K question as a model prompt.

    The parser uses the last number in the model output as the predicted answer,
    so both prompt styles explicitly require the final numeric answer to appear
    at the very end.
    """
    if include_reasoning:
        return (
            "Solve the math problem. Show concise reasoning if needed. "
            "End your response with exactly 'Final answer: <number>', with the "
            "number as the final text in your response.\n\n"
            f"Question: {question}\n"
            "Solution:"
        )
    return (
        "Solve the math problem. Return only the final numeric answer, with the "
        "number as the final text in your response.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def parse_gsm8k_gold_answer(answer: str) -> str:
    """Parse the gold numeric answer from GSM8K's `####` answer format."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    parsed = parse_model_answer(answer)
    if parsed is None:
        raise ValueError(f"could not parse gold answer from: {answer!r}")
    return parsed


def parse_model_answer(text: str) -> str | None:
    """Extract a normalized numeric answer from model output text.

    The parser takes the last number in the output. This handles outputs like
    `"42"`, `"Answer: 42"`, `"Final answer: 42"`, and `"#### 42"`.
    """
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return normalize_number(matches[-1])


def normalize_number(text: str) -> str:
    """Normalize numeric strings for exact-match comparison."""
    value = text.strip().replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    if value == "-0":
        value = "0"
    return value


def gsm8k_rewards(
    outputs: list[str],
    gold_answers: list[str],
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str | None]]:
    """Return exact-match rewards for generated GSM8K outputs.

    `gold_answers` has one entry per prompt. `outputs` has `group_size` entries
    per prompt, laid out contiguously. The gold answers are repeated to align
    with the sampled completions, just like `repeat_interleave` in the parity
    reward.
    """
    gold_indices = torch.arange(len(gold_answers), device=device).repeat_interleave(group_size)
    parsed = [parse_model_answer(output) for output in outputs]
    rewards = torch.zeros(len(outputs), dtype=torch.float32, device=device)
    for i, pred in enumerate(parsed):
        gold = gold_answers[int(gold_indices[i].item())]
        rewards[i] = 1.0 if pred == gold else 0.0
    return rewards, parsed


def gsm8k_reward_stats(
    rewards: torch.Tensor,
    parsed: list[str | None],
    batch_size: int,
    group_size: int,
) -> dict[str, float]:
    """Compute logging stats for a GSM8K rollout batch."""
    grouped = rewards.view(batch_size, group_size)
    group_std = grouped.std(dim=1, unbiased=False)
    parse_failures = sum(value is None for value in parsed)
    return {
        "reward": float(rewards.mean().detach().cpu()),
        "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
        "mean_group_std": float(group_std.mean().detach().cpu()),
        "zero_var_frac": float((group_std == 0).float().mean().detach().cpu()),
        "parse_fail_frac": parse_failures / max(1, len(parsed)),
    }
