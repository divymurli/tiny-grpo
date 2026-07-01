from __future__ import annotations

import torch

from tiny_grpo.tokenizer import TinyTokenizer


def parity_rewards(
    tokenizer: TinyTokenizer,
    digits: torch.Tensor,
    completions: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    repeated_digits = digits.repeat_interleave(group_size)
    rewards = torch.zeros(completions.shape[0], device=completions.device)

    for i in range(completions.shape[0]):
        first_token = tokenizer.first_content_token(completions[i].tolist())
        expected = "even" if int(repeated_digits[i].item()) % 2 == 0 else "odd"
        rewards[i] = 1.0 if first_token == expected else 0.0

    return rewards
