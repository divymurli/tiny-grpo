from __future__ import annotations

import torch

from tiny_grpo.tokenizer import TinyTokenizer


def parity_rewards(
    tokenizer: TinyTokenizer,
    digits: torch.Tensor,
    completions: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Score generated completions for the parity task.

    Each prompt has `group_size` sampled completions. If `digits` has shape
    `[batch_size]`, `completions` has shape
    `[batch_size * group_size, max_new_tokens]`. `repeat_interleave` expands the
    labels so completion rows line up with their original digit:

    `digits=[5, 6]`, `group_size=4` becomes `[5, 5, 5, 5, 6, 6, 6, 6]`.

    Reward is 1.0 when the first non-special generated token is the correct
    parity word, and 0.0 otherwise.
    """
    repeated_digits = digits.repeat_interleave(group_size)
    rewards = torch.zeros(completions.shape[0], device=completions.device)

    for i in range(completions.shape[0]):
        first_token = tokenizer.first_content_token(completions[i].tolist())
        expected = "even" if int(repeated_digits[i].item()) % 2 == 0 else "odd"
        rewards[i] = 1.0 if first_token == expected else 0.0

    return rewards
