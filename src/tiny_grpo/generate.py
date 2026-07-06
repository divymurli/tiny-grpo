from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_completions(
    model,
    prompts: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `group_size` completions for each prompt.

    Args:
        model: Causal language model returning logits `[batch, seq, vocab]`.
        prompts: LongTensor of shape `[batch_size, prompt_len]`.
        group_size: Number of completions to sample per prompt.
        max_new_tokens: Fixed generation budget for each completion.
        eos_id: Token id that marks the end of a completion.
        temperature: Softmax temperature used only for sampling.

    Returns:
        A tuple `(completions, old_logprobs, completion_mask)`:
        - `completions`: sampled token ids with shape
          `[batch_size * group_size, max_new_tokens]`.
        - `old_logprobs`: log probabilities from the rollout policy at the
          moment tokens were sampled, same shape as `completions`.
        - `completion_mask`: 1 for real generated tokens through the first
          `<eos>`, 0 for forced tokens after the completion already ended.

    The function runs under `torch.no_grad()` because rollout collection should
    save values, not build a training graph.
    """
    model.eval()
    batch_size, prompt_len = prompts.shape
    input_ids = prompts.repeat_interleave(group_size, dim=0)
    generated = []
    logprobs = []
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_logprobs = torch.log(probs.gather(1, next_ids[:, None]).squeeze(1).clamp_min(1e-12))

        next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
        next_logprobs = torch.where(finished, torch.zeros_like(next_logprobs), next_logprobs)
        generated.append(next_ids)
        logprobs.append(next_logprobs)

        finished = finished | (next_ids == eos_id)
        input_ids = torch.cat([input_ids, next_ids[:, None]], dim=1)

    completions = torch.stack(generated, dim=1)
    old_logprobs = torch.stack(logprobs, dim=1)
    completion_mask = completion_token_mask(completions, eos_id)
    return completions, old_logprobs, completion_mask


def completion_token_mask(completions: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Build a loss mask for generated completion tokens.

    Tokens before and including the first `<eos>` receive mask value 1. Tokens
    after the first `<eos>` receive 0 because they are forced filler created to
    keep every completion at the same tensor length.
    """
    mask = torch.ones_like(completions, dtype=torch.float32)
    seen_eos = torch.zeros(completions.shape[0], dtype=torch.bool, device=completions.device)
    for t in range(completions.shape[1]):
        mask[:, t] = (~seen_eos).float()
        seen_eos = seen_eos | (completions[:, t] == eos_id)
    return mask


def gather_completion_logprobs(
    model,
    prompts: torch.Tensor,
    completions: torch.Tensor,
) -> torch.Tensor:
    """Recompute log probabilities for fixed completions under `model`.

    Args:
        model: Causal language model.
        prompts: Prompt token ids with shape `[n, prompt_len]`.
        completions: Completion token ids with shape `[n, max_new_tokens]`.

    Returns:
        Log probabilities of the provided completion tokens with shape
        `[n, max_new_tokens]`.

    This is used twice in GRPO: once for the trainable policy (`new_logprobs`)
    and once for the frozen reference policy (`ref_logprobs`). Prompt-token
    logprobs are sliced away so the loss only applies to generated tokens.
    """
    input_ids = torch.cat([prompts, completions], dim=1)
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(2, targets[:, :, None]).squeeze(2)
    prompt_len = prompts.shape[1]
    return token_logprobs[:, prompt_len - 1 :]
