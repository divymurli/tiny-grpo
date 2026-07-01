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
    input_ids = torch.cat([prompts, completions], dim=1)
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    logprobs = F.log_softmax(logits, dim=-1)
    token_logprobs = logprobs.gather(2, targets[:, :, None]).squeeze(2)
    prompt_len = prompts.shape[1]
    return token_logprobs[:, prompt_len - 1 :]
