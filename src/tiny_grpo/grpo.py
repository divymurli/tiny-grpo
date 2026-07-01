from __future__ import annotations

import torch

from tiny_grpo.generate import gather_completion_logprobs


def group_advantages(
    rewards: torch.Tensor,
    batch_size: int,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    grouped = rewards.view(batch_size, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True, unbiased=False)
    return ((grouped - mean) / (std + eps)).view(-1)


def grpo_loss(
    policy,
    reference,
    prompts: torch.Tensor,
    completions: torch.Tensor,
    old_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    new_logprobs = gather_completion_logprobs(policy, prompts, completions)
    with torch.no_grad():
        ref_logprobs = gather_completion_logprobs(reference, prompts, completions)

    ratio = torch.exp(new_logprobs - old_logprobs)
    token_advantages = advantages[:, None]
    unclipped = ratio * token_advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * token_advantages
    policy_loss = -torch.minimum(unclipped, clipped)

    log_ratio_ref = ref_logprobs - new_logprobs
    sampled_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    total_loss = policy_loss + kl_beta * sampled_kl
    denom = completion_mask.sum().clamp_min(1.0)
    loss = (total_loss * completion_mask).sum() / denom

    with torch.no_grad():
        approx_kl = ((new_logprobs - old_logprobs) * completion_mask).sum() / denom
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float()
        clip_frac = (clip_frac * completion_mask).sum() / denom

    stats = {
        "loss": float(loss.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "ref_kl": float((sampled_kl * completion_mask).sum().detach().cpu() / denom.cpu()),
        "clip_frac": float(clip_frac.detach().cpu()),
    }
    return loss, stats
