from __future__ import annotations

import torch

from tiny_grpo.generate import gather_completion_logprobs


def group_advantages(
    rewards: torch.Tensor,
    batch_size: int,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize rewards within each prompt's sampled group.

    GRPO compares completions sampled for the same prompt. If rewards has shape
    `[batch_size * group_size]`, it is viewed as `[batch_size, group_size]`.
    Each row is normalized to roughly zero mean and unit variance, then flattened
    back to match the completion rows.
    """
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
    """Compute the token-level GRPO/PPO-style objective for one rollout batch.

    Args:
        policy: Trainable model being updated.
        reference: Frozen model used for the KL penalty.
        prompts: Repeated prompts with shape `[batch_size * group_size, prompt_len]`.
        completions: Sampled completions with shape
          `[batch_size * group_size, max_new_tokens]`.
        old_logprobs: Detached rollout logprobs from `sample_completions`, same
          shape as `completions`.
        completion_mask: 1/0 mask for real generated tokens, same shape.
        advantages: Group-normalized sequence advantages with shape
          `[batch_size * group_size]`.
        clip_eps: PPO clipping range around probability ratio 1.
        kl_beta: Weight on the sampled-token KL penalty to the reference model.

    Returns:
        `(loss, stats)` where `loss` is a differentiable scalar and `stats`
        contains detached logging values.

    The same sequence-level advantage is applied to every real token in that
    completion. `old_logprobs` anchor the probability ratio to the policy that
    generated the rollout; `new_logprobs` are recomputed with gradients from the
    current policy.
    """
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
        clipped_token_frac = ((ratio - 1.0).abs() > clip_eps).float()
        clipped_token_frac = (clipped_token_frac * completion_mask).sum() / denom

    stats = {
        "loss": float(loss.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "ref_kl": float((sampled_kl * completion_mask).sum().detach().cpu() / denom.cpu()),
        "clipped_token_frac": float(clipped_token_frac.detach().cpu()),
    }
    return loss, stats
