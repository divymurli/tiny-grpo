from __future__ import annotations

import torch
import torch.nn.functional as F


def completion_token_mask(completion_ids: torch.Tensor, eos_id: int | None, pad_id: int) -> torch.Tensor:
    """Build a 1/0 mask for generated completion tokens.

    The mask includes tokens through the first EOS token and excludes padding.
    It has the same shape as `completion_ids`.
    """
    not_pad = completion_ids.ne(pad_id)
    mask = torch.zeros_like(completion_ids, dtype=torch.float32)
    still_active = torch.ones(completion_ids.shape[0], dtype=torch.bool, device=completion_ids.device)
    for t in range(completion_ids.shape[1]):
        current = still_active & not_pad[:, t]
        mask[:, t] = current.float()
        if eos_id is not None:
            still_active = still_active & completion_ids[:, t].ne(eos_id)
    return mask


def gather_hf_completion_logprobs(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Recompute logprobs for fixed completion tokens under a HF causal LM.

    Args:
        prompt_ids: Padded prompt ids, shape `[n, prompt_len]`.
        prompt_mask: Attention mask for prompts, same shape as `prompt_ids`.
        completion_ids: Generated completion ids, shape `[n, completion_len]`.
        completion_mask: 1/0 mask for real completion tokens, same shape.

    Returns:
        Log probabilities for each completion token, shape
        `[n, completion_len]`.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask.long()], dim=1)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(2, targets[:, :, None]).squeeze(2)
    prompt_len = prompt_ids.shape[1]
    return token_logprobs[:, prompt_len - 1 :]


@torch.no_grad()
def sample_hf_completions(
    model,
    tokenizer,
    prompt_texts: list[str],
    group_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    """Sample grouped completions from a Hugging Face causal LM.

    Returns a dictionary containing repeated prompt tensors, completion tensors,
    old rollout logprobs, completion masks, and decoded completion strings.
    """
    model.eval()
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    prompt_ids = encoded["input_ids"].to(device)
    prompt_mask = encoded["attention_mask"].to(device)

    generated = model.generate(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        do_sample=True,
        num_return_sequences=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = prompt_ids.shape[1]
    completion_ids = generated[:, prompt_len:]
    repeated_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
    repeated_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)
    completion_mask = completion_token_mask(
        completion_ids,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
    )
    old_logprobs = gather_hf_completion_logprobs(
        model,
        repeated_prompt_ids,
        repeated_prompt_mask,
        completion_ids,
        completion_mask,
    )
    outputs = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt_ids": repeated_prompt_ids,
        "prompt_mask": repeated_prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "old_logprobs": old_logprobs,
        "outputs": outputs,
    }


def hf_grpo_loss(
    policy,
    reference,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    old_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss for padded Hugging Face causal-LM batches."""
    new_logprobs = gather_hf_completion_logprobs(
        policy,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
    )
    with torch.no_grad():
        ref_logprobs = gather_hf_completion_logprobs(
            reference,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )

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
        ref_kl = (sampled_kl * completion_mask).sum() / denom

    return loss, {
        "loss": float(loss.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
        "ref_kl": float(ref_kl.detach().cpu()),
        "clipped_token_frac": float(clipped_token_frac.detach().cpu()),
    }
