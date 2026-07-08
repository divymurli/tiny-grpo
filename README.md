# tiny-grpo

A small, from-scratch GRPO learning repo using only PyTorch.

The first experiment trains a tiny causal Transformer to answer whether a digit is
`odd` or `even`. The task is intentionally simple so the reinforcement learning
mechanics are easy to inspect:

1. sample several completions per prompt
2. score each completion with a deterministic reward
3. normalize rewards within each prompt group
4. recompute token log probabilities
5. apply the GRPO clipped policy objective
6. optionally add a sampled-token KL penalty to a frozen reference model

## Quickstart

```bash
cd /Users/divyanshumurli/Software/tiny-grpo
python -m venv .venv
source .venv/bin/activate
pip install torch
python train.py
```

For a short smoke test:

```bash
python train.py --steps 20 --batch-size 8 --group-size 4 --eval-every 10
```

## GSM8K Smoke Test

The GSM8K path is a bridge from the tiny parity task to real LLM GRPO. It uses
Hugging Face models/tokenizers and optional LoRA.

Install optional dependencies:

```bash
pip install -e ".[gsm8k]"
```

Prepare small local splits:

```bash
python scripts/prepare_gsm8k.py --output-dir data/gsm8k --train-limit 500 --test-limit 100
```

Inspect examples and prompts:

```bash
python scripts/inspect_gsm8k.py --path data/gsm8k/train.jsonl --limit 3
```

Evaluate a frozen model:

```bash
python scripts/evaluate_gsm8k.py \
  --path data/gsm8k/test.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --limit 20
```

Run a tiny GRPO post-training smoke test:

```bash
python scripts/train_grpo_gsm8k.py \
  --train-path data/gsm8k/train.jsonl \
  --val-path data/gsm8k/test.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --use-lora \
  --steps 50 \
  --batch-size 2 \
  --group-size 4 \
  --kl-beta 0.0
```

This is meant to validate generation, answer parsing, grouped rewards,
old/new/reference logprobs, and LoRA updates. It is not tuned for GSM8K
accuracy.

## Layout

```text
tiny-grpo/
  train.py
  src/tiny_grpo/
    data.py       # dataset, collator, and infinite loader
    generate.py   # autoregressive sampling
    grpo.py       # advantages and loss
    gsm8k.py      # GSM8K JSONL data, prompts, parsing, and rewards
    hf_grpo.py    # Hugging Face generation/logprob/GRPO helpers
    model.py      # tiny causal Transformer
    rewards.py    # parity reward
    tokenizer.py  # tiny hand-written tokenizer
    utils.py
```

## Multi-GPU Later

The single-GPU loop keeps every prompt's sampled group on the same device. That
is the shape you want before adding DDP:

```text
rank 0: prompts A, B, C -> groups -> local GRPO loss
rank 1: prompts D, E, F -> groups -> local GRPO loss
DDP: average gradients
```

That avoids cross-rank group normalization at first.
