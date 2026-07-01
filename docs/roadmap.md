# Roadmap

## 1. Parity

Prompt:

```text
<bos> 7
```

Completion:

```text
odd <eos>
```

Reward:

```text
1.0 if the first completion token is correct, else 0.0
```

This is the smallest useful environment for checking GRPO mechanics.

## 2. Short Arithmetic

Extend the tokenizer with `+`, `-`, and multi-token numeric answers.

Prompt:

```text
<bos> 4 + 5
```

Completion:

```text
9 <eos>
```

Reward remains exact match on the final parsed answer.

## 3. Template Word Problems

Add tiny GSM8K-style templates without using GSM8K yet:

```text
sam has 4 apples and buys 5 more. how many apples?
```

Keep answers short at first. Add reasoning traces only after the answer-only setup
is stable.

## 4. Query Rewriting

Replace the reward function with retrieval quality:

```text
reward = ndcg_at_10(bm25(rewrite), qrels)
```

The GRPO loop should not need to change much. The environment changes from
`parity_rewards` to `query_rewrite_rewards`.

## 5. DDP

Keep each prompt group on one rank:

```text
rank 0: prompt batch A -> sample G completions -> normalize rewards per prompt
rank 1: prompt batch B -> sample G completions -> normalize rewards per prompt
DDP averages gradients after backward
```

This avoids cross-rank reward normalization and makes the first multi-GPU version
mostly a wrapper around the existing training step.
