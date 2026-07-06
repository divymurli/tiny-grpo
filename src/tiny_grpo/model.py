from __future__ import annotations

import math

import torch
from torch import nn


class TinyCausalTransformer(nn.Module):
    """A small decoder-only language model for the toy GRPO task.

    The model predicts the next token for a sequence of token ids. It uses
    `nn.TransformerEncoder` with a causal attention mask, which makes it behave
    like a decoder-only Transformer while keeping the implementation compact.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Create the Transformer, embeddings, normalization, and LM head.

        Args:
            vocab_size: Number of tokens in `TinyTokenizer`.
            max_seq_len: Maximum prompt-plus-completion length.
            d_model: Hidden size of token and position embeddings.
            n_heads: Number of attention heads per layer.
            n_layers: Number of Transformer blocks.
            dropout: Dropout probability used inside Transformer layers.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear and embedding weights with small Gaussian noise."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute next-token logits for each position.

        Args:
            input_ids: LongTensor of shape `[batch_size, seq_len]`.

        Returns:
            FloatTensor of shape `[batch_size, seq_len, vocab_size]`. Position
            `t` contains logits for predicting the token after the prefix ending
            at position `t`.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) * math.sqrt(self.token_emb.embedding_dim)
        x = x + self.pos_emb(positions)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)
