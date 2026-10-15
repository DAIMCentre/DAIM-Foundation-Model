"""A readable decoder-only Transformer built from PyTorch primitives."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import LMConfig


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask."""

    def __init__(self, config: LMConfig, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False, device=device)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False, device=device)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool, device=device)
        )
        self.register_buffer("causal_mask", mask.view(1, 1, config.context_length, config.context_length), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, width = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)

        query, key, value = map(split_heads, (query, key, value))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, width)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False, device=device),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False, device=device),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: LMConfig, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, device=device)
        self.attention = CausalSelfAttention(config, device=device)
        self.feed_forward_norm = nn.LayerNorm(config.d_model, device=device)
        self.feed_forward = FeedForward(config, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class TransformerLM(nn.Module):
    """A small causal language model with tied token/output weights."""

    def __init__(self, config: LMConfig, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, device=device)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model, device=device)
        self.blocks = nn.ModuleList(
            TransformerBlock(config, device=device) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model, device=device)
        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module: nn.Module) -> None:
        """Use a small, explicit initial scale instead of framework defaults."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must use torch.long")
        _, sequence_length = token_ids.shape
        if sequence_length == 0:
            raise ValueError("token_ids must not be empty")
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.config.context_length}"
            )
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.config.vocab_size):
            raise ValueError("token_ids contain values outside the vocabulary")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = F.linear(self.final_norm(hidden), self.token_embedding.weight)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be torch.long with the same shape as token_ids")
            loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
