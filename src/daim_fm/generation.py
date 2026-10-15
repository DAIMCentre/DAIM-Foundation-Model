"""Autoregressive token generation using the same model used for training."""

from __future__ import annotations

import torch

from .model import TransformerLM


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    sample: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens greedily by default, with optional seeded sampling."""
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("prompt must have shape (1, sequence) and must not be empty")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0 or top_k > model.config.vocab_size
    ):
        raise ValueError("top_k must be between 1 and the vocabulary size")

    was_training = model.training
    model.eval()
    generated = prompt
    for _ in range(max_new_tokens):
        context = generated[:, -model.config.context_length :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / float(temperature)
        if top_k is not None:
            threshold = torch.topk(next_logits, top_k).values[:, -1, None]
            next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
        if sample:
            probabilities = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    model.train(was_training)
    return generated
