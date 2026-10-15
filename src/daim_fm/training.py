"""A compact, explicit CPU training loop."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch

from .model import TransformerLM
from .text_data import causal_batch


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    steps: int = 50
    learning_rate: float = 3e-4
    eval_interval: int = 10
    eval_batches: int = 2
    device: str = "cpu"

    def __post_init__(self) -> None:
        for field in ("batch_size", "steps", "eval_interval", "eval_batches"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.device != "cpu":
            raise ValueError("the introductory training loop supports device='cpu' only")


def set_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    batches: int,
    generator: torch.Generator,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = causal_batch(
            tokens,
            context_length=model.config.context_length,
            batch_size=batch_size,
            generator=generator,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train(was_training)
    return float(np.mean(losses))


def train_language_model(
    model: TransformerLM,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    config: TrainingConfig,
    seed: int,
) -> list[dict[str, Any]]:
    """Train one model and return serialisable step-level measurements."""
    set_seed(seed)
    model.to(config.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    measurements: list[dict[str, Any]] = []

    for step in range(1, config.steps + 1):
        inputs, targets = causal_batch(
            train_tokens,
            context_length=model.config.context_length,
            batch_size=config.batch_size,
            generator=train_generator,
            device=config.device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        record: dict[str, Any] = {"step": step, "training_loss": float(loss.item())}
        if step == 1 or step % config.eval_interval == 0 or step == config.steps:
            record["validation_loss"] = estimate_loss(
                model,
                validation_tokens,
                batch_size=config.batch_size,
                batches=config.eval_batches,
                generator=eval_generator,
            )
        measurements.append(record)
    return measurements
