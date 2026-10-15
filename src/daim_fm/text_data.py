"""Text loading, checksums and causal next-token batches."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path

import torch

from .tokenizer import ByteTokenizer


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_text_tokens(path: str | Path, tokenizer: ByteTokenizer | None = None) -> torch.Tensor:
    """Read UTF-8 text and return a one-dimensional tensor of byte IDs."""
    input_path = Path(path)
    try:
        text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read text data {input_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"text data is not valid UTF-8: {input_path}") from exc
    tokens = (tokenizer or ByteTokenizer()).encode(text)
    if not tokens:
        raise ValueError(f"text data is empty: {input_path}")
    return torch.tensor(tokens, dtype=torch.long)


def causal_batch(
    tokens: torch.Tensor | Sequence[int],
    *,
    context_length: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample windows where each target is the corresponding input shifted once."""
    values = torch.as_tensor(tokens, dtype=torch.long)
    if values.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if not isinstance(context_length, int) or isinstance(context_length, bool) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    available_starts = values.numel() - context_length
    if available_starts <= 0:
        raise ValueError(
            f"need at least {context_length + 1} tokens for context_length={context_length}"
        )
    starts = torch.randint(available_starts, (batch_size,), generator=generator)
    inputs = torch.stack([values[start : start + context_length] for start in starts.tolist()])
    targets = torch.stack([values[start + 1 : start + context_length + 1] for start in starts.tolist()])
    return inputs.to(device), targets.to(device)
