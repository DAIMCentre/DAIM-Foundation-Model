"""A deliberately simple UTF-8 byte tokenizer."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal


class ByteTokenizer:
    """Map UTF-8 text reversibly to byte IDs 0 through 255."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        errors: Literal["strict", "replace"] = "strict",
    ) -> str:
        values = list(token_ids)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("token IDs must be integers")
        if any(value < 0 or value >= self.vocab_size for value in values):
            raise ValueError("token IDs must be between 0 and 255")
        return bytes(values).decode("utf-8", errors=errors)
