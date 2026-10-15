"""Small, validated language-model configurations."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class LMConfig:
    """Describe one decoder-only Transformer without hiding its dimensions."""

    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "context_length",
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ValueError("dropout must be a number between 0 and 1")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be between 0 and 1")

    def with_overrides(self, **changes: object) -> "LMConfig":
        """Return a validated copy, useful for deliberately small tests."""
        return replace(self, **changes)


PRESETS: dict[str, LMConfig] = {
    "tiny": LMConfig(
        vocab_size=256,
        context_length=128,
        d_model=192,
        n_heads=6,
        n_layers=6,
        d_ff=768,
    ),
    "0.1b": LMConfig(
        vocab_size=16_384,
        context_length=512,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3_072,
    ),
}


def get_preset(name: str) -> LMConfig:
    """Return a named immutable configuration."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {choices}") from exc
