"""Transparent language-model and time-series research utilities."""

from .baseline import last_value_forecast
from .config import LMConfig, get_preset
from .data import generate_synthetic_series, split_series
from .generation import generate
from .metrics import mae, rmse
from .model import TransformerLM
from .tokenizer import ByteTokenizer
from .training import TrainingConfig, train_language_model

__all__ = [
    "ByteTokenizer",
    "LMConfig",
    "TrainingConfig",
    "TransformerLM",
    "generate",
    "generate_synthetic_series",
    "get_preset",
    "last_value_forecast",
    "mae",
    "rmse",
    "split_series",
    "train_language_model",
]
