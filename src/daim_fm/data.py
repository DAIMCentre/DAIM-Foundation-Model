"""Tiny data helpers used by the introductory experiment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_synthetic_series(length: int = 48, seed: int = 42) -> NDArray[np.float64]:
    """Return a deterministic trend-and-seasonality series with small noise."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 4:
        raise ValueError("length must be an integer of at least 4")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    time = np.arange(length, dtype=float)
    rng = np.random.default_rng(seed)
    values = 10.0 + 0.08 * time + 1.5 * np.sin(2.0 * np.pi * time / 12.0)
    values += rng.normal(loc=0.0, scale=0.15, size=length)
    return values.astype(float)


def split_series(series: ArrayLike, horizon: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split a one-dimensional series into history and a held-out horizon."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("series must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("series must contain only finite values")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if horizon >= values.size:
        raise ValueError("horizon must be shorter than the series")

    return values[:-horizon].copy(), values[-horizon:].copy()

