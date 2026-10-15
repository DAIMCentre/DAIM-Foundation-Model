"""Small, explicit forecast metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_pairs(actual: ArrayLike, predicted: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("actual and predicted values must be one-dimensional")
    if actual_values.size == 0:
        raise ValueError("actual and predicted values must not be empty")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted values must have the same length")
    if not np.all(np.isfinite(actual_values)) or not np.all(np.isfinite(predicted_values)):
        raise ValueError("actual and predicted values must be finite")

    return actual_values, predicted_values


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return mean absolute error."""
    actual_values, predicted_values = _validated_pairs(actual, predicted)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return root mean squared error."""
    actual_values, predicted_values = _validated_pairs(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))

