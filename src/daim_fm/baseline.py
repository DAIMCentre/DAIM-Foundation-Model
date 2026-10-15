"""Simple forecasting baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def last_value_forecast(history: ArrayLike, horizon: int) -> NDArray[np.float64]:
    """Forecast every future point using the latest observed value."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    return np.full(horizon, values[-1], dtype=float)

