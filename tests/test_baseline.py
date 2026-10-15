import numpy as np
import pytest

from daim_fm import last_value_forecast


def test_last_value_forecast_has_requested_shape_and_value() -> None:
    forecast = last_value_forecast([2.0, 4.0, 7.5], horizon=3)
    np.testing.assert_array_equal(forecast, [7.5, 7.5, 7.5])


def test_last_value_forecast_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        last_value_forecast([], horizon=2)


def test_last_value_forecast_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        last_value_forecast([1.0], horizon=0)

