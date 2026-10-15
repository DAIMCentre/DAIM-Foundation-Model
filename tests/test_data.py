import numpy as np
import pytest

from daim_fm import generate_synthetic_series, split_series


def test_synthetic_series_is_deterministic() -> None:
    first = generate_synthetic_series(length=24, seed=7)
    second = generate_synthetic_series(length=24, seed=7)
    np.testing.assert_array_equal(first, second)


def test_split_holds_out_requested_horizon() -> None:
    history, held_out = split_series([1, 2, 3, 4, 5], horizon=2)
    np.testing.assert_array_equal(history, [1, 2, 3])
    np.testing.assert_array_equal(held_out, [4, 5])


@pytest.mark.parametrize("horizon", [0, -1, 3])
def test_split_rejects_invalid_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        split_series([1, 2, 3], horizon=horizon)


def test_split_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        split_series([], horizon=1)

