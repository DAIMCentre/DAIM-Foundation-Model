import pytest

from daim_fm import mae, rmse


def test_metrics_have_expected_values() -> None:
    actual = [1.0, 2.0, 3.0]
    predicted = [2.0, 1.0, 4.0]
    assert mae(actual, predicted) == pytest.approx(1.0)
    assert rmse(actual, predicted) == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [mae, rmse])
def test_metrics_reject_mismatched_lengths(metric) -> None:
    with pytest.raises(ValueError, match="same length"):
        metric([1.0, 2.0], [1.0])


@pytest.mark.parametrize("metric", [mae, rmse])
def test_metrics_reject_empty_inputs(metric) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        metric([], [])

