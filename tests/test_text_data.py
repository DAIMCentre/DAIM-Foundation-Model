from pathlib import Path

import pytest
import torch

from daim_fm.text_data import causal_batch, load_text_tokens


def test_causal_targets_are_inputs_shifted_by_one() -> None:
    tokens = torch.arange(20)
    generator = torch.Generator().manual_seed(7)
    inputs, targets = causal_batch(
        tokens,
        context_length=5,
        batch_size=4,
        generator=generator,
    )
    torch.testing.assert_close(targets[:, :-1], inputs[:, 1:])
    torch.testing.assert_close(targets, inputs + 1)


def test_causal_batches_repeat_with_same_seed() -> None:
    tokens = torch.arange(30)
    first = causal_batch(
        tokens,
        context_length=4,
        batch_size=3,
        generator=torch.Generator().manual_seed(11),
    )
    second = causal_batch(
        tokens,
        context_length=4,
        batch_size=3,
        generator=torch.Generator().manual_seed(11),
    )
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])


def test_text_data_must_be_long_enough(tmp_path: Path) -> None:
    data = tmp_path / "tiny.txt"
    data.write_text("abc", encoding="utf-8")
    tokens = load_text_tokens(data)
    with pytest.raises(ValueError, match="at least 5"):
        causal_batch(tokens, context_length=4, batch_size=1)
