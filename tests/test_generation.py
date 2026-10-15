import pytest
import torch

from daim_fm import LMConfig, TransformerLM, generate


def test_generation_appends_requested_number_of_valid_tokens() -> None:
    config = LMConfig(256, 8, 16, 4, 1, 32)
    model = TransformerLM(config)
    result = generate(model, torch.tensor([[65, 66]]), max_new_tokens=4)
    assert result.shape == (1, 6)
    assert torch.all((result >= 0) & (result < 256))


def test_generation_rejects_empty_prompt() -> None:
    model = TransformerLM(LMConfig(256, 8, 16, 4, 1, 32))
    with pytest.raises(ValueError, match="must not be empty"):
        generate(model, torch.empty((1, 0), dtype=torch.long), max_new_tokens=1)
