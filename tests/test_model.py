import torch

from daim_fm import LMConfig, TransformerLM, get_preset


def small_config() -> LMConfig:
    return LMConfig(
        vocab_size=256,
        context_length=8,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
    )


def test_model_returns_logits_and_finite_loss() -> None:
    model = TransformerLM(small_config())
    tokens = torch.randint(0, 256, (2, 8))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 8, 256)
    assert loss is not None and torch.isfinite(loss)


def test_causal_mask_prevents_future_token_leakage() -> None:
    torch.manual_seed(3)
    model = TransformerLM(small_config()).eval()
    original = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = original.clone()
    changed[0, 4:] = torch.tensor([90, 91])
    with torch.no_grad():
        first, _ = model(original)
        second, _ = model(changed)
    torch.testing.assert_close(first[:, :4], second[:, :4])


def test_named_presets_have_documented_exact_parameter_counts() -> None:
    tiny = TransformerLM(get_preset("tiny"), device="meta")
    larger = TransformerLM(get_preset("0.1b"), device="meta")
    assert tiny.parameter_count == 2_732_928
    assert larger.parameter_count == 97_949_184
