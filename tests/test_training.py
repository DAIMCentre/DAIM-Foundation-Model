import torch

from daim_fm import LMConfig, TrainingConfig, TransformerLM, train_language_model
from daim_fm.training import set_seed


def test_small_model_loss_decreases_on_repeated_sequence() -> None:
    config = LMConfig(256, 8, 24, 4, 1, 48)
    tokens = torch.tensor(list(range(32)) * 8, dtype=torch.long)
    set_seed(5)
    model = TransformerLM(config)
    measurements = train_language_model(
        model,
        tokens,
        tokens,
        config=TrainingConfig(
            batch_size=4,
            steps=30,
            learning_rate=0.01,
            eval_interval=10,
            eval_batches=1,
        ),
        seed=5,
    )
    assert measurements[-1]["training_loss"] < measurements[0]["training_loss"]
    assert "validation_loss" in measurements[-1]
