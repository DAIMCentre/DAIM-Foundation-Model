from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

from daim_fm import LMConfig, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_example_module():
    path = PROJECT_ROOT / "examples" / "run_lm_from_scratch.py"
    spec = importlib.util.spec_from_file_location("run_lm_from_scratch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lm_example_writes_complete_research_record(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "tests" / "fixtures" / "micro_corpus.txt"
    output = tmp_path / "output"
    module = load_example_module()
    record = module.run_experiment(
        train_path=source,
        validation_path=source,
        output_dir=output,
        model_config=LMConfig(256, 8, 16, 4, 1, 32),
        training_config=TrainingConfig(
            batch_size=2,
            steps=2,
            eval_interval=1,
            eval_batches=1,
        ),
        seed=9,
        max_new_tokens=2,
    )
    assert {path.name for path in output.iterdir()} == {"run.yaml", "metrics.jsonl", "sample.txt"}
    saved = yaml.safe_load((output / "run.yaml").read_text(encoding="utf-8"))
    assert saved["seed"] == 9
    assert saved["data"]["train_sha256"] == saved["data"]["validation_sha256"]
    assert saved["status"] == "demonstration_not_scientific_finding"
    lines = (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all("training_loss" in json.loads(line) for line in lines)
    assert record["model"]["parameter_count"] > 0
