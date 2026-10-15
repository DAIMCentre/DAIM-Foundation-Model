from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_example_records_complete_result(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "examples" / "run_simple_baseline.py"),
            "--seed",
            "9",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "fm-example-naive-seed-9" in completed.stdout
    assert {path.name for path in tmp_path.iterdir()} == {
        "experiment.yaml",
        "forecast.csv",
        "metrics.json",
    }

    metadata = yaml.safe_load((tmp_path / "experiment.yaml").read_text(encoding="utf-8"))
    assert set(metadata) == {
        "experiment_id",
        "date",
        "research_question",
        "dataset",
        "model",
        "configuration",
        "metrics",
        "notes",
    }
    assert metadata["configuration"]["seed"] == 9

    metric_values = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert set(metric_values) == {"mae", "rmse"}

