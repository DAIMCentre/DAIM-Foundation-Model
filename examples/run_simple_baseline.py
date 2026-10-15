"""Run and record one tiny, CPU-only forecasting experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import yaml

from daim_fm import generate_synthetic_series, last_value_forecast, mae, rmse, split_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(output_dir: Path, seed: int) -> dict[str, object]:
    """Run the example and return its human-readable experiment record."""
    series_length = 48
    horizon = 12
    series = generate_synthetic_series(length=series_length, seed=seed)
    history, actual = split_series(series, horizon=horizon)
    predicted = last_value_forecast(history, horizon=horizon)
    metric_values = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": f"fm-example-naive-seed-{seed}",
        "date": datetime.now(timezone.utc).date().isoformat(),
        "research_question": "How does a transparent naive forecast behave on a small seasonal synthetic series?",
        "dataset": "Synthetic monthly trend-and-seasonality example generated locally",
        "model": "Last-value naive forecast",
        "configuration": {
            "seed": seed,
            "series_length": series_length,
            "training_points": int(history.size),
            "forecast_horizon": horizon,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
        },
        "metrics": metric_values,
        "notes": (
            "Introductory reproducibility example only. The data are synthetic, "
            "and these measurements are not evidence about foundation models."
        ),
    }

    with (output_dir / "experiment.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metric_values, handle, indent=2)
        handle.write("\n")
    with (output_dir / "forecast.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_index", "actual", "forecast"])
        writer.writeheader()
        for offset, (observed, forecast) in enumerate(zip(actual, predicted, strict=True)):
            writer.writerow(
                {
                    "time_index": int(history.size + offset),
                    "actual": float(observed),
                    "forecast": float(forecast),
                }
            )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Synthetic-data random seed (default: 42).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "example",
        help="Directory for experiment.yaml, metrics.json and forecast.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_experiment(args.output_dir, args.seed)
    print(f"Recorded {metadata['experiment_id']} in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

