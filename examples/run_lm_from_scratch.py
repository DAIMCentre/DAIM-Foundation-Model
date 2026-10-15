"""Train, evaluate and sample from the small from-scratch language model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import torch
import yaml

from daim_fm import ByteTokenizer, LMConfig, TrainingConfig, TransformerLM, generate, get_preset
from daim_fm.text_data import load_text_tokens, sha256_file
from daim_fm.tinystories import validate_prepared_tinystories
from daim_fm.training import set_seed, train_language_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tinystories_v2"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def run_experiment(
    *,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    model_config: LMConfig,
    training_config: TrainingConfig,
    seed: int,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 80,
    manifest_path: Path | None = None,
    manifest: dict[str, Any] | None = None,
    save_checkpoint: bool = False,
) -> dict[str, Any]:
    """Run one recorded experiment; callers may supply tiny local test files."""
    tokenizer = ByteTokenizer()
    train_tokens = load_text_tokens(train_path, tokenizer)
    validation_tokens = load_text_tokens(validation_path, tokenizer)
    set_seed(seed)
    model = TransformerLM(model_config)
    measurements = train_language_model(
        model,
        train_tokens,
        validation_tokens,
        config=training_config,
        seed=seed,
    )

    prompt_tokens = tokenizer.encode(prompt)
    generated = generate(
        model,
        torch.tensor([prompt_tokens], dtype=torch.long),
        max_new_tokens=max_new_tokens,
        temperature=0.9,
        top_k=40,
        sample=True,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    generated_text = tokenizer.decode(generated[0].tolist(), errors="replace")

    now = datetime.now(timezone.utc)
    run_id = f"lm-tiny-seed-{seed}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "research_question": (
            "Can the complete next-token training path run reproducibly on a small "
            "TinyStories subset?"
        ),
        "status": "demonstration_not_scientific_finding",
        "seed": seed,
        "device": training_config.device,
        "model": {
            "architecture": "decoder-only causal Transformer",
            "parameter_count": model.parameter_count,
            "configuration": asdict(model_config),
        },
        "training": asdict(training_config),
        "data": {
            "dataset": (manifest or {}).get("dataset", "explicit local UTF-8 files"),
            "version": (manifest or {}).get("version"),
            "manifest": str(manifest_path.resolve()) if manifest_path else None,
            "train_file": str(train_path.resolve()),
            "train_sha256": sha256_file(train_path),
            "validation_file": str(validation_path.resolve()),
            "validation_sha256": sha256_file(validation_path),
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "package": _version("daim-foundation-models"),
        },
        "outputs": {
            "metrics": "metrics.jsonl",
            "sample": "sample.txt",
            "checkpoint": "checkpoint.pt" if save_checkpoint else None,
        },
        "note": (
            "This run demonstrates mechanics and provenance. It is not evidence about "
            "full-data language-model or foundation-model performance."
        ),
    }

    (output_dir / "run.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for measurement in measurements:
            handle.write(json.dumps(measurement, sort_keys=True) + "\n")
    (output_dir / "sample.txt").write_text(
        f"Prompt:\n{prompt}\n\nGenerated continuation:\n{generated_text}\n",
        encoding="utf-8",
    )
    if save_checkpoint:
        torch.save(model.state_dict(), output_dir / "checkpoint.pt")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "lm-example")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.train_data is None) != (args.valid_data is None):
        raise SystemExit("error: --train-data and --valid-data must be supplied together")
    if args.train_data is None:
        train_path, validation_path, manifest = validate_prepared_tinystories(args.manifest)
        manifest_path: Path | None = args.manifest
    else:
        train_path, validation_path = args.train_data, args.valid_data
        manifest, manifest_path = None, None

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        eval_interval=min(10, args.steps),
        eval_batches=2,
    )
    record = run_experiment(
        train_path=train_path,
        validation_path=validation_path,
        output_dir=args.output_dir,
        model_config=get_preset("tiny"),
        training_config=training_config,
        seed=args.seed,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        manifest_path=manifest_path,
        manifest=manifest,
        save_checkpoint=args.save_checkpoint,
    )
    print(f"Recorded {record['run_id']} in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
