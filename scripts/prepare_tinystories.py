"""Download small, reproducible TinyStories V2 GPT-4 subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from daim_fm.tinystories import DataPreparationError, prepare_tinystories


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-stories", type=int, default=1000)
    parser.add_argument("--valid-stories", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tinystories_v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        manifest = prepare_tinystories(
            args.output_dir,
            train_stories=args.train_stories,
            valid_stories=args.valid_stories,
            chunk_size=args.chunk_size,
        )
    except (ValueError, DataPreparationError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Prepared TinyStories subsets and manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
