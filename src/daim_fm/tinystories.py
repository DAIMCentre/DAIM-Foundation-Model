"""Prepare small, provenance-recorded TinyStories V2 GPT-4 subsets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from .text_data import sha256_file


BOUNDARY = b"<|endoftext|>"
DATASET_NAME = "roneneldan/TinyStories"
DATASET_VERSION = "V2-GPT4"
DATASET_LICENSE = "CDLA-Sharing-1.0"
SOURCE_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


class DataPreparationError(RuntimeError):
    """Raised when a complete, verifiable subset cannot be prepared."""


RangeFetcher = Callable[[int, int], bytes]


def select_story_bytes(
    fetch_range: RangeFetcher,
    *,
    story_count: int,
    chunk_size: int = 1024 * 1024,
    max_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    """Select complete boundary-to-boundary stories from byte-range chunks.

    Content preceding the first observed boundary is discarded. This makes the
    rule safe even if a future source starts a range inside a story.
    """
    if not isinstance(story_count, int) or isinstance(story_count, bool) or story_count <= 0:
        raise ValueError("story_count must be a positive integer")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    collected = bytearray()
    offset = 0
    while len(collected) < max_bytes:
        chunk = fetch_range(offset, offset + chunk_size - 1)
        if not chunk:
            break
        collected.extend(chunk)
        offset += len(chunk)

        positions: list[int] = []
        search_from = 0
        while True:
            found = collected.find(BOUNDARY, search_from)
            if found < 0:
                break
            positions.append(found)
            search_from = found + len(BOUNDARY)
        if len(positions) >= story_count + 1:
            start = positions[0] + len(BOUNDARY)
            end = positions[story_count] + len(BOUNDARY)
            selected = bytes(collected[start:end])
            try:
                selected.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DataPreparationError("selected TinyStories bytes are not valid UTF-8") from exc
            return selected
        if len(chunk) < chunk_size:
            break
    raise DataPreparationError(
        f"source ended before {story_count} complete stories were available after the first boundary"
    )


def http_range_fetcher(url: str) -> RangeFetcher:
    """Create a fetcher that refuses servers without byte-range support."""
    def fetch(start: int, end: int) -> bytes:
        request = Request(
            url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "daim-foundation-models/0.2 data preparation",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                status = getattr(response, "status", response.getcode())
                if status != 206:
                    raise DataPreparationError(
                        f"source did not honour byte ranges (HTTP {status}): {url}"
                    )
                return response.read(end - start + 1)
        except DataPreparationError:
            raise
        except OSError as exc:
            raise DataPreparationError(f"could not retrieve {url}: {exc}") from exc

    return fetch


def prepare_tinystories(
    output_dir: str | Path,
    *,
    train_stories: int = 1000,
    valid_stories: int = 100,
    chunk_size: int = 1024 * 1024,
    fetcher_factory: Callable[[str], RangeFetcher] = http_range_fetcher,
) -> Path:
    """Write deterministic train/validation subsets and their manifest."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    requested = {"train": train_stories, "valid": valid_stories}
    source_records: dict[str, dict[str, Any]] = {}

    for split in ("train", "valid"):
        data = select_story_bytes(
            fetcher_factory(SOURCE_URLS[split]),
            story_count=requested[split],
            chunk_size=chunk_size,
        )
        path = destination / f"{split}.txt"
        path.write_bytes(data)
        source_records[split] = {
            "url": SOURCE_URLS[split],
            "file": path.name,
            "requested_stories": requested[split],
            "observed_stories": data.count(BOUNDARY),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest = {
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "license": DATASET_LICENSE,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Discard through the first end-of-text boundary, then preserve the requested "
            "number of complete boundary-to-boundary story spans."
        ),
        "boundary": BOUNDARY.decode("ascii"),
        "sources": source_records,
    }
    manifest_path = destination / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return manifest_path


def validate_prepared_tinystories(manifest_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    """Validate all manifested files before a training run begins."""
    path = Path(manifest_path)
    try:
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DataPreparationError(
            f"prepared TinyStories data is missing; run: python scripts/prepare_tinystories.py"
        ) from exc
    except yaml.YAMLError as exc:
        raise DataPreparationError(f"malformed TinyStories manifest: {path}") from exc
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("sources"), Mapping):
        raise DataPreparationError("TinyStories manifest must contain a sources mapping")

    resolved: dict[str, Path] = {}
    for split in ("train", "valid"):
        record = manifest["sources"].get(split)
        if not isinstance(record, Mapping) or not isinstance(record.get("file"), str):
            raise DataPreparationError(f"manifest is missing the {split} file record")
        data_path = path.parent / record["file"]
        if not data_path.is_file():
            raise DataPreparationError(f"manifested {split} data does not exist: {data_path}")
        actual = sha256_file(data_path)
        if actual != record.get("sha256"):
            raise DataPreparationError(f"SHA-256 mismatch for {split} data: {data_path}")
        resolved[split] = data_path
    return resolved["train"], resolved["valid"], dict(manifest)
