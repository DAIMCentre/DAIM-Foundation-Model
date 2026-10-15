from pathlib import Path

import pytest
import yaml

from daim_fm.tinystories import (
    BOUNDARY,
    DataPreparationError,
    prepare_tinystories,
    select_story_bytes,
    validate_prepared_tinystories,
)


def local_fetcher(data: bytes):
    def fetch(start: int, end: int) -> bytes:
        return data[start : end + 1]

    return fetch


def test_story_selection_discards_prefix_and_keeps_complete_spans() -> None:
    payload = b"discarded" + BOUNDARY + b"\nOne." + BOUNDARY + b"\nTwo." + BOUNDARY + b"tail"
    selected = select_story_bytes(local_fetcher(payload), story_count=2, chunk_size=7)
    assert selected == b"\nOne." + BOUNDARY + b"\nTwo." + BOUNDARY


def test_preparation_writes_verifiable_manifest_without_network(tmp_path: Path) -> None:
    payload = b"prefix" + BOUNDARY + b"\nOne." + BOUNDARY + b"\nTwo." + BOUNDARY

    def factory(_url: str):
        return local_fetcher(payload)

    manifest_path = prepare_tinystories(
        tmp_path,
        train_stories=2,
        valid_stories=1,
        chunk_size=5,
        fetcher_factory=factory,
    )
    train, valid, manifest = validate_prepared_tinystories(manifest_path)
    assert train.is_file() and valid.is_file()
    assert manifest["sources"]["train"]["observed_stories"] == 2
    assert manifest["sources"]["valid"]["observed_stories"] == 1
    assert len(manifest["sources"]["train"]["sha256"]) == 64

    train.write_text("changed", encoding="utf-8")
    with pytest.raises(DataPreparationError, match="SHA-256 mismatch"):
        validate_prepared_tinystories(manifest_path)


def test_incomplete_source_fails_clearly() -> None:
    with pytest.raises(DataPreparationError, match="before 2 complete stories"):
        select_story_bytes(local_fetcher(b"prefix" + BOUNDARY + b"only one"), story_count=2)
