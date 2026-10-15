"""Inspect the approximately 0.1B model without allocating or training weights."""

from __future__ import annotations

from dataclasses import asdict
import json

from daim_fm import TransformerLM, get_preset


def main() -> None:
    config = get_preset("0.1b")
    model = TransformerLM(config, device="meta")
    print(json.dumps({"configuration": asdict(config), "parameter_count": model.parameter_count}, indent=2))
    print("Inspection only: no weights were allocated, trained or downloaded.")


if __name__ == "__main__":
    main()
