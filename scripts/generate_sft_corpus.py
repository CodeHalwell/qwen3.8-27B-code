#!/usr/bin/env python3
"""Generate the execution-verified bootstrap SFT corpus.

Run with:
    uv run --group dev python scripts/generate_sft_corpus.py

Every trajectory drives the real six-tool harness (including real pytest
runs), so generation takes a few minutes. Rendered token statistics are added
separately by scripts/validate_dataset_rendering.py using the pinned
tokenizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_8_27b_code.fixtures import VARIANTS_PER_FAMILY
from qwen3_8_27b_code.trajectories import write_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "native_sft" / "trajectories.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "native_sft" / "quality_report.json")
    parser.add_argument("--variants-per-family", type=int, default=VARIANTS_PER_FAMILY)
    arguments = parser.parse_args()

    report = write_corpus(arguments.out, arguments.report, arguments.variants_per_family)
    print(json.dumps(report, indent=2))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
