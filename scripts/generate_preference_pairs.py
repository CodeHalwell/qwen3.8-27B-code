#!/usr/bin/env python3
"""Generate execution-derived preference pairs for the DPO stage.

Run with:
    uv run --group dev python scripts/generate_preference_pairs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_8_27b_code.preferences import PAIRS_PER_FAMILY, write_pairs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "preferences" / "pairs.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "preferences" / "quality_report.json")
    parser.add_argument("--pairs-per-family", type=int, default=PAIRS_PER_FAMILY)
    arguments = parser.parse_args()

    report = write_pairs(arguments.out, arguments.report, arguments.pairs_per_family)
    print(json.dumps(report, indent=2))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
