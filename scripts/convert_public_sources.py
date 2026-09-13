#!/usr/bin/env python3
"""Stream public Hub datasets and write native-schema rows, on a CPU.

    uv run --group dev python scripts/convert_public_sources.py \
        --source nvidia/OpenCodeInstruct --limit 2000 \
        --out data/public_native/opencodeinstruct.jsonl

Notebook 02 does the same conversion in-line with the model's tokenizer
counting tokens; this script uses a character estimate and is for looking
at what a source yields before spending Colab time on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_8_27b_code.public_sources import SOURCE_LOADERS, collect_public_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCE_LOADERS), required=True)
    parser.add_argument("--limit", type=int, default=500, help="native rows to produce")
    parser.add_argument("--budget-tokens", type=int, default=6_000, help="longest row kept, estimated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args()

    rows, report = collect_public_rows({arguments.source: arguments.limit}, arguments.budget_tokens)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report_path = arguments.report or arguments.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {len(rows)} rows to {arguments.out}")


if __name__ == "__main__":
    main()
