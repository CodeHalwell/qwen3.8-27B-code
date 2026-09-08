#!/usr/bin/env python3
"""Collect execution-verified native trajectories by rejection sampling.

Every attempt is graded from outside the model's workspace and only verified
attempts become rows, so the output is a drop-in source for notebook 02.

Smoke run with the scripted gold policy (no GPU needed):

    uv run --group dev python scripts/collect_trajectories.py \
        --policy gold --suite evaluation --attempts 1 --out /tmp/rows.jsonl

Real collection supplies a model-backed policy as `module:attribute`, where the
attribute is a callable taking ``(task, seed)`` and returning a policy:

    uv run --group dev python scripts/collect_trajectories.py \
        --policy my_policies:unsloth_policy --suite training --attempts 3

The training suite is the single-file fixtures plus the multi-file families in
``long_horizon``. With more than one attempt per task the rows kept are the
verified attempts that reasoned least, and the verified attempts that were not
kept still contribute reasoning-length preference pairs (docs/thinking-budget.md),
written next to the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_8_27b_code.collection import SELECTIONS, collect, write_corpus  # noqa: E402
from qwen3_8_27b_code.episodes import EpisodeBudget  # noqa: E402
from qwen3_8_27b_code.fixtures import iter_tasks  # noqa: E402
from qwen3_8_27b_code.long_horizon import training_tasks as long_horizon_training_tasks  # noqa: E402
from qwen3_8_27b_code.policies import load_policy_factory  # noqa: E402
from qwen3_8_27b_code.tasks import evaluation_tasks, task_from_fixture  # noqa: E402
from qwen3_8_27b_code.thinking import (  # noqa: E402
    DEFAULT_MAX_PAIRS_PER_TASK,
    DEFAULT_MIN_LENGTH_RATIO,
    build_reasoning_length_pairs,
    write_length_pairs,
)


def build_tasks(suite: str, variants: int) -> list:
    if suite == "evaluation":
        # Held-out by design. Useful for a smoke run; never collect training
        # data from the suite the gate is measured on.
        return evaluation_tasks(variants_per_family=variants)
    single_file = [task_from_fixture(fixture) for fixture in iter_tasks(variants)]
    return single_file + long_horizon_training_tasks(variants)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", default="gold", help="built-in name or module:attribute")
    parser.add_argument("--suite", choices=("training", "evaluation"), default="training")
    parser.add_argument("--variants-per-family", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3, help="attempts per task")
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 9176, 20261])
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--tool-calls", type=int, default=10, help="per-episode tool-call budget")
    parser.add_argument("--wall-seconds", type=float, default=480.0)
    parser.add_argument("--max-rows-per-task", type=int, default=None)
    parser.add_argument(
        "--selection",
        choices=SELECTIONS,
        default="shortest_reasoning",
        help="which verified attempts become rows under the per-task cap",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "collected" / "trajectories.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "collected" / "quality_report.json")
    parser.add_argument(
        "--length-pairs-out",
        type=Path,
        default=ROOT / "data" / "collected" / "length_pairs.jsonl",
        help="reasoning-length preference pairs built from the verified attempts",
    )
    parser.add_argument(
        "--length-pairs-report",
        type=Path,
        default=ROOT / "data" / "collected" / "length_pairs_report.json",
    )
    parser.add_argument("--min-length-ratio", type=float, default=DEFAULT_MIN_LENGTH_RATIO)
    parser.add_argument("--max-pairs-per-task", type=int, default=DEFAULT_MAX_PAIRS_PER_TASK)
    arguments = parser.parse_args()

    if arguments.suite == "evaluation":
        print(
            "warning: collecting from the held-out evaluation suite. Use it for smoke runs "
            "only; training on it invalidates the gate.",
            file=sys.stderr,
        )

    tasks = build_tasks(arguments.suite, arguments.variants_per_family)
    result = collect(
        tasks,
        load_policy_factory(arguments.policy),
        attempts_per_task=arguments.attempts,
        seeds=tuple(arguments.seeds),
        budget=EpisodeBudget(tool_calls=arguments.tool_calls, wall_seconds=arguments.wall_seconds),
        reasoning_effort=arguments.reasoning_effort,
        max_rows_per_task=arguments.max_rows_per_task,
        selection=arguments.selection,
    )
    report = write_corpus(result, arguments.out, arguments.report)
    report["policy"] = arguments.policy
    report["suite"] = arguments.suite
    arguments.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {arguments.out} ({len(result.rows)} rows)")

    pairs = build_reasoning_length_pairs(
        result.attempts,
        min_ratio=arguments.min_length_ratio,
        max_pairs_per_task=arguments.max_pairs_per_task,
    )
    pairs_report = write_length_pairs(pairs, arguments.length_pairs_out, arguments.length_pairs_report)
    pairs_report["policy"] = arguments.policy
    arguments.length_pairs_report.write_text(json.dumps(pairs_report, indent=2) + "\n")
    print(f"wrote {arguments.length_pairs_out} ({len(pairs)} reasoning-length pairs)")


if __name__ == "__main__":
    main()
