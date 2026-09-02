#!/usr/bin/env python3
"""Score a policy on the held-out suite, or compare two scored runs.

Run a policy and write its report:

    uv run --group dev python scripts/evaluate_agent.py run \
        --policy gold --label candidate --out reports/candidate.json

Compare a candidate against a frozen baseline and apply the gate:

    uv run --group dev python scripts/evaluate_agent.py compare \
        reports/baseline.json reports/candidate.json

The gate thresholds come from docs/evaluation.md and should be frozen before a
candidate's results are looked at. Exit status is non-zero when the gate fails,
so this can sit in front of a promotion step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_8_27b_code.episodes import EpisodeBudget  # noqa: E402
from qwen3_8_27b_code.evaluation import (  # noqa: E402
    compare,
    evaluate,
    gate,
    gate_passed,
    read_report,
    write_report,
)
from qwen3_8_27b_code.policies import load_policy_factory  # noqa: E402
from qwen3_8_27b_code.tasks import EVALUATION_VARIANTS_PER_FAMILY, evaluation_tasks  # noqa: E402


def run(arguments: argparse.Namespace) -> int:
    tasks = evaluation_tasks(variants_per_family=arguments.variants_per_family)
    report = evaluate(
        tasks,
        load_policy_factory(arguments.policy),
        label=arguments.label,
        attempts_per_task=arguments.attempts,
        seeds=tuple(arguments.seeds),
        budget=EpisodeBudget(tool_calls=arguments.tool_calls, wall_seconds=arguments.wall_seconds),
    )
    write_report(report, arguments.out)
    print(json.dumps(report.scorecard(), indent=2))
    print(f"wrote {arguments.out}")
    return 0


def run_compare(arguments: argparse.Namespace) -> int:
    comparison = compare(read_report(arguments.baseline), read_report(arguments.candidate))
    checks = gate(comparison, minimum_success_delta=arguments.minimum_success_delta)
    comparison["gate"] = [
        {"name": check.name, "passed": check.passed, "detail": check.detail} for check in checks
    ]
    comparison["gate_passed"] = gate_passed(checks)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(comparison, indent=2) + "\n")

    task_level = comparison["task_level"]
    print(json.dumps({"deltas": comparison["deltas"], "task_level": task_level}, indent=2))
    for check in checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    # A small suite reports paired outcomes rather than implying significance.
    print(
        f"\n{task_level['wins']} tasks improved, {task_level['losses']} regressed, "
        f"{task_level['ties']} unchanged, across {task_level['tasks']} tasks."
    )
    print("GATE PASSED" if comparison["gate_passed"] else "GATE FAILED")
    return 0 if comparison["gate_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run", help="score one policy on the held-out suite")
    runner.add_argument("--policy", default="gold", help="built-in name or module:attribute")
    runner.add_argument("--label", default="candidate")
    runner.add_argument("--variants-per-family", type=int, default=EVALUATION_VARIANTS_PER_FAMILY)
    runner.add_argument("--attempts", type=int, default=1, help="attempts per task")
    runner.add_argument("--seeds", type=int, nargs="+", default=[3407, 9176, 20261])
    runner.add_argument("--tool-calls", type=int, default=10)
    runner.add_argument("--wall-seconds", type=float, default=480.0)
    runner.add_argument("--out", type=Path, default=ROOT / "reports" / "evaluation.json")
    runner.set_defaults(handler=run)

    comparer = subparsers.add_parser("compare", help="apply the gate to two reports")
    comparer.add_argument("baseline", type=Path)
    comparer.add_argument("candidate", type=Path)
    comparer.add_argument("--minimum-success-delta", type=float, default=0.0)
    comparer.add_argument("--out", type=Path, default=None)
    comparer.set_defaults(handler=run_compare)

    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
