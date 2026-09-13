#!/usr/bin/env python3
"""Score a policy on the held-out suite, or compare two scored runs.

Run a policy and write its report:

    uv run --group dev python scripts/evaluate_agent.py run \
        --policy gold --label candidate --out reports/candidate.json

Compare a candidate against a frozen baseline and apply the gate:

    uv run --group dev python scripts/evaluate_agent.py compare \
        reports/baseline.json reports/candidate.json

Tabulate the effort ladder from three reports of the same policy and pick
the cheapest effort that keeps the best success (docs/thinking-budget.md):

    uv run --group dev python scripts/evaluate_agent.py ladder \
        low=reports/ladder_low.json medium=reports/ladder_medium.json \
        xhigh=reports/ladder_xhigh.json --out reports/effort_ladder.json

The gate thresholds come from docs/evaluation.md and should be frozen before a
candidate's results are looked at. Exit status is 1 when the gate fails and 2
when the two reports were not measured the same way (different effort, caps,
budget or attempt count, or the same model on both sides), so this can sit in
front of a promotion step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_8_27b_code.episodes import EpisodeBudget  # noqa: E402
from qwen3_8_27b_code.evaluation import (  # noqa: E402
    DEFAULT_MAX_REASONING_GROWTH,
    build_provenance,
    compare,
    effort_ladder,
    evaluate,
    gate,
    gate_passed,
    pairing_problems,
    read_report,
    write_report,
)
from qwen3_8_27b_code.policies import load_policy_factory  # noqa: E402
from qwen3_8_27b_code.tasks import EVALUATION_VARIANTS_PER_FAMILY, evaluation_tasks  # noqa: E402


def harness_revision() -> str:
    """The commit this harness ran at, or ``unknown`` outside a checkout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def run(arguments: argparse.Namespace) -> int:
    tasks = evaluation_tasks(variants_per_family=arguments.variants_per_family)
    budget = EpisodeBudget(tool_calls=arguments.tool_calls, wall_seconds=arguments.wall_seconds)
    report = evaluate(
        tasks,
        load_policy_factory(arguments.policy),
        label=arguments.label,
        attempts_per_task=arguments.attempts,
        seeds=tuple(arguments.seeds),
        budget=budget,
    )
    # The same record notebook 07 writes, so `compare` can refuse to pair
    # this report with one measured differently.
    report.metadata = build_provenance(
        model=arguments.model or arguments.policy,
        harness_revision=harness_revision(),
        reasoning_effort=arguments.reasoning_effort,
        max_new_tokens=arguments.max_new_tokens,
        max_sequence_length=arguments.max_sequence_length,
        episode_budget=budget,
        attempts_per_task=arguments.attempts,
        variants_per_family=arguments.variants_per_family,
    )
    write_report(report, arguments.out)
    print(json.dumps(report.scorecard(), indent=2))
    print(f"wrote {arguments.out}")
    return 0


def run_compare(arguments: argparse.Namespace) -> int:
    baseline = read_report(arguments.baseline)
    candidate = read_report(arguments.candidate)
    blocking, advisory = pairing_problems(baseline, candidate)
    if blocking:
        for problem in blocking:
            print(f"  [REFUSED] {problem}")
        print("GATE NOT RUN: the two reports were not measured the same way.")
        return 2
    for note in advisory:
        print(f"  [NOTE] {note}")
    comparison = compare(baseline, candidate)
    comparison["provenance"] = {
        "baseline": {**baseline.metadata, "path": str(arguments.baseline)},
        "candidate": {**candidate.metadata, "path": str(arguments.candidate)},
        "notes": advisory,
    }
    checks = gate(
        comparison,
        minimum_success_delta=arguments.minimum_success_delta,
        max_reasoning_growth=None if arguments.ignore_thinking_budget else arguments.max_reasoning_growth,
    )
    comparison["gate"] = [
        {"name": check.name, "passed": check.passed, "detail": check.detail} for check in checks
    ]
    comparison["gate_passed"] = gate_passed(checks)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(comparison, indent=2) + "\n")

    task_level = comparison["task_level"]
    print(json.dumps(
        {"deltas": comparison["deltas"], "thinking": comparison["thinking"], "task_level": task_level},
        indent=2,
    ))
    for check in checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    # A small suite reports paired outcomes rather than implying significance.
    print(
        f"\n{task_level['wins']} tasks improved, {task_level['losses']} regressed, "
        f"{task_level['ties']} unchanged, across {task_level['tasks']} tasks."
    )
    print("GATE PASSED" if comparison["gate_passed"] else "GATE FAILED")
    return 0 if comparison["gate_passed"] else 1


def run_ladder(arguments: argparse.Namespace) -> int:
    reports = {}
    for item in arguments.reports:
        effort, separator, path = item.partition("=")
        if not separator:
            raise SystemExit(f"expected effort=path, got {item!r}")
        reports[effort] = read_report(Path(path))
    ladder = effort_ladder(reports, success_tolerance=arguments.success_tolerance)
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(ladder, indent=2) + "\n")
    print(json.dumps(ladder, indent=2))
    if ladder["recommended"] is None:
        print(f"\n{ladder['note']}.")
        return 1
    print(
        f"\nRecommended deployment effort: {ladder['recommended']} "
        f"(best success {ladder['best_success']:.4f} in aggregate and per band within "
        f"{ladder['success_tolerance']:.4f}, measured in {ladder['unit']})."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runner = subparsers.add_parser("run", help="score one policy on the held-out suite")
    runner.add_argument("--policy", default="gold", help="built-in name or module:attribute")
    runner.add_argument("--label", default="candidate")
    runner.add_argument("--variants-per-family", type=int, default=EVALUATION_VARIANTS_PER_FAMILY)
    runner.add_argument("--attempts", type=int, default=1, help="attempts per task")
    runner.add_argument("--seeds", type=int, nargs="+", default=[3407, 9176, 20261])
    runner.add_argument("--tool-calls", type=int, default=30)
    runner.add_argument("--wall-seconds", type=float, default=900.0)
    runner.add_argument(
        "--model",
        default=None,
        help="what the policy measures, recorded as provenance (a Hub id, adapter@revision, ...); "
        "defaults to the --policy reference",
    )
    runner.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    runner.add_argument("--max-new-tokens", type=int, default=None, help="generation cap the policy ran with")
    runner.add_argument("--max-sequence-length", type=int, default=None, help="context window the policy ran with")
    runner.add_argument("--out", type=Path, default=ROOT / "reports" / "evaluation.json")
    runner.set_defaults(handler=run)

    comparer = subparsers.add_parser("compare", help="apply the gate to two reports")
    comparer.add_argument("baseline", type=Path)
    comparer.add_argument("candidate", type=Path)
    comparer.add_argument("--minimum-success-delta", type=float, default=0.0)
    comparer.add_argument(
        "--max-reasoning-growth",
        type=float,
        default=DEFAULT_MAX_REASONING_GROWTH,
        help="thinking-budget ceiling: candidate reasoning tokens per turn may exceed the "
        "baseline by at most this fraction (default 0.10); only checked when both reports "
        "counted reasoning tokens",
    )
    comparer.add_argument(
        "--ignore-thinking-budget",
        action="store_true",
        help="drop the thinking-budget check from the gate",
    )
    comparer.add_argument("--out", type=Path, default=None)
    comparer.set_defaults(handler=run_compare)

    ladder = subparsers.add_parser("ladder", help="tabulate low/medium/xhigh reports and pick an effort")
    ladder.add_argument("reports", nargs="+", help="effort=path, one per rung")
    ladder.add_argument(
        "--success-tolerance",
        type=float,
        default=0.0,
        help="a rung is eligible when its success is within this much of the best rung",
    )
    ladder.add_argument("--out", type=Path, default=None)
    ladder.set_defaults(handler=run_ladder)

    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
