#!/usr/bin/env python3
"""Distil a larger open model into student data through the six-tool harness.

The teacher (a bigger Qwen3.8, Kimi K3, GLM 5.3 or anything behind an
OpenAI-compatible endpoint) attempts the training tasks through the exact
deployment tool schema. Every attempt is graded from outside the workspace;
only verified attempts become SFT rows, the verified attempts that thought
more than another become reasoning-length pairs, and every attempt, kept or
not, is persisted so outcome pairs can be built against the student's own
attempts. See docs/distillation.md.

Probe first: one round trip that says whether the endpoint returns native tool
calls and exposes its reasoning.

    uv run --group dev python scripts/collect_from_teacher.py \\
        --preset hf-inference-providers --model <hub-model-id> --probe

Then collect (cost first: attempts x tasks x the teacher's per-episode price):

    uv run --group dev python scripts/collect_from_teacher.py \\
        --preset moonshot --model <kimi-model-id> --attempts 3 \\
        --student-attempts data/collected/attempts.jsonl

Vendor endpoints move; --base-url and --api-key-env override any preset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen3_8_27b_code.collection import (  # noqa: E402
    SELECTIONS,
    collect,
    read_attempts,
    write_attempts,
    write_corpus,
)
from qwen3_8_27b_code.distillation import build_outcome_pairs, write_outcome_pairs  # noqa: E402
from qwen3_8_27b_code.episodes import EpisodeBudget  # noqa: E402
from qwen3_8_27b_code.fixtures import iter_tasks  # noqa: E402
from qwen3_8_27b_code.long_horizon import training_tasks  # noqa: E402
from qwen3_8_27b_code.tasks import task_from_fixture  # noqa: E402
from qwen3_8_27b_code.teachers import (  # noqa: E402
    PRESETS,
    TeacherConfig,
    probe_teacher,
    teacher_policy_factory,
)
from qwen3_8_27b_code.thinking import build_reasoning_length_pairs, write_length_pairs  # noqa: E402


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-").lower()


def build_config(arguments: argparse.Namespace) -> TeacherConfig:
    overrides = {
        "temperature": arguments.temperature,
        "top_p": arguments.top_p,
        "max_tokens": arguments.max_tokens,
        "timeout_seconds": arguments.timeout,
        "max_retries": arguments.max_retries,
        "require_reasoning": not arguments.no_require_reasoning,
        "send_reasoning_back": not arguments.no_send_reasoning_back,
        "reasoning_effort": arguments.reasoning_effort,
        "extra_body": json.loads(arguments.extra_body) if arguments.extra_body else {},
    }
    if arguments.base_url:
        return TeacherConfig(
            model=arguments.model,
            base_url=arguments.base_url,
            api_key_env=arguments.api_key_env,
            **overrides,
        )
    config = TeacherConfig.from_preset(arguments.preset, arguments.model, **overrides)
    if arguments.api_key_env is not None:
        config = TeacherConfig(**{**config.__dict__, "api_key_env": arguments.api_key_env})
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model id as the endpoint names it")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="hf-inference-providers")
    parser.add_argument("--base-url", default=None, help="override the preset's endpoint")
    parser.add_argument("--api-key-env", default=None, help="environment variable holding the API key")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=4096, help="per-turn completion cap at the teacher")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-require-reasoning", action="store_true", help="accept turns whose reasoning the endpoint hides")
    parser.add_argument("--no-send-reasoning-back", action="store_true", help="for strict validators that reject reasoning_content in input")
    parser.add_argument("--reasoning-effort", default=None, help="sent as the API's reasoning_effort parameter when set")
    parser.add_argument("--extra-body", default=None, help="JSON merged into every request body")
    parser.add_argument(
        "--effort-label",
        choices=("low", "medium", "xhigh"),
        default="medium",
        help="reasoning_effort label stored on the rows; see docs/distillation.md before choosing",
    )
    parser.add_argument("--probe", action="store_true", help="one round trip, then exit")
    parser.add_argument("--suite", choices=("training", "evaluation"), default="training")
    parser.add_argument("--variants-per-family", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3407, 9176, 20261])
    parser.add_argument("--tool-calls", type=int, default=10)
    parser.add_argument("--wall-seconds", type=float, default=900.0)
    parser.add_argument("--max-rows-per-task", type=int, default=2)
    parser.add_argument("--selection", choices=SELECTIONS, default="shortest_reasoning")
    parser.add_argument(
        "--student-attempts",
        type=Path,
        action="append",
        default=[],
        help="attempts.jsonl from a student collection; repeatable. Enables teacher-versus-student outcome pairs",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="default data/collected/teacher-<model>")
    arguments = parser.parse_args()

    config = build_config(arguments)
    if arguments.probe:
        report = probe_teacher(config)
        print(json.dumps(report, indent=2))
        return 0 if report["usable"] else 1

    if arguments.suite == "evaluation":
        print(
            "warning: collecting from the held-out evaluation suite. Use it for smoke runs "
            "only; training on it invalidates the gate.",
            file=sys.stderr,
        )
        from qwen3_8_27b_code.tasks import evaluation_tasks

        tasks = evaluation_tasks(variants_per_family=arguments.variants_per_family)
    else:
        tasks = [task_from_fixture(fixture) for fixture in iter_tasks(arguments.variants_per_family)]
        tasks += training_tasks(arguments.variants_per_family)

    out_dir = arguments.out_dir or ROOT / "data" / "collected" / f"teacher-{slug(arguments.model)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = collect(
        tasks,
        teacher_policy_factory(config),
        attempts_per_task=arguments.attempts,
        seeds=tuple(arguments.seeds),
        budget=EpisodeBudget(tool_calls=arguments.tool_calls, wall_seconds=arguments.wall_seconds),
        reasoning_effort=arguments.effort_label,
        max_rows_per_task=arguments.max_rows_per_task,
        selection=arguments.selection,
        policy_label=config.label,
        source=config.label,
        provenance={
            "teacher": {
                "model": config.model,
                "endpoint": config.base_url,
                "reasoning_effort_parameter": config.reasoning_effort,
                "extra_body": config.extra_body,
                "temperature": config.temperature,
            }
        },
    )
    report = write_corpus(result, out_dir / "trajectories.jsonl", out_dir / "quality_report.json")
    report["suite"] = arguments.suite
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_dir / 'trajectories.jsonl'} ({len(result.rows)} rows)")
    print(f"wrote {out_dir / 'attempts.jsonl'} ({write_attempts(result, out_dir / 'attempts.jsonl')} attempts)")

    length_pairs = build_reasoning_length_pairs(result.attempts)
    write_length_pairs(length_pairs, out_dir / "length_pairs.jsonl", out_dir / "length_pairs_report.json")
    print(f"wrote {out_dir / 'length_pairs.jsonl'} ({len(length_pairs)} reasoning-length pairs)")

    attempts = list(result.attempts)
    for path in arguments.student_attempts:
        attempts.extend(read_attempts(path))
    outcome_pairs = build_outcome_pairs(attempts)
    outcome_report = write_outcome_pairs(
        outcome_pairs, out_dir / "outcome_pairs.jsonl", out_dir / "outcome_pairs_report.json"
    )
    print(json.dumps({"policy_pairings": outcome_report["policy_pairings"]}, indent=2))
    print(f"wrote {out_dir / 'outcome_pairs.jsonl'} ({len(outcome_pairs)} outcome pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
