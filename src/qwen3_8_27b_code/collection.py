"""Rejection sampling: turn attempts into execution-verified SFT rows.

This is the route out of the scripted bootstrap corpus. A policy attempts each
task several times, every attempt is graded from outside the workspace, and
only attempts that actually passed become training rows. The reasoning in
those rows is the model's own, generated at the requested effort, which is what
the scripted corpus cannot provide.

Filtering follows docs/data-strategy.md: reject rows whose tests were edited,
whose episode never verified anything, that carry a malformed tool call, or
that ended on a budget rather than an answer. Infrastructure failures are
counted separately and never treated as model failures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import statistics
from typing import Callable

from .episodes import Episode, EpisodeBudget, Policy, run_episode
from .schema import TOOL_SCHEMA_JSON, TOOL_SCHEMA_VERSION, TOOLS
from .tasks import AgentTask, Verdict, materialise

COLLECTOR_VERSION = "rejection-sampling-v1"

PolicyFactory = Callable[[AgentTask, int], Policy]


@dataclass
class Attempt:
    """One graded attempt, kept whether or not it became a training row."""

    task_id: str
    family: str
    seed: int
    episode: Episode
    verdict: Verdict
    rejection: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejection is None

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "termination": self.episode.termination,
            "succeeded": self.verdict.succeeded,
            "rejection": self.rejection,
            "usage": self.episode.usage(),
        }


def rejection_reason(episode: Episode, verdict: Verdict) -> str | None:
    """Why this attempt must not become a demonstration, or None to keep it."""
    if episode.is_infrastructure_failure:
        return "infrastructure_failure"
    if verdict.tampered_paths:
        # Passing by deleting the failing test is the reward hack this corpus
        # must never teach.
        return "protected_files_modified"
    if episode.termination != "assistant_complete":
        return f"terminated_{episode.termination}"
    if episode.invalid_tool_calls:
        return "malformed_tool_call"
    if not verdict.succeeded:
        return "verification_failed"
    if verdict.regression:
        return "regression"
    if not episode.ran_tests:
        # A demonstration that never verifies teaches not verifying, even when
        # the patch happens to be right.
        return "completed_without_verification"
    if not any(message.get("tool_calls") for message in episode.messages):
        return "no_tool_call"
    return None


def action_fingerprint(episode: Episode) -> str:
    """Identify an attempt by the actions it took, for deduplication."""
    actions = [
        [call["function"]["name"], call["function"].get("arguments", {})]
        for message in episode.messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    payload = json.dumps(actions, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_row(
    task: AgentTask,
    attempt: Attempt,
    reasoning_effort: str,
    source: str = COLLECTOR_VERSION,
) -> dict:
    """Render one accepted attempt in the native SFT schema."""
    return {
        "id": f"sft/{task.family}-{task.variant:03d}-s{attempt.seed}",
        "source": source,
        "repo_family": task.family,
        "shape": "sampled",
        "lane": "agentic",
        "reasoning_effort": reasoning_effort,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_schema_json": TOOL_SCHEMA_JSON,
        "tools": TOOLS,
        "messages": attempt.episode.messages,
        "verification": {
            "all_required_tests_pass": True,
            "runner": "python -m pytest -q",
            "hidden_verified": task.has_hidden_verification,
            "hidden_checks": dict(sorted(attempt.verdict.hidden.items())),
        },
        "provenance": {
            "task_id": task.task_id,
            "seed": attempt.seed,
            "collector_version": COLLECTOR_VERSION,
            "usage": attempt.episode.usage(),
        },
    }


@dataclass
class CollectionResult:
    rows: list[dict] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    duplicates_dropped: int = 0

    def report(self) -> dict:
        scored = [attempt for attempt in self.attempts if not attempt.episode.is_infrastructure_failure]
        by_task: dict[str, list[Attempt]] = {}
        for attempt in scored:
            by_task.setdefault(attempt.task_id, []).append(attempt)
        success_rate = {
            task_id: sum(attempt.verdict.succeeded for attempt in group) / len(group)
            for task_id, group in sorted(by_task.items())
        }
        durations = [attempt.episode.wall_seconds for attempt in self.attempts]
        completions = [attempt.episode.completion_tokens for attempt in self.attempts]
        return {
            "collector_version": COLLECTOR_VERSION,
            "attempts": len(self.attempts),
            "infrastructure_failures": len(self.attempts) - len(scored),
            "accepted_rows": len(self.rows),
            "duplicates_dropped": self.duplicates_dropped,
            "acceptance_rate": round(len(self.rows) / len(self.attempts), 4) if self.attempts else 0.0,
            "rejections": dict(
                sorted(Counter(a.rejection for a in self.attempts if a.rejection).items())
            ),
            "hidden_verified_rows": sum(
                1 for row in self.rows if row["verification"]["hidden_verified"]
            ),
            "task_success_rate": success_rate,
            # docs/data-strategy.md difficulty ladder: only the learnable band
            # is useful for the preference and RL curriculum.
            "difficulty_bands": dict(sorted(Counter(
                "trivial" if rate > 0.9 else "learnable" if rate >= 0.2 else "frontier"
                for rate in success_rate.values()
            ).items())),
            "episode_seconds": {
                "mean": round(statistics.mean(durations), 3) if durations else 0.0,
                "max": round(max(durations), 3) if durations else 0.0,
            },
            "completion_tokens": {
                "mean": round(statistics.mean(completions), 1) if completions else 0.0,
                "total": sum(completions),
            },
        }


def collect(
    tasks: list[AgentTask],
    policy_factory: PolicyFactory,
    attempts_per_task: int = 3,
    seeds: tuple[int, ...] = (3407, 9176, 20261),
    budget: EpisodeBudget | None = None,
    reasoning_effort: str = "medium",
    max_rows_per_task: int | None = None,
) -> CollectionResult:
    """Attempt every task repeatedly and keep only what verified."""
    if attempts_per_task > len(seeds):
        raise ValueError(f"{attempts_per_task} attempts requested but only {len(seeds)} seeds given")
    result = CollectionResult()
    seen_actions: set[str] = set()

    for task in tasks:
        kept_for_task = 0
        for seed in seeds[:attempts_per_task]:
            with materialise(task) as workspace:
                episode = run_episode(
                    workspace.harness,
                    policy_factory(task, seed),
                    request=task.request,
                    developer=task.developer,
                    budget=budget,
                )
                verdict = workspace.verify()

            attempt = Attempt(
                task_id=task.task_id,
                family=task.family,
                seed=seed,
                episode=episode,
                verdict=verdict,
                rejection=rejection_reason(episode, verdict),
            )
            result.attempts.append(attempt)
            if not attempt.accepted:
                continue

            fingerprint = action_fingerprint(episode)
            if fingerprint in seen_actions:
                result.duplicates_dropped += 1
                attempt.rejection = "duplicate_actions"
                continue
            if max_rows_per_task is not None and kept_for_task >= max_rows_per_task:
                attempt.rejection = "task_row_cap"
                continue

            seen_actions.add(fingerprint)
            kept_for_task += 1
            result.rows.append(build_row(task, attempt, reasoning_effort))

    return result


def write_corpus(result: CollectionResult, out_path: Path, report_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = result.report()
    report["corpus_sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
