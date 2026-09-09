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

Selection follows docs/thinking-budget.md: when several attempts at a task
verify, the ones that reached the answer with the least reasoning are kept
first (shortest rejection sampling). The attempts that did not become rows are
retained, because the reasoning-length preference pairs are built from them.
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
from .thinking import reasoning_length

COLLECTOR_VERSION = "rejection-sampling-v2"

# Which verified attempts become rows when a task has more than the cap.
# ``shortest_reasoning`` keeps the ones that thought least; ``first`` keeps
# them in seed order, which is the pre-thinking-budget behaviour.
SELECTIONS = ("shortest_reasoning", "first")

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
    policy: str = ""

    @property
    def accepted(self) -> bool:
        return self.rejection is None

    @property
    def verified_success(self) -> bool:
        """Passed every filter, whether or not it was kept as a row.

        ``rejection`` is overwritten when an accepted attempt is dropped as a
        duplicate or by the per-task cap; this re-derives the underlying fact,
        which the length-pair builder needs.
        """
        return rejection_reason(self.episode, self.verdict) is None

    def summary(self) -> dict:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "policy": self.policy,
            "termination": self.episode.termination,
            "succeeded": self.verdict.succeeded,
            "rejection": self.rejection,
            "usage": self.episode.usage(),
        }

    def as_dict(self) -> dict:
        """The full record, enough to rebuild the attempt in another process.

        Pair builders need attempts from more than one run (a teacher's and a
        student's), and a run's attempts are worth keeping anyway: the
        rejected ones are the audit trail behind the corpus.
        """
        return {
            "task_id": self.task_id,
            "family": self.family,
            "seed": self.seed,
            "policy": self.policy,
            "rejection": self.rejection,
            "episode": {
                "messages": self.episode.messages,
                "termination": self.episode.termination,
                "final_text": self.episode.final_text,
                "tool_calls": self.episode.tool_calls,
                "invalid_tool_calls": self.episode.invalid_tool_calls,
                "repeated_calls": self.episode.repeated_calls,
                "tool_errors": self.episode.tool_errors,
                "turns": self.episode.turns,
                "prompt_tokens": self.episode.prompt_tokens,
                "completion_tokens": self.episode.completion_tokens,
                "wall_seconds": self.episode.wall_seconds,
                "reasoning_tokens": self.episode.reasoning_tokens,
                "reasoning_chars": self.episode.reasoning_chars,
                "reported_reasoning_turns": self.episode.reported_reasoning_turns,
                "thinking_overrun": self.episode.thinking_overrun,
                "turn_reasoning_tokens": self.episode.turn_reasoning_tokens,
            },
            "verdict": {
                "visible_exit": self.verdict.visible_exit,
                "hidden": self.verdict.hidden,
                "hidden_output": self.verdict.hidden_output,
                "tampered_paths": self.verdict.tampered_paths,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Attempt":
        return cls(
            task_id=payload["task_id"],
            family=payload["family"],
            seed=payload["seed"],
            episode=Episode(**payload["episode"]),
            verdict=Verdict(**payload["verdict"]),
            rejection=payload.get("rejection"),
            policy=payload.get("policy", ""),
        )


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
    selection: str = "shortest_reasoning",
    provenance: dict | None = None,
) -> dict:
    """Render one accepted attempt in the native SFT schema."""
    length, unit = reasoning_length(attempt.episode)
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
            "policy": attempt.policy,
            "collector_version": COLLECTOR_VERSION,
            "selection": selection,
            "reasoning_length": length,
            "reasoning_unit": unit,
            "usage": attempt.episode.usage(),
            **(provenance or {}),
        },
    }


@dataclass
class CollectionResult:
    rows: list[dict] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    duplicates_dropped: int = 0
    selection: str = "shortest_reasoning"
    reasoning_effort: str = "medium"
    policy_label: str = ""

    def verified_attempts(self) -> list[Attempt]:
        """Every attempt that passed the filters, kept as a row or not."""
        return [attempt for attempt in self.attempts if attempt.verified_success]

    def thinking_report(self) -> dict:
        """How much the policy thought, split by whether it worked.

        Read this next to the acceptance rate: if failed attempts think far
        more per turn than verified ones, the policy is spending tokens on
        the tasks it cannot do, and a tighter budget costs little. If the
        reverse holds, brevity is being bought with correctness.
        """
        scored = [attempt for attempt in self.attempts if not attempt.episode.is_infrastructure_failure]
        measured = bool(scored) and all(attempt.episode.reasoning_tokens_reported for attempt in scored)
        unit = "tokens" if measured else "chars"

        def per_turn(group: list[Attempt]) -> float | None:
            turns = sum(attempt.episode.turns for attempt in group)
            if not turns:
                return None
            total = sum(
                attempt.episode.reasoning_tokens if measured else attempt.episode.reasoning_chars
                for attempt in group
            )
            return round(total / turns, 2)

        verified = [attempt for attempt in scored if attempt.verified_success]
        failed = [attempt for attempt in scored if not attempt.verified_success]
        return {
            "unit": unit,
            "reasoning_tokens_reported": measured,
            "selection": self.selection,
            "reasoning_effort": self.reasoning_effort,
            "per_turn": {
                "verified": per_turn(verified),
                "not_verified": per_turn(failed),
                "kept_rows": per_turn(
                    [attempt for attempt in verified if attempt.rejection is None]
                ),
            },
            "thinking_overruns": sum(attempt.episode.thinking_overrun for attempt in scored),
        }

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
            "policy": self.policy_label,
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
            "thinking": self.thinking_report(),
        }


def collect(
    tasks: list[AgentTask],
    policy_factory: PolicyFactory,
    attempts_per_task: int = 3,
    seeds: tuple[int, ...] = (3407, 9176, 20261),
    budget: EpisodeBudget | None = None,
    reasoning_effort: str = "medium",
    max_rows_per_task: int | None = None,
    selection: str = "shortest_reasoning",
    policy_label: str = "",
    source: str = COLLECTOR_VERSION,
    provenance: dict | None = None,
) -> CollectionResult:
    """Attempt every task repeatedly and keep only what verified.

    With ``selection="shortest_reasoning"`` the rows kept under
    ``max_rows_per_task`` are the verified attempts that reasoned least, so
    the SFT corpus teaches the shortest path the policy has itself shown to
    work. Duplicate action sequences are still dropped first: two attempts
    that took the same actions are one demonstration, not two.

    ``policy_label`` names the policy on every attempt and row (a teacher's
    model id, a checkpoint revision), ``source`` replaces the row's source
    field, and ``provenance`` is merged into each row's provenance, which is
    how a distillation corpus records where its rows came from.
    """
    if attempts_per_task > len(seeds):
        raise ValueError(f"{attempts_per_task} attempts requested but only {len(seeds)} seeds given")
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}, not {selection!r}")
    result = CollectionResult(selection=selection, reasoning_effort=reasoning_effort, policy_label=policy_label)
    seen_actions: set[str] = set()

    def preference_order(attempt: Attempt) -> tuple:
        if selection == "shortest_reasoning":
            return (reasoning_length(attempt.episode)[0], attempt.seed)
        return (attempt.seed,)

    for task in tasks:
        accepted: list[Attempt] = []
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
                policy=policy_label,
            )
            result.attempts.append(attempt)
            if attempt.accepted:
                accepted.append(attempt)

        # Two attempts that took the same actions are one demonstration. Which
        # one survives is the selection's decision, so order first and dedupe
        # second: under shortest_reasoning the copy that thought least wins.
        candidates: list[Attempt] = []
        for attempt in sorted(accepted, key=preference_order):
            fingerprint = action_fingerprint(attempt.episode)
            if fingerprint in seen_actions:
                result.duplicates_dropped += 1
                attempt.rejection = "duplicate_actions"
                continue
            seen_actions.add(fingerprint)
            candidates.append(attempt)

        for position, attempt in enumerate(candidates):
            if max_rows_per_task is not None and position >= max_rows_per_task:
                attempt.rejection = "task_row_cap"
                continue
            result.rows.append(
                build_row(
                    task, attempt, reasoning_effort, source=source, selection=selection, provenance=provenance
                )
            )

    return result


def write_attempts(result: CollectionResult, path: Path) -> int:
    """Persist every attempt, kept or not, as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for attempt in result.attempts:
            handle.write(json.dumps(attempt.as_dict(), ensure_ascii=False) + "\n")
    return len(result.attempts)


def read_attempts(path: Path) -> list[Attempt]:
    return [
        Attempt.from_dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
