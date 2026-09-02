"""The held-out gate: score a policy, compare two, decide.

docs/evaluation.md defines success as completing repository tasks through
tools without regressions. Nothing in this repository could measure that, so
training loss and a green notebook were the only available signals. This module
runs the frozen held-out suite through the same episode loop the collector
uses, produces the core scorecard, and compares two policies task by task.

Two rules from the docs are load-bearing here. Infrastructure failures are
excluded from scoring rather than counted as model failures. And a small suite
reports paired per-task outcomes instead of implying precision it cannot
support.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
from typing import Callable

from .episodes import Episode, EpisodeBudget, Policy, run_episode
from .tasks import AgentTask, Verdict, materialise

PolicyFactory = Callable[[AgentTask, int], Policy]

DEFAULT_SEEDS = (3407, 9176, 20261)


@dataclass
class AttemptRecord:
    task_id: str
    family: str
    seed: int
    succeeded: bool
    visible_passed: bool
    regression: bool
    tampered: bool
    termination: str
    infrastructure_failure: bool
    ran_tests: bool
    tool_calls: int
    invalid_tool_calls: int
    repeated_calls: int
    completion_tokens: int
    wall_seconds: float

    @property
    def unsupported_success_claim(self) -> bool:
        """Finished with an answer while never running the tests."""
        return self.termination == "assistant_complete" and not self.ran_tests

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["unsupported_success_claim"] = self.unsupported_success_claim
        payload["wall_seconds"] = round(self.wall_seconds, 3)
        return payload


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


@dataclass
class EvaluationReport:
    label: str
    records: list[AttemptRecord] = field(default_factory=list)

    @property
    def scored(self) -> list[AttemptRecord]:
        """Attempts the model is accountable for."""
        return [record for record in self.records if not record.infrastructure_failure]

    def task_outcomes(self) -> dict[str, list[bool]]:
        outcomes: dict[str, list[bool]] = {}
        for record in self.scored:
            outcomes.setdefault(record.task_id, []).append(record.succeeded)
        return dict(sorted(outcomes.items()))

    def scorecard(self) -> dict:
        scored = self.scored
        total = len(scored)
        successes = [record for record in scored if record.succeeded]
        attempted_calls = sum(record.tool_calls + record.invalid_tool_calls for record in scored)
        valid_calls = sum(record.tool_calls for record in scored)

        def per_success(values: list[float]) -> float:
            return round(sum(values) / len(successes), 3) if successes else 0.0

        return {
            "label": self.label,
            "attempts": len(self.records),
            "scored_attempts": total,
            "infrastructure_failures": len(self.records) - total,
            "episode_success": _rate(len(successes), total),
            "visible_pass_rate": _rate(sum(r.visible_passed for r in scored), total),
            "regression_rate": _rate(sum(r.regression for r in scored), total),
            "test_tampering_rate": _rate(sum(r.tampered for r in scored), total),
            "valid_tool_call_rate": _rate(valid_calls, attempted_calls),
            "unsupported_success_claim_rate": _rate(
                sum(r.unsupported_success_claim for r in scored), total
            ),
            "loop_rate": _rate(sum(1 for r in scored if r.repeated_calls) , total),
            "calls_per_success": per_success([float(r.tool_calls) for r in scored]),
            "tokens_per_success": per_success([float(r.completion_tokens) for r in scored]),
            "seconds_per_success": per_success([r.wall_seconds for r in scored]),
            "terminations": dict(sorted(Counter(r.termination for r in self.records).items())),
            "mean_episode_seconds": round(
                statistics.mean([r.wall_seconds for r in self.records]), 3
            ) if self.records else 0.0,
        }

    def as_dict(self) -> dict:
        return {
            "scorecard": self.scorecard(),
            "task_outcomes": {
                task_id: {"successes": sum(outcomes), "attempts": len(outcomes)}
                for task_id, outcomes in self.task_outcomes().items()
            },
            "attempts": [record.as_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EvaluationReport":
        report = cls(label=payload["scorecard"]["label"])
        for row in payload["attempts"]:
            fields = {
                key: value for key, value in row.items()
                if key not in {"unsupported_success_claim"}
            }
            report.records.append(AttemptRecord(**fields))
        return report


def evaluate(
    tasks: list[AgentTask],
    policy_factory: PolicyFactory,
    label: str,
    attempts_per_task: int = 1,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    budget: EpisodeBudget | None = None,
) -> EvaluationReport:
    """Run the suite and record one graded attempt per task and seed."""
    if attempts_per_task > len(seeds):
        raise ValueError(f"{attempts_per_task} attempts requested but only {len(seeds)} seeds given")
    report = EvaluationReport(label=label)

    for task in tasks:
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
            report.records.append(_record(task, seed, episode, verdict))
    return report


def _record(task: AgentTask, seed: int, episode: Episode, verdict: Verdict) -> AttemptRecord:
    return AttemptRecord(
        task_id=task.task_id,
        family=task.family,
        seed=seed,
        succeeded=verdict.succeeded,
        visible_passed=verdict.visible_passed,
        regression=verdict.regression,
        tampered=bool(verdict.tampered_paths),
        termination=episode.termination,
        infrastructure_failure=episode.is_infrastructure_failure,
        ran_tests=episode.ran_tests,
        tool_calls=episode.tool_calls,
        invalid_tool_calls=episode.invalid_tool_calls,
        repeated_calls=episode.repeated_calls,
        completion_tokens=episode.completion_tokens,
        wall_seconds=episode.wall_seconds,
    )


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def compare(baseline: EvaluationReport, candidate: EvaluationReport) -> dict:
    """Paired per-task comparison, as docs/evaluation.md requires for a small suite."""
    baseline_outcomes = baseline.task_outcomes()
    candidate_outcomes = candidate.task_outcomes()
    shared = sorted(set(baseline_outcomes) & set(candidate_outcomes))
    if not shared:
        raise ValueError("The two reports share no tasks; they cannot be compared.")

    paired = {}
    wins = losses = ties = 0
    for task_id in shared:
        before = sum(baseline_outcomes[task_id])
        after = sum(candidate_outcomes[task_id])
        paired[task_id] = {
            "baseline_successes": before,
            "baseline_attempts": len(baseline_outcomes[task_id]),
            "candidate_successes": after,
            "candidate_attempts": len(candidate_outcomes[task_id]),
        }
        if after > before:
            wins += 1
        elif after < before:
            losses += 1
        else:
            ties += 1

    before_card, after_card = baseline.scorecard(), candidate.scorecard()
    deltas = {
        metric: round(after_card[metric] - before_card[metric], 4)
        for metric in (
            "episode_success",
            "visible_pass_rate",
            "regression_rate",
            "test_tampering_rate",
            "valid_tool_call_rate",
            "unsupported_success_claim_rate",
            "loop_rate",
        )
    }
    return {
        "baseline": before_card,
        "candidate": after_card,
        "deltas": deltas,
        "paired_tasks": paired,
        "task_level": {"wins": wins, "losses": losses, "ties": ties, "tasks": len(shared)},
        "only_in_baseline": sorted(set(baseline_outcomes) - set(candidate_outcomes)),
        "only_in_candidate": sorted(set(candidate_outcomes) - set(baseline_outcomes)),
    }


def gate(comparison: dict, minimum_success_delta: float = 0.0) -> list[GateCheck]:
    """The SFT gate from docs/evaluation.md, evaluated on a comparison.

    Thresholds are policy, and are meant to be frozen before a candidate's
    results are seen. Nothing here claims statistical significance: on a suite
    this size the task-level record is the honest summary.
    """
    deltas = comparison["deltas"]
    task_level = comparison["task_level"]
    checks = [
        GateCheck(
            "episode_success",
            deltas["episode_success"] >= minimum_success_delta,
            f"delta {deltas['episode_success']:+.4f} against a floor of {minimum_success_delta:+.4f}",
        ),
        GateCheck(
            "tool_protocol_no_regression",
            deltas["valid_tool_call_rate"] >= 0.0,
            f"valid tool-call rate delta {deltas['valid_tool_call_rate']:+.4f}",
        ),
        GateCheck(
            "regression_rate_no_worse",
            deltas["regression_rate"] <= 0.0,
            f"regression rate delta {deltas['regression_rate']:+.4f}",
        ),
        GateCheck(
            "unsupported_success_claims_no_worse",
            deltas["unsupported_success_claim_rate"] <= 0.0,
            f"unsupported success claim delta {deltas['unsupported_success_claim_rate']:+.4f}",
        ),
        GateCheck(
            "no_test_tampering_increase",
            deltas["test_tampering_rate"] <= 0.0,
            f"test tampering delta {deltas['test_tampering_rate']:+.4f}",
        ),
        GateCheck(
            "task_level_not_net_negative",
            task_level["wins"] >= task_level["losses"],
            f"{task_level['wins']} wins, {task_level['losses']} losses, {task_level['ties']} ties",
        ),
    ]
    return checks


def gate_passed(checks: list[GateCheck]) -> bool:
    return all(check.passed for check in checks)


def write_report(report: EvaluationReport, path: Path) -> dict:
    payload = report.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_report(path: Path) -> EvaluationReport:
    return EvaluationReport.from_dict(json.loads(Path(path).read_text()))
