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

The scorecard also carries the thinking budget (reasoning tokens per turn,
the share of generated tokens spent thinking, and turns cut off inside the
think block) and the long-horizon bands, so "shorter thinking at equal
quality" is something the gate can check rather than a hope.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib

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
    # Thinking budget. Defaults keep reports written before these existed
    # loadable; such reports simply do not measure thinking.
    turns: int = 0
    reasoning_tokens: int = 0
    reasoning_chars: int = 0
    reasoning_tokens_reported: bool = False
    thinking_overrun: bool = False
    # Long horizon. The task's designed band is fixed per task, so success
    # per band can be compared between two policies; the band by calls made
    # (horizon_band) moves with the policy and stays a diagnostic.
    task_horizon: str = ""
    peak_prompt_tokens: int = 0

    @property
    def unsupported_success_claim(self) -> bool:
        """Finished with an answer while never running the tests."""
        return self.termination == "assistant_complete" and not self.ran_tests

    @property
    def horizon_band(self) -> str:
        """docs/evaluation.md long-horizon band, by tool calls actually made."""
        return horizon_band(self.tool_calls)

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["unsupported_success_claim"] = self.unsupported_success_claim
        payload["horizon_band"] = self.horizon_band
        # Stored exactly: rounding here made time-based aggregates differ by
        # a thousandth after a JSON round trip, which is a false regression.
        payload["wall_seconds"] = self.wall_seconds
        return payload


HORIZON_BANDS = (("short", 5), ("medium", 15), ("long", 30))


def horizon_band(tool_calls: int) -> str:
    """Short 0-5, medium 6-15, long 16-30, extended 31+ tool calls."""
    for name, upper in HORIZON_BANDS:
        if tool_calls <= upper:
            return name
    return "extended"


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


@dataclass
class EvaluationReport:
    label: str
    records: list[AttemptRecord] = field(default_factory=list)
    # How the numbers were measured: model reference, harness revision,
    # effort, caps, budget and attempt count. ``pairing_problems`` reads it
    # before two reports are gated against each other.
    metadata: dict = field(default_factory=dict)

    @property
    def scored(self) -> list[AttemptRecord]:
        """Attempts the model is accountable for."""
        return [record for record in self.records if not record.infrastructure_failure]

    def task_outcomes(self) -> dict[str, list[bool]]:
        outcomes: dict[str, list[bool]] = {}
        for record in self.scored:
            outcomes.setdefault(record.task_id, []).append(record.succeeded)
        return dict(sorted(outcomes.items()))

    def task_horizons(self) -> dict[str, str]:
        """Designed band per task; empty where a report predates the label."""
        horizons: dict[str, str] = {}
        for record in self.records:
            horizons.setdefault(record.task_id, record.task_horizon)
        return dict(sorted(horizons.items()))

    def attempt_signature(self, scored_only: bool = True) -> tuple[tuple[str, int], ...]:
        """The (task, seed) samples a report's numbers are computed from.

        Only scored attempts count by default: an attempt lost to the
        harness is excluded from every rate, so two reports that attempted
        the same samples but scored different ones are not the same
        experiment, and the ladder refuses to rank them. Success computed
        from different samples would let sampling variation, or a lost
        attempt, pick the effort. ``scored_only=False`` gives what was
        attempted, for reporting.
        """
        records = self.scored if scored_only else self.records
        return tuple(sorted((record.task_id, record.seed) for record in records))

    def scorecard(self) -> dict:
        scored = self.scored
        total = len(scored)
        successes = [record for record in scored if record.succeeded]
        attempted_calls = sum(record.tool_calls + record.invalid_tool_calls for record in scored)
        valid_calls = sum(record.tool_calls for record in scored)

        def per_success(values: list[float]) -> float:
            return round(sum(values) / len(successes), 3) if successes else 0.0

        # Thinking is normalised per assistant turn, not per success: a
        # candidate that solves more tasks legitimately takes more turns, and
        # a per-success ratio against a baseline with no successes is
        # undefined. Per turn is the budget the user actually pays at
        # deployment, one decision at a time.
        total_turns = sum(record.turns for record in scored)
        total_completion = sum(record.completion_tokens for record in scored)
        total_reasoning_tokens = sum(record.reasoning_tokens for record in scored)
        reasoning_measured = bool(scored) and all(record.reasoning_tokens_reported for record in scored)

        def per_turn(total: float) -> float:
            return round(total / total_turns, 3) if total_turns else 0.0

        bands = Counter(record.horizon_band for record in scored)
        success_by_horizon = {
            band: _rate(sum(1 for r in scored if r.horizon_band == band and r.succeeded), count)
            for band, count in sorted(bands.items())
        }
        task_bands = Counter(record.task_horizon or "unlabelled" for record in scored)
        success_by_task_horizon = {
            band: _rate(
                sum(1 for r in scored if (r.task_horizon or "unlabelled") == band and r.succeeded), count
            )
            for band, count in sorted(task_bands.items())
        }

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
            # Thinking budget.
            "reasoning_tokens_reported": reasoning_measured,
            "reasoning_tokens_per_turn": per_turn(total_reasoning_tokens) if reasoning_measured else 0.0,
            "reasoning_chars_per_turn": per_turn(sum(r.reasoning_chars for r in scored)),
            "completion_tokens_per_turn": per_turn(total_completion),
            "reasoning_share_of_completion": (
                round(total_reasoning_tokens / total_completion, 4)
                if reasoning_measured and total_completion else 0.0
            ),
            "thinking_overrun_rate": _rate(sum(r.thinking_overrun for r in scored), total),
            # Long-horizon bands, by tool calls made (diagnostic) and by the
            # band each task was designed for (gated).
            "horizon_bands": dict(sorted(bands.items())),
            "success_by_horizon": success_by_horizon,
            "task_horizon_bands": dict(sorted(task_bands.items())),
            "success_by_task_horizon": success_by_task_horizon,
            # Context: the largest prompt any turn needed, and how often an
            # episode ended because the window ran out.
            "peak_prompt_tokens_max": max((r.peak_prompt_tokens for r in scored), default=0),
            "context_budget_rate": _rate(
                sum(1 for r in scored if r.termination == "context_budget"), total
            ),
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
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EvaluationReport":
        report = cls(label=payload["scorecard"]["label"], metadata=dict(payload.get("metadata") or {}))
        for row in payload["attempts"]:
            fields = {
                key: value for key, value in row.items()
                if key not in {"unsupported_success_claim", "horizon_band"}
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
        turns=episode.turns,
        reasoning_tokens=episode.reasoning_tokens,
        reasoning_chars=episode.reasoning_chars,
        reasoning_tokens_reported=episode.reasoning_tokens_reported,
        thinking_overrun=episode.thinking_overrun,
        task_horizon=task.horizon,
        peak_prompt_tokens=episode.peak_prompt_tokens,
    )


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


# Measurement settings two reports must share before the gate pairs them.
# The harness fingerprint is among them: a change to the verifier, the
# episode loop, the scoring or the task construction changes what a
# report measures even when no task id changes, so two reports taken
# under different harness code are different experiments.
PROVENANCE_STRICT_KEYS = (
    "harness_fingerprint",
    "reasoning_effort",
    "max_new_tokens",
    "max_sequence_length",
    "episode_budget",
    "attempts_per_task",
    "seeds",
    "variants_per_family",
)
# Recorded on the comparison when it differs, but not fatal on its own: a
# repository revision moves with every docs or notebook commit, and the
# fingerprint above already blocks the ones that changed the harness.
PROVENANCE_ADVISORY_KEYS = ("harness_revision",)
# Every key a report must carry before it can be paired at all.
PROVENANCE_REQUIRED_KEYS = ("model",) + PROVENANCE_STRICT_KEYS


def compute_harness_fingerprint(package_dir: Path | None = None) -> str:
    """A digest of the measuring code: every module of this package.

    Name and content of each ``.py`` file, in a fixed order, so the value
    moves only when the harness itself changes, not with docs, notebooks
    or unrelated commits.
    """
    package_dir = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def build_provenance(
    *,
    model: str,
    harness_revision: str,
    reasoning_effort: str,
    max_new_tokens: int | None,
    max_sequence_length: int | None,
    episode_budget: EpisodeBudget | dict,
    attempts_per_task: int,
    seeds: tuple[int, ...] | list[int],
    variants_per_family: int,
    harness_fingerprint: str | None = None,
    measured_at: str | None = None,
) -> dict:
    """The provenance every scored report records.

    One writer for the notebook and the CLI, so the keys ``pairing_problems``
    requires cannot drift between them. ``model`` is whatever was measured,
    pinned: a Hub id or adapter id with its resolved revision, or a scripted
    policy name. The harness fingerprint is taken from the code that is
    running unless given.
    """
    if isinstance(episode_budget, EpisodeBudget):
        episode_budget = {"tool_calls": episode_budget.tool_calls, "wall_seconds": episode_budget.wall_seconds}
    return {
        "model": model,
        "harness_revision": harness_revision,
        "harness_fingerprint": harness_fingerprint or compute_harness_fingerprint(),
        "reasoning_effort": reasoning_effort,
        "max_new_tokens": max_new_tokens,
        "max_sequence_length": max_sequence_length,
        "episode_budget": dict(episode_budget),
        "attempts_per_task": attempts_per_task,
        # The samples themselves, not just their count: two runs of the same
        # attempt count on different seeds are different samples.
        "seeds": list(seeds),
        "variants_per_family": variants_per_family,
        "measured_at": measured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def provenance_mismatches(
    recorded: dict, expected: dict, keys: tuple[str, ...] = PROVENANCE_REQUIRED_KEYS
) -> list[str]:
    """How a recorded provenance differs from the one a session would write.

    Lets a notebook decide whether a stored baseline can stand in for one
    measured now, before it spends a candidate evaluation that the gate
    would then refuse to pair. A key missing from the record counts as a
    difference.
    """
    mismatches = []
    for key in keys:
        if key not in recorded:
            mismatches.append(f"{key}: not recorded, expected {expected.get(key)!r}")
        elif recorded[key] != expected.get(key):
            mismatches.append(f"{key}: recorded {recorded[key]!r}, expected {expected.get(key)!r}")
    return mismatches


def pairing_problems(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> tuple[list[str], list[str]]:
    """Why two reports must not be gated against each other.

    Returns ``(blocking, advisory)``. Blocking: a report missing any of
    ``PROVENANCE_REQUIRED_KEYS`` (a partial record is no evidence), a
    measurement setting in ``PROVENANCE_STRICT_KEYS`` that differs, or two
    reports of the same model reference, which is a baseline paired with
    itself. Advisory: a ``PROVENANCE_ADVISORY_KEYS`` value that differs,
    worth recording next to the verdict.
    """
    blocking: list[str] = []
    advisory: list[str] = []
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        if not report.metadata:
            blocking.append(f"{name} report {report.label!r} carries no provenance; re-measure it")
            continue
        missing = [key for key in PROVENANCE_REQUIRED_KEYS if key not in report.metadata]
        if missing:
            blocking.append(f"{name} report {report.label!r} lacks provenance keys {missing}; re-measure it")
    if blocking:
        return blocking, advisory
    for key in PROVENANCE_STRICT_KEYS:
        before, after = baseline.metadata[key], candidate.metadata[key]
        if before != after:
            blocking.append(f"{key}: baseline {before!r}, candidate {after!r}")
    if baseline.metadata["model"] == candidate.metadata["model"]:
        blocking.append(f"both reports measure {baseline.metadata['model']!r}")
    for key in PROVENANCE_ADVISORY_KEYS:
        before, after = baseline.metadata.get(key), candidate.metadata.get(key)
        if before != after:
            advisory.append(f"{key}: baseline {before!r}, candidate {after!r}")
    return blocking, advisory


def compare(baseline: EvaluationReport, candidate: EvaluationReport) -> dict:
    """Paired per-task comparison, as docs/evaluation.md requires for a small suite."""
    baseline_outcomes = baseline.task_outcomes()
    candidate_outcomes = candidate.task_outcomes()
    shared = sorted(set(baseline_outcomes) & set(candidate_outcomes))
    if not shared:
        raise ValueError("The two reports share no tasks; they cannot be compared.")

    baseline_horizons = baseline.task_horizons()
    candidate_horizons = candidate.task_horizons()
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
            "baseline_task_horizon": baseline_horizons.get(task_id, ""),
            "candidate_task_horizon": candidate_horizons.get(task_id, ""),
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
            "thinking_overrun_rate",
            "reasoning_chars_per_turn",
            "reasoning_tokens_per_turn",
            "context_budget_rate",
        )
    }
    return {
        "baseline": before_card,
        "candidate": after_card,
        "deltas": deltas,
        "thinking": thinking_comparison(before_card, after_card),
        "paired_tasks": paired,
        "task_level": {"wins": wins, "losses": losses, "ties": ties, "tasks": len(shared)},
        "only_in_baseline": sorted(set(baseline_outcomes) - set(candidate_outcomes)),
        "only_in_candidate": sorted(set(candidate_outcomes) - set(baseline_outcomes)),
    }


def _relative_change(before: float, after: float) -> float | None:
    if before <= 0:
        return None
    return round((after - before) / before, 4)


def thinking_comparison(before_card: dict, after_card: dict) -> dict:
    """How much the candidate thinks relative to the baseline.

    Tokens are the measure the user pays for and the one the gate uses; they
    exist only when both policies counted them. Characters are always present
    and serve as the diagnostic when tokens are missing (scripted policies,
    older reports).
    """
    measured = bool(before_card["reasoning_tokens_reported"] and after_card["reasoning_tokens_reported"])
    return {
        "measured_in_tokens": measured,
        "baseline_reasoning_tokens_per_turn": before_card["reasoning_tokens_per_turn"],
        "candidate_reasoning_tokens_per_turn": after_card["reasoning_tokens_per_turn"],
        "relative_change_tokens_per_turn": (
            _relative_change(before_card["reasoning_tokens_per_turn"], after_card["reasoning_tokens_per_turn"])
            if measured else None
        ),
        "baseline_reasoning_chars_per_turn": before_card["reasoning_chars_per_turn"],
        "candidate_reasoning_chars_per_turn": after_card["reasoning_chars_per_turn"],
        "relative_change_chars_per_turn": _relative_change(
            before_card["reasoning_chars_per_turn"], after_card["reasoning_chars_per_turn"]
        ),
        "baseline_reasoning_share": before_card["reasoning_share_of_completion"],
        "candidate_reasoning_share": after_card["reasoning_share_of_completion"],
        "baseline_success_by_horizon": before_card["success_by_horizon"],
        "candidate_success_by_horizon": after_card["success_by_horizon"],
        "baseline_success_by_task_horizon": before_card["success_by_task_horizon"],
        "candidate_success_by_task_horizon": after_card["success_by_task_horizon"],
        "baseline_peak_prompt_tokens_max": before_card["peak_prompt_tokens_max"],
        "candidate_peak_prompt_tokens_max": after_card["peak_prompt_tokens_max"],
    }


DEFAULT_MAX_REASONING_GROWTH = 0.10


def gate(
    comparison: dict,
    minimum_success_delta: float = 0.0,
    max_reasoning_growth: float | None = DEFAULT_MAX_REASONING_GROWTH,
) -> list[GateCheck]:
    """The SFT gate from docs/evaluation.md, evaluated on a comparison.

    Thresholds are policy, and are meant to be frozen before a candidate's
    results are seen. Nothing here claims statistical significance: on a suite
    this size the task-level record is the honest summary.

    ``max_reasoning_growth`` is the thinking-budget check from
    docs/thinking-budget.md: the candidate may not spend more than
    ``(1 + growth)`` times the baseline's reasoning tokens per turn. It is
    evaluated only when both policies counted reasoning tokens, and reports
    itself as unmeasured otherwise rather than failing a run that could not
    measure it. ``None`` removes the check.
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
        GateCheck(
            "thinking_overrun_no_worse",
            deltas["thinking_overrun_rate"] <= 0.0,
            f"turns cut off inside the think block: rate delta {deltas['thinking_overrun_rate']:+.4f}",
        ),
        task_horizon_check(comparison),
    ]
    if max_reasoning_growth is not None:
        checks.append(thinking_budget_check(comparison, max_reasoning_growth))
    return checks


def task_horizon_check(comparison: dict) -> GateCheck:
    """Success may not fall in any horizon band the tasks were designed for.

    The aggregate can hold while the long tasks are lost: a policy taught to
    think less on three-call fixes may stop inspecting enough on seventeen-
    call pipelines. The bands are computed here from the paired tasks, so
    both sides are the same tasks by construction; a report that scored
    tasks the other did not (a narrower suite, or a band lost entirely to
    infrastructure failures) fails the check rather than slipping past it,
    and so do two reports that label a shared task with different bands or
    scored it a different number of times: an attempt lost to the harness
    leaves the success rate intact and the coverage hollow, and the check
    refuses to compare hollow coverage rather than silently accepting it.
    """
    name = "task_horizon_no_worse"
    unmatched = list(comparison.get("only_in_baseline", [])) + list(comparison.get("only_in_candidate", []))
    if unmatched:
        return GateCheck(
            name,
            False,
            "task membership differs, so bands cannot be compared: "
            f"only in baseline {sorted(comparison.get('only_in_baseline', []))}, "
            f"only in candidate {sorted(comparison.get('only_in_candidate', []))}",
        )
    paired = comparison["paired_tasks"]
    before_labels = {task_id: entry.get("baseline_task_horizon", "") for task_id, entry in paired.items()}
    after_labels = {task_id: entry.get("candidate_task_horizon", "") for task_id, entry in paired.items()}
    if not any(before_labels.values()) or not any(after_labels.values()):
        return GateCheck(name, True, "not measured: at least one report carries no task horizons")
    disagreements = sorted(task_id for task_id in paired if before_labels[task_id] != after_labels[task_id])
    if disagreements:
        return GateCheck(name, False, f"horizon labels disagree between the reports for {disagreements}")
    uneven = sorted(
        task_id for task_id, entry in paired.items()
        if entry["baseline_attempts"] != entry["candidate_attempts"]
    )
    if uneven:
        return GateCheck(
            name,
            False,
            "scored attempt counts differ per task, so a band could be hollowed out by "
            f"infrastructure failures on one side: {uneven}",
        )

    def band_rate(side: str) -> dict[str, float]:
        successes: Counter = Counter()
        attempts: Counter = Counter()
        for task_id, entry in paired.items():
            band = before_labels[task_id] or "unlabelled"
            successes[band] += entry[f"{side}_successes"]
            attempts[band] += entry[f"{side}_attempts"]
        return {band: _rate(successes[band], attempts[band]) for band in sorted(attempts)}

    before, after = band_rate("baseline"), band_rate("candidate")
    dropped = [f"{band} {before[band]:.2f}->{after[band]:.2f}" for band in before if after[band] < before[band]]
    held = ", ".join(f"{band} {before[band]:.2f}->{after[band]:.2f}" for band in before)
    return GateCheck(
        name,
        not dropped,
        f"success fell in band(s): {'; '.join(dropped)}" if dropped else f"no band fell: {held}",
    )


def thinking_budget_check(comparison: dict, max_reasoning_growth: float) -> GateCheck:
    thinking = comparison.get("thinking") or {}
    if not thinking.get("measured_in_tokens"):
        return GateCheck(
            "thinking_budget",
            True,
            "not measured: at least one policy did not report reasoning tokens",
        )
    before = thinking["baseline_reasoning_tokens_per_turn"]
    after = thinking["candidate_reasoning_tokens_per_turn"]
    allowed = before * (1.0 + max_reasoning_growth)
    return GateCheck(
        "thinking_budget",
        after <= allowed,
        f"reasoning tokens per turn {before:.1f} -> {after:.1f}; "
        f"ceiling {allowed:.1f} at +{max_reasoning_growth:.0%}",
    )


def gate_passed(checks: list[GateCheck]) -> bool:
    return all(check.passed for check in checks)


EFFORT_ORDER = ("low", "medium", "xhigh")


def effort_ladder(reports: dict[str, EvaluationReport], success_tolerance: float = 0.0) -> dict:
    """Tabulate the effort ladder of docs/thinking-budget.md and pick a deployment effort.

    ``reports`` maps an effort label to the held-out report scored at that
    effort with the same policy on the same tasks, attempts and seeds, none
    of them lost to the harness. A
    rung is eligible when its episode success is within ``success_tolerance``
    of the best rung both in aggregate and in every designed horizon band,
    so a cheaper rung that trades the pipeline tasks for an extra short one
    is never recommended. Among the eligible rungs the recommendation is the
    one that thinks least; a tie goes to the lower overrun rate, then to the
    lower effort. When no rung keeps every band, ``recommended`` is None and
    the rungs have to be read by band. It chooses the dial setting for
    deployment and for the gate baseline; the training levers in
    thinking-budget.md are what move the model.
    """
    if not reports:
        raise ValueError("an effort ladder needs at least one report")
    unknown = sorted(set(reports) - set(EFFORT_ORDER))
    if unknown:
        raise ValueError(f"unknown reasoning effort(s) {unknown}; the template accepts {EFFORT_ORDER}")
    signatures = {effort: report.attempt_signature() for effort, report in reports.items()}
    if len(set(signatures.values())) != 1:
        lost = {
            effort: len(report.records) - len(report.scored)
            for effort, report in reports.items()
            if len(report.records) != len(report.scored)
        }
        raise ValueError(
            "every rung of the ladder must score the same tasks with the same attempts and seeds; "
            "success from unequal samples would let sampling variation pick the effort"
            + (f" (attempts lost to infrastructure failures, re-run them: {lost})" if lost else "")
        )

    rungs = {}
    for effort in EFFORT_ORDER:
        if effort not in reports:
            continue
        card = reports[effort].scorecard()
        rungs[effort] = {
            "label": card["label"],
            "scored_attempts": card["scored_attempts"],
            "infrastructure_failures": card["infrastructure_failures"],
            "episode_success": card["episode_success"],
            "reasoning_tokens_reported": card["reasoning_tokens_reported"],
            "reasoning_tokens_per_turn": card["reasoning_tokens_per_turn"],
            "reasoning_chars_per_turn": card["reasoning_chars_per_turn"],
            "thinking_overrun_rate": card["thinking_overrun_rate"],
            "context_budget_rate": card["context_budget_rate"],
            "peak_prompt_tokens_max": card["peak_prompt_tokens_max"],
            "success_by_task_horizon": card["success_by_task_horizon"],
        }
    unit = "tokens" if all(rung["reasoning_tokens_reported"] for rung in rungs.values()) else "chars"
    best = max(rung["episode_success"] for rung in rungs.values())
    bands = sorted({band for rung in rungs.values() for band in rung["success_by_task_horizon"]})
    best_by_band = {
        band: max(rung["success_by_task_horizon"].get(band, 0.0) for rung in rungs.values()) for band in bands
    }

    def keeps_every_band(rung: dict) -> bool:
        return all(
            rung["success_by_task_horizon"].get(band, 0.0) >= best_by_band[band] - success_tolerance
            for band in bands
        )

    eligible = [
        effort
        for effort, rung in rungs.items()
        if rung["episode_success"] >= best - success_tolerance and keeps_every_band(rung)
    ]
    recommended = min(
        eligible,
        key=lambda effort: (
            rungs[effort][f"reasoning_{unit}_per_turn"],
            rungs[effort]["thinking_overrun_rate"],
            EFFORT_ORDER.index(effort),
        ),
    ) if eligible else None
    return {
        "unit": unit,
        "best_success": best,
        "best_success_by_band": best_by_band,
        "success_tolerance": success_tolerance,
        "rungs": rungs,
        "eligible": eligible,
        "recommended": recommended,
        "note": (
            None if recommended is not None else
            "no rung keeps every designed band within the tolerance; nothing is recommended, read the rungs by band"
        ),
    }


def write_report(report: EvaluationReport, path: Path) -> dict:
    payload = report.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def read_report(path: Path) -> EvaluationReport:
    return EvaluationReport.from_dict(json.loads(Path(path).read_text()))
