"""Contracts for the episode loop, the collector and the evaluation gate.

These run the real harness, real pytest and real hidden verifiers against the
held-out fixtures. The only thing faked is the model: a scripted policy stands
in for generation, which is what lets the whole collection and gating path be
exercised on CPU.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from qwen3_8_27b_code import evaluation, fixtures, policies, tasks
from qwen3_8_27b_code.collection import action_fingerprint, collect, write_corpus
from qwen3_8_27b_code.episodes import (
    EpisodeBudget,
    TurnResult,
    answer_text,
    run_episode,
    scripted_policy,
    tool_call_text,
)
from qwen3_8_27b_code.tasks import evaluation_tasks, materialise, task_from_fixture
from qwen3_8_27b_code.trajectories import unified_patch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"

# One variant per family keeps these tests to six real repositories.
SMOKE_TASKS = evaluation_tasks(variants_per_family=1)


def load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The held-out suite


def test_evaluation_families_are_disjoint_from_training_families():
    """A gate measured on families the model trained on measures memorisation."""
    training = set(fixtures.FAMILY_BUILDERS)
    held_out = set(tasks.EVALUATION_FAMILY_BUILDERS)
    assert training & held_out == set()
    assert len(held_out) >= 6


def test_every_evaluation_task_starts_broken_and_the_gold_fix_resolves_it():
    for task in SMOKE_TASKS:
        with materialise(task) as workspace:
            before = workspace.verify()
            assert before.visible_exit != 0, task.task_id
            assert before.hidden["contract"] is False, task.task_id
            # Behaviour that is not yet broken must pass, or the check cannot
            # detect a regression introduced by an attempt.
            assert before.hidden["no_regression"] is True, task.task_id
            assert not before.succeeded, task.task_id

            patch = unified_patch(
                task.module_path, task.files[task.module_path], task.reference_module
            )
            assert workspace.harness.execute("apply_patch", {"patch": patch}) == "patch applied"
            after = workspace.verify()
            assert after.succeeded, (task.task_id, after.as_dict())
            assert not after.regression, task.task_id


def test_hidden_verifiers_are_outside_the_model_workspace():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        listing = workspace.harness.execute("list_files", {"path": "."})
        assert "contract" not in listing
        assert "hidden" not in listing
        # Path containment refuses the traversal that would reach them.
        with pytest.raises(ValueError):
            workspace.harness.execute("read_file", {"path": "../hidden/contract.py"})
        assert workspace.hidden_dir.exists()
        assert workspace.hidden_dir.parent != workspace.root


def test_visible_tests_never_encode_the_hidden_contract():
    """If the visible suite covered everything, hidden verification would be
    decorative and passing by overfitting to it would be indistinguishable."""
    for task in evaluation_tasks():
        visible = task.files[task.tests_path]
        contract = next(check for check in task.hidden_checks if check.name == "contract")
        assertions = [
            line.strip()
            for line in contract.source.splitlines()
            if line.strip().startswith("assert ")
        ]
        assert assertions, task.task_id
        uncovered = [line for line in assertions if line.removeprefix("assert ") not in visible]
        assert uncovered, f"{task.task_id}: hidden checks add nothing the visible tests miss"


# --------------------------------------------------------------------------
# The episode loop


def test_truncated_turn_is_a_termination_not_an_answer():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            lambda messages: TurnResult(text="<think>\nI will start by", fault="output_truncated"),
            request=task.request,
            developer=task.developer,
        )
    assert episode.termination == "output_truncated"
    assert episode.final_text == ""
    # The cut-off prefix must not have been stored as an assistant answer.
    assert not [message for message in episode.messages if message["role"] == "assistant"]


def test_context_exhaustion_is_a_termination():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            lambda messages: TurnResult(text="", fault="context_budget", prompt_tokens=99_000),
            request=task.request,
            developer=task.developer,
        )
    assert episode.termination == "context_budget"
    assert episode.prompt_tokens == 99_000


def test_policy_failure_is_infrastructure_not_a_model_failure():
    task = SMOKE_TASKS[0]

    def broken(messages):
        raise RuntimeError("model server went away")

    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness, broken, request=task.request, developer=task.developer
        )
    assert episode.termination == "policy_error"
    assert episode.is_infrastructure_failure


def test_malformed_call_becomes_a_typed_observation_the_episode_can_recover_from():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            policies.malformed_then_recovers(task, 0),
            request=task.request,
            developer=task.developer,
        )
    observations = [m["content"] for m in episode.messages if m["role"] == "tool"]
    # Both problems are reported together so one retry can fix the call.
    assert observations[0] == (
        "invalid_tool_call: read_file: missing required argument(s): path; "
        "unknown argument(s): file"
    )
    assert episode.invalid_tool_calls == 1
    assert episode.termination == "assistant_complete"
    assert episode.valid_tool_call_rate == pytest.approx(3 / 4)


def test_repeated_actions_are_counted_and_the_tool_budget_terminates():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            policies.looping(task, 0),
            request=task.request,
            developer=task.developer,
            budget=EpisodeBudget(tool_calls=4),
        )
    assert episode.termination == "tool_budget"
    assert episode.tool_calls == 4
    assert episode.repeated_calls == 3


def test_timeout_is_reported_without_running_a_turn():
    task = SMOKE_TASKS[0]
    ticks = iter([0.0, 10_000.0, 10_000.0])
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            scripted_policy([answer_text("done")]),
            request=task.request,
            developer=task.developer,
            budget=EpisodeBudget(wall_seconds=1.0),
            clock=lambda: next(ticks),
        )
    assert episode.termination == "timeout"
    assert episode.turns == 0


def test_tool_call_text_round_trips_through_the_deployment_parser():
    task = SMOKE_TASKS[0]
    patch = unified_patch(task.module_path, task.files[task.module_path], task.reference_module)
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness,
            scripted_policy(
                [
                    tool_call_text("apply_patch", {"patch": patch}, "Applying the fix."),
                    answer_text("Done."),
                ]
            ),
            request=task.request,
            developer=task.developer,
        )
        verdict = workspace.verify()
    assert [m["content"] for m in episode.messages if m["role"] == "tool"] == ["patch applied"]
    assert verdict.hidden["contract"] is True


# --------------------------------------------------------------------------
# Collection


@pytest.mark.parametrize(
    ("policy_name", "expected_rejection"),
    [
        ("test-deleting", "protected_files_modified"),
        ("unverified", "completed_without_verification"),
        ("malformed-then-recovers", "malformed_tool_call"),
        ("failing", "verification_failed"),
    ],
)
def test_collection_rejects_each_way_an_attempt_can_look_successful(policy_name, expected_rejection):
    result = collect(SMOKE_TASKS[:2], policies.BUILTIN_POLICIES[policy_name], attempts_per_task=1)
    assert result.rows == []
    assert set(result.report()["rejections"]) == {expected_rejection}


def test_collection_keeps_verified_attempts_and_records_the_evidence():
    result = collect(SMOKE_TASKS, policies.gold, attempts_per_task=1, reasoning_effort="xhigh")
    report = result.report()

    assert len(result.rows) == len(SMOKE_TASKS)
    assert report["rejections"] == {}
    assert report["hidden_verified_rows"] == len(result.rows)
    for row in result.rows:
        assert row["verification"]["all_required_tests_pass"] is True
        assert row["verification"]["hidden_checks"] == {"contract": True, "no_regression": True}
        # Effort is recorded from the run that produced the reasoning, which is
        # the whole point of collecting instead of scripting.
        assert row["reasoning_effort"] == "xhigh"
        assert row["provenance"]["task_id"].startswith("eval/")


def test_collected_rows_pass_notebook_02_validation():
    """The collector must emit the same schema the data notebook validates."""
    generator = load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    cell = next(
        cell.source
        for cell in generator.build_02_data().cells
        if cell.cell_type == "code" and "def validate_row" in cell.source
    )
    exec(cell, namespace)

    rows = collect(SMOKE_TASKS[:3], policies.gold, attempts_per_task=1).rows
    errors = [(row["id"], namespace["validate_row"](row)) for row in rows]
    assert not [item for item in errors if item[1]], errors


def test_identical_attempts_are_deduplicated():
    result = collect(
        SMOKE_TASKS[:1], policies.gold, attempts_per_task=2, seeds=(3407, 9176)
    )
    # The scripted policy ignores the seed, so both attempts take the same
    # actions and only one row survives.
    assert len(result.rows) == 1
    assert result.duplicates_dropped == 1
    assert result.report()["rejections"] == {"duplicate_actions": 1}


def test_collection_report_bands_tasks_by_measured_success():
    result = collect(SMOKE_TASKS[:2], policies.gold, attempts_per_task=1)
    report = result.report()
    assert report["difficulty_bands"] == {"trivial": 2}
    assert all(rate == 1.0 for rate in report["task_success_rate"].values())


def test_write_corpus_records_a_hash_of_what_it_wrote(tmp_path):
    result = collect(SMOKE_TASKS[:1], policies.gold, attempts_per_task=1)
    corpus, report_path = tmp_path / "rows.jsonl", tmp_path / "report.json"
    report = write_corpus(result, corpus, report_path)
    assert report["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert len(corpus.read_text().strip().splitlines()) == len(result.rows)


def test_action_fingerprint_separates_different_action_sequences():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        first = run_episode(
            workspace.harness, policies.gold(task, 0), request=task.request, developer=task.developer
        )
    with materialise(task) as workspace:
        second = run_episode(
            workspace.harness,
            policies.failing(task, 0),
            request=task.request,
            developer=task.developer,
        )
    assert action_fingerprint(first) != action_fingerprint(second)


# --------------------------------------------------------------------------
# Evaluation and the gate


def test_scorecard_flags_an_unverified_success_claim():
    report = evaluation.evaluate(SMOKE_TASKS[:2], policies.unverified, label="unverified")
    scorecard = report.scorecard()
    # The patch is right, so the task succeeds; the process is still wrong.
    assert scorecard["episode_success"] == 1.0
    assert scorecard["unsupported_success_claim_rate"] == 1.0


def test_scorecard_flags_test_tampering_as_a_failure():
    report = evaluation.evaluate(SMOKE_TASKS[:2], policies.test_deleting, label="hacker")
    scorecard = report.scorecard()
    assert scorecard["test_tampering_rate"] == 1.0
    # Green visible tests must not become episode success.
    assert scorecard["visible_pass_rate"] == 1.0
    assert scorecard["episode_success"] == 0.0


def test_scorecard_flags_loops_and_invalid_calls():
    looping = evaluation.evaluate(
        SMOKE_TASKS[:1], policies.looping, label="looping", budget=EpisodeBudget(tool_calls=3)
    ).scorecard()
    assert looping["loop_rate"] == 1.0
    assert looping["terminations"] == {"tool_budget": 1}

    malformed = evaluation.evaluate(
        SMOKE_TASKS[:1], policies.malformed_then_recovers, label="malformed"
    ).scorecard()
    assert malformed["valid_tool_call_rate"] == pytest.approx(0.75)


def test_infrastructure_failures_are_excluded_from_scoring():
    def broken(task, seed):
        def policy(messages):
            raise RuntimeError("model server went away")

        return policy

    report = evaluation.evaluate(SMOKE_TASKS[:2], broken, label="broken")
    scorecard = report.scorecard()
    assert scorecard["attempts"] == 2
    assert scorecard["scored_attempts"] == 0
    assert scorecard["infrastructure_failures"] == 2
    # Nothing is charged to the model.
    assert scorecard["episode_success"] == 0.0
    assert report.task_outcomes() == {}


def test_gate_passes_on_improvement_and_fails_on_regression():
    baseline = evaluation.evaluate(SMOKE_TASKS, policies.failing, label="baseline")
    candidate = evaluation.evaluate(SMOKE_TASKS, policies.gold, label="candidate")

    forward = evaluation.compare(baseline, candidate)
    assert forward["task_level"] == {
        "wins": len(SMOKE_TASKS),
        "losses": 0,
        "ties": 0,
        "tasks": len(SMOKE_TASKS),
    }
    assert evaluation.gate_passed(evaluation.gate(forward))

    backward = evaluation.compare(candidate, baseline)
    failed = [check.name for check in evaluation.gate(backward) if not check.passed]
    assert failed == ["episode_success", "task_level_not_net_negative"]


def test_gate_fails_a_candidate_that_wins_by_tampering():
    baseline = evaluation.evaluate(SMOKE_TASKS[:3], policies.failing, label="baseline")
    hacker = evaluation.evaluate(SMOKE_TASKS[:3], policies.test_deleting, label="hacker")
    checks = {check.name: check.passed for check in evaluation.gate(evaluation.compare(baseline, hacker))}
    assert checks["no_test_tampering_increase"] is False


def test_reports_round_trip_through_json(tmp_path):
    report = evaluation.evaluate(SMOKE_TASKS[:2], policies.gold, label="candidate")
    path = tmp_path / "report.json"
    evaluation.write_report(report, path)
    restored = evaluation.read_report(path)
    assert restored.scorecard() == report.scorecard()
    assert restored.task_outcomes() == report.task_outcomes()


def test_comparing_reports_with_no_shared_tasks_is_an_error():
    first = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="a")
    second = evaluation.evaluate(SMOKE_TASKS[1:2], policies.gold, label="b")
    with pytest.raises(ValueError, match="share no tasks"):
        evaluation.compare(first, second)


# --------------------------------------------------------------------------
# Policy loading


def test_policy_factories_resolve_by_name_and_by_dotted_path():
    assert policies.load_policy_factory("gold") is policies.gold
    assert policies.load_policy_factory("qwen3_8_27b_code.policies:failing") is policies.failing
    with pytest.raises(ValueError, match="Unknown policy"):
        policies.load_policy_factory("does-not-exist")


def test_training_fixtures_can_be_collected_but_carry_no_hidden_verifier():
    """Fixture tasks are graded only by tests the agent can see, and the row
    says so, because that is weaker evidence than an external verifier."""
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    assert not task.has_hidden_verification
    result = collect([task], policies.gold, attempts_per_task=1)
    assert len(result.rows) == 1
    assert result.rows[0]["verification"]["hidden_verified"] is False
    assert result.report()["hidden_verified_rows"] == 0
