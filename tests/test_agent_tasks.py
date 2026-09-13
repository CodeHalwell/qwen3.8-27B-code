"""Contracts for the episode loop, the collector and the evaluation gate.

These run the real harness, real pytest and real hidden verifiers against the
held-out fixtures. The only thing faked is the model: a scripted policy stands
in for generation, which is what lets the whole collection and gating path be
exercised on CPU.
"""

from __future__ import annotations

import argparse
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
from qwen3_8_27b_code.tasks import (
    evaluation_family_builders,
    evaluation_tasks,
    gold_patch,
    materialise,
    task_from_fixture,
)
from qwen3_8_27b_code.trajectories import unified_patch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"

# One variant per family keeps these tests to nine real repositories: six
# single-file families and the three multi-file ones from long_horizon.
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
    from qwen3_8_27b_code import long_horizon

    training = set(fixtures.FAMILY_BUILDERS) | set(long_horizon.TRAINING_FAMILY_BUILDERS)
    held_out = set(evaluation_family_builders())
    assert training & held_out == set()
    assert set(tasks.EVALUATION_FAMILY_BUILDERS) < held_out
    assert set(long_horizon.EVALUATION_FAMILY_BUILDERS) < held_out
    assert len(held_out) >= 8


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

            assert workspace.harness.execute("apply_patch", {"patch": gold_patch(task)}) == "patch applied"
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
        visible = "\n".join(task.files[path] for path in task.test_paths)
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
    assert patch == gold_patch(task)
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
    # The effort travels with every attempt, kept or not, so the pair
    # builders can render each pair under the instruction it ran at.
    assert {attempt.reasoning_effort for attempt in result.attempts} == {"xhigh"}

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
    assert failed == ["episode_success", "task_level_not_net_negative", "task_horizon_no_worse"]


def test_gate_fails_when_a_horizon_band_drops_while_the_aggregate_holds():
    """Brevity or any other change may not be paid for with the long tasks."""
    long_ids = {task.task_id for task in SMOKE_TASKS if task.horizon == "long"}
    short_ids = {task.task_id for task in SMOKE_TASKS if task.horizon == "short"}
    assert long_ids and short_ids
    sacrificed_short = sorted(short_ids)[0]

    def loses_long(task, seed):
        return policies.failing(task, seed) if task.task_id in long_ids else policies.gold(task, seed)

    def loses_one_short(task, seed):
        return policies.failing(task, seed) if task.task_id == sacrificed_short else policies.gold(task, seed)

    baseline = evaluation.evaluate(SMOKE_TASKS, loses_one_short, label="baseline")
    candidate = evaluation.evaluate(SMOKE_TASKS, loses_long, label="candidate")
    comparison = evaluation.compare(baseline, candidate)
    # Same aggregate, one win and one loss: every aggregate check passes.
    assert comparison["deltas"]["episode_success"] == 0.0
    assert comparison["task_level"]["wins"] == comparison["task_level"]["losses"] == 1
    checks = {check.name: check for check in evaluation.gate(comparison)}
    assert checks["episode_success"].passed and checks["task_level_not_net_negative"].passed
    assert checks["task_horizon_no_worse"].passed is False
    assert "long" in checks["task_horizon_no_worse"].detail
    assert comparison["thinking"]["candidate_success_by_task_horizon"]["long"] == 0.0


def test_scorecard_reports_context_pressure():
    def widening(task, seed):
        turns = iter([
            (tool_call_text("read_file", {"path": task.module_path}, "Reading."), 1_000),
            (answer_text("Done."), 9_000),
        ])

        def policy(messages):
            text, prompt_tokens = next(turns)
            return TurnResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=10)

        return policy

    def exhausted(task, seed):
        return lambda messages: TurnResult(text="", fault="context_budget", prompt_tokens=32_000)

    widened = evaluation.evaluate(SMOKE_TASKS[:1], widening, label="widening")
    assert widened.records[0].peak_prompt_tokens == 9_000
    assert widened.scorecard()["peak_prompt_tokens_max"] == 9_000
    assert widened.scorecard()["context_budget_rate"] == 0.0
    out_of_room = evaluation.evaluate(SMOKE_TASKS[:2], exhausted, label="exhausted").scorecard()
    assert out_of_room["context_budget_rate"] == 1.0
    assert out_of_room["terminations"] == {"context_budget": 2}


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
    assert {record.task_horizon for record in restored.records} == {"short"}


def test_every_task_declares_the_horizon_it_was_designed_for():
    horizons = {task.family: task.horizon for task in evaluation_tasks(variants_per_family=1)}
    assert set(horizons.values()) == {"short", "medium", "long"}
    assert horizons["invoice_pipeline"] == "long"
    assert horizons["ledger"] == "medium"
    assert horizons["word_wrap"] == "short"
    assert task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0)).horizon == "short"


def test_effort_ladder_picks_the_cheapest_rung_that_keeps_the_best_success():
    def scaled(scale):
        def factory(task, seed):
            policy = policies.gold(task, seed)

            def counted(messages):
                turn = policy(messages)
                reasoning = turn.text.split("</think>")[0]
                return TurnResult(
                    text=turn.text,
                    completion_tokens=len(turn.text.split()),
                    reasoning_tokens=len(reasoning.split()) * scale,
                )

            return counted

        return factory

    suite = SMOKE_TASKS[:3]
    reports = {
        "low": evaluation.evaluate(suite, scaled(1), label="low"),
        "medium": evaluation.evaluate(suite, scaled(3), label="medium"),
        "xhigh": evaluation.evaluate(suite, scaled(9), label="xhigh"),
    }
    ladder = evaluation.effort_ladder(reports)
    assert ladder["unit"] == "tokens"
    assert list(ladder["rungs"]) == ["low", "medium", "xhigh"]
    assert ladder["best_success"] == 1.0
    assert ladder["eligible"] == ["low", "medium", "xhigh"]
    assert ladder["recommended"] == "low"
    low, xhigh = ladder["rungs"]["low"], ladder["rungs"]["xhigh"]
    assert xhigh["reasoning_tokens_per_turn"] == pytest.approx(9 * low["reasoning_tokens_per_turn"])

    # A cheaper rung that loses tasks is not eligible, whatever it saves.
    reports["low"] = evaluation.evaluate(suite, policies.failing, label="low-fails")
    ladder = evaluation.effort_ladder(reports)
    assert ladder["eligible"] == ["medium", "xhigh"]
    assert ladder["recommended"] == "medium"
    # Unless the tolerance says that loss is acceptable.
    assert evaluation.effort_ladder(reports, success_tolerance=1.0)["recommended"] == "low"

    with pytest.raises(ValueError, match="unknown reasoning effort"):
        evaluation.effort_ladder({"high": reports["medium"]})
    with pytest.raises(ValueError, match="same tasks"):
        evaluation.effort_ladder({
            "low": reports["medium"],
            "medium": evaluation.evaluate(SMOKE_TASKS[3:5], policies.gold, label="other"),
        })
    # The same tasks with different attempts or seeds are a different
    # experiment: success from unequal samples must not pick the effort.
    with pytest.raises(ValueError, match="same attempts and seeds"):
        evaluation.effort_ladder({
            "low": reports["medium"],
            "medium": evaluation.evaluate(suite, scaled(3), label="two-seeds", attempts_per_task=2),
        })
    with pytest.raises(ValueError, match="same attempts and seeds"):
        evaluation.effort_ladder({
            "low": reports["medium"],
            "medium": evaluation.evaluate(suite, scaled(3), label="other-seed", seeds=(1,)),
        })

    # An attempt lost to the harness is excluded from every rate, so equal
    # attempted samples are not equal scored samples: the rung is refused
    # rather than ranked on an inflated success.
    def broken_once(task, seed):
        if task.task_id == suite[0].task_id:
            def policy(messages):
                raise RuntimeError("model server went away")
            return policy
        return scaled(3)(task, seed)

    lost_one = evaluation.evaluate(suite, broken_once, label="lost-one")
    assert lost_one.attempt_signature(scored_only=False) == reports["medium"].attempt_signature(scored_only=False)
    assert lost_one.attempt_signature() != reports["medium"].attempt_signature()
    with pytest.raises(ValueError, match="infrastructure failures"):
        evaluation.effort_ladder({"low": reports["medium"], "medium": lost_one})


def test_effort_ladder_never_recommends_a_rung_that_loses_a_band():
    """Equal or tolerable aggregate success does not excuse losing the pipeline."""
    long_task = next(task for task in SMOKE_TASKS if task.horizon == "long")
    short_a, short_b = [task for task in SMOKE_TASKS if task.horizon == "short"][:2]
    suite = [short_a, short_b, long_task]

    def solving(solved, scale):
        def factory(task, seed):
            policy = (policies.gold if task.task_id in solved else policies.failing)(task, seed)

            def counted(messages):
                turn = policy(messages)
                return TurnResult(text=turn.text, completion_tokens=50, reasoning_tokens=10 * scale)

            return counted

        return factory

    everything = {task.task_id for task in suite}
    # low solves both short tasks and loses the pipeline; medium solves all
    # three at three times the reasoning.
    reports = {
        "low": evaluation.evaluate(suite, solving({short_a.task_id, short_b.task_id}, 1), label="low"),
        "medium": evaluation.evaluate(suite, solving(everything, 3), label="medium"),
    }
    strict = evaluation.effort_ladder(reports)
    assert strict["recommended"] == "medium"
    # A tolerance that would excuse the aggregate loss still does not
    # excuse losing the whole long band.
    tolerant = evaluation.effort_ladder(reports, success_tolerance=0.34)
    assert tolerant["eligible"] == ["medium"]
    assert tolerant["recommended"] == "medium"
    assert tolerant["best_success_by_band"] == {"long": 1.0, "short": 1.0}

    # Two rungs with equal aggregate success that split the bands between
    # them: neither keeps every band, so nothing is recommended.
    reports["medium"] = evaluation.evaluate(suite, solving({short_a.task_id, long_task.task_id}, 3), label="medium")
    split = evaluation.effort_ladder(reports)
    assert split["rungs"]["low"]["episode_success"] == split["rungs"]["medium"]["episode_success"]
    assert split["eligible"] == []
    assert split["recommended"] is None
    assert "read the rungs by band" in split["note"]


def test_horizon_gate_refuses_reports_that_scored_different_tasks():
    """A candidate that skipped the long tasks cannot pass the band check by omission."""
    baseline = evaluation.evaluate(SMOKE_TASKS, policies.gold, label="baseline")
    narrower = [task for task in SMOKE_TASKS if task.horizon != "long"]
    candidate = evaluation.evaluate(narrower, policies.gold, label="narrow")
    comparison = evaluation.compare(baseline, candidate)
    check = {check.name: check for check in evaluation.gate(comparison)}["task_horizon_no_worse"]
    assert check.passed is False
    assert "task membership differs" in check.detail
    assert all(
        entry["baseline_task_horizon"] == entry["candidate_task_horizon"] != ""
        for entry in comparison["paired_tasks"].values()
    )

    # A band lost entirely to infrastructure failures is an unmatched task
    # on that side, not a silently absent band.
    def broken_on_long(task, seed):
        if task.horizon == "long":
            def policy(messages):
                raise RuntimeError("model server went away")
            return policy
        return policies.gold(task, seed)

    partial = evaluation.evaluate(SMOKE_TASKS, broken_on_long, label="partial")
    check = {check.name: check for check in evaluation.gate(evaluation.compare(baseline, partial))}["task_horizon_no_worse"]
    assert check.passed is False
    assert "only in baseline" in check.detail

    # An attempt lost to the harness on one side leaves that task's success
    # rate intact and its coverage hollow; uneven attempt counts fail too.
    trio = [task for task in SMOKE_TASKS if task.horizon == "short"][:2] + [
        task for task in SMOKE_TASKS if task.horizon == "long"
    ]

    def flaky_on_long(task, seed):
        if task.horizon == "long" and seed == 9176:
            def policy(messages):
                raise RuntimeError("model server went away")
            return policy
        return policies.gold(task, seed)

    steady = evaluation.evaluate(trio, policies.gold, label="steady", attempts_per_task=2)
    flaky = evaluation.evaluate(trio, flaky_on_long, label="flaky", attempts_per_task=2)
    comparison = evaluation.compare(steady, flaky)
    assert comparison["deltas"]["episode_success"] == 0.0
    check = {check.name: check for check in evaluation.gate(comparison)}["task_horizon_no_worse"]
    assert check.passed is False
    assert "attempt counts differ" in check.detail

    # Reports that predate the label are reported as unmeasured, not failed;
    # reports that label a shared task differently are a changed suite.
    unlabelled = evaluation.EvaluationReport.from_dict({
        **baseline.as_dict(),
        "attempts": [{**row, "task_horizon": ""} for row in baseline.as_dict()["attempts"]],
    })
    check = {check.name: check for check in evaluation.gate(evaluation.compare(unlabelled, baseline))}["task_horizon_no_worse"]
    assert check.passed is True and "not measured" in check.detail
    relabelled = evaluation.EvaluationReport.from_dict({
        **baseline.as_dict(),
        "attempts": [{**row, "task_horizon": "short"} for row in baseline.as_dict()["attempts"]],
    })
    check = {check.name: check for check in evaluation.gate(evaluation.compare(baseline, relabelled))}["task_horizon_no_worse"]
    assert check.passed is False and "labels disagree" in check.detail


def test_comparing_reports_with_no_shared_tasks_is_an_error():
    first = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="a")
    second = evaluation.evaluate(SMOKE_TASKS[1:2], policies.gold, label="b")
    with pytest.raises(ValueError, match="share no tasks"):
        evaluation.compare(first, second)


def _provenance(model: str, **overrides) -> dict:
    return {
        "model": model,
        "harness_revision": "abc123",
        "reasoning_effort": "medium",
        "max_new_tokens": 4096,
        "max_sequence_length": 32768,
        "episode_budget": {"tool_calls": 30, "wall_seconds": 900.0},
        "attempts_per_task": 1,
        "variants_per_family": 1,
        **overrides,
    }


def test_report_provenance_round_trips_through_the_report_file(tmp_path):
    report = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="a")
    report.metadata = _provenance("unsloth/Qwen3.8-27B")
    payload = evaluation.write_report(report, tmp_path / "a.json")
    assert payload["metadata"] == report.metadata
    assert evaluation.read_report(tmp_path / "a.json").metadata == report.metadata
    # A report written before provenance existed still loads, empty.
    (tmp_path / "old.json").write_text(json.dumps({k: v for k, v in payload.items() if k != "metadata"}))
    assert evaluation.read_report(tmp_path / "old.json").metadata == {}


def test_build_provenance_records_every_key_the_pairing_check_requires():
    provenance = evaluation.build_provenance(
        model="gold",
        harness_revision="abc123",
        reasoning_effort="medium",
        max_new_tokens=None,
        max_sequence_length=None,
        episode_budget=EpisodeBudget(tool_calls=30, wall_seconds=900.0),
        attempts_per_task=1,
        variants_per_family=1,
        measured_at="2026-09-13T00:00:00+00:00",
    )
    assert set(evaluation.PROVENANCE_REQUIRED_KEYS) <= set(provenance)
    assert provenance["episode_budget"] == {"tool_calls": 30, "wall_seconds": 900.0}
    assert provenance["measured_at"] == "2026-09-13T00:00:00+00:00"


def test_cli_compare_refuses_reports_without_matching_provenance(tmp_path, capsys):
    """The CLI gate applies the same pairing check as notebook 07."""
    spec = importlib.util.spec_from_file_location("evaluate_agent", ROOT / "scripts" / "evaluate_agent.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def report(label: str, model: str | None) -> Path:
        scored = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label=label)
        if model is not None:
            scored.metadata = _provenance(model)
        return Path(evaluation.write_report(scored, tmp_path / f"{label}.json") and tmp_path / f"{label}.json")

    def compare(baseline: Path, candidate: Path) -> int:
        arguments = argparse.Namespace(
            baseline=baseline, candidate=candidate, minimum_success_delta=0.0,
            max_reasoning_growth=0.1, ignore_thinking_budget=False, out=tmp_path / "comparison.json",
        )
        return cli.run_compare(arguments)

    assert compare(report("bare-a", None), report("bare-b", None)) == 2
    assert "GATE NOT RUN" in capsys.readouterr().out
    assert not (tmp_path / "comparison.json").exists()

    assert compare(report("same-a", "gold"), report("same-b", "gold")) == 2
    assert "both reports measure 'gold'" in capsys.readouterr().out

    status = compare(report("base", "gold"), report("cand", "gold-v2"))
    assert status in (0, 1)
    written = json.loads((tmp_path / "comparison.json").read_text())
    assert written["provenance"]["baseline"]["model"] == "gold"
    assert written["provenance"]["candidate"]["model"] == "gold-v2"


def test_gate_pairing_refuses_reports_measured_differently():
    """A stale report pulled from storage must not be gated against a fresh
    one unless both record the same measurement settings."""
    baseline = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="baseline")
    candidate = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="candidate")

    blocking, advisory = evaluation.pairing_problems(baseline, candidate)
    assert len(blocking) == 2 and all("no provenance" in problem for problem in blocking)

    baseline.metadata = _provenance("unsloth/Qwen3.8-27B")
    candidate.metadata = _provenance("me/adapter@deadbeef")
    assert evaluation.pairing_problems(baseline, candidate) == ([], [])

    # A partial record is no evidence: a missing setting cannot "match".
    partial = _provenance("me/adapter@deadbeef")
    del partial["max_new_tokens"], partial["model"]
    candidate.metadata = partial
    blocking, _ = evaluation.pairing_problems(baseline, candidate)
    assert blocking == ["candidate report 'candidate' lacks provenance keys ['model', 'max_new_tokens']; re-measure it"]

    # Same model on both sides is a baseline paired with itself.
    candidate.metadata = _provenance("unsloth/Qwen3.8-27B")
    blocking, _ = evaluation.pairing_problems(baseline, candidate)
    assert blocking and "both reports measure" in blocking[0]

    # A different effort, cap, budget or attempt count is a different experiment.
    for key, value in (
        ("reasoning_effort", "low"),
        ("max_new_tokens", 2048),
        ("episode_budget", {"tool_calls": 10, "wall_seconds": 900.0}),
        ("attempts_per_task", 3),
    ):
        candidate.metadata = _provenance("me/adapter@deadbeef", **{key: value})
        blocking, _ = evaluation.pairing_problems(baseline, candidate)
        assert blocking == [f"{key}: baseline {baseline.metadata[key]!r}, candidate {value!r}"]

    # A harness revision that moved is recorded, not fatal: compare() already
    # refuses a different task set.
    candidate.metadata = _provenance("me/adapter@deadbeef", harness_revision="def456")
    blocking, advisory = evaluation.pairing_problems(baseline, candidate)
    assert blocking == []
    assert advisory == ["harness_revision: baseline 'abc123', candidate 'def456'"]


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
