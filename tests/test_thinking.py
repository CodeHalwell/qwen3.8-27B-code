"""Contracts for the thinking budget: measurement, gate, selection, pairs, reward.

The objective is "shorter thinking at equal quality", and every test here
pins one half of that sentence: the quality half stays with the existing
gate, and nothing below may reward brevity without a verified success.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from datasets import Dataset, load_dataset
import pytest

from qwen3_8_27b_code import evaluation, fixtures, policies, thinking
from qwen3_8_27b_code.collection import collect
from qwen3_8_27b_code.episodes import (
    TurnResult,
    answer_text,
    run_episode,
    scripted_policy,
    tool_call_text,
)
from qwen3_8_27b_code.parsing import split_reasoning
from qwen3_8_27b_code.tasks import evaluation_tasks, gold_patch, materialise, task_from_fixture

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"

SMOKE_TASKS = evaluation_tasks(variants_per_family=1)


def load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counting(policy_factory, scale: int = 1):
    """Make a scripted policy report reasoning tokens, as a GPU policy would.

    Whitespace tokens stand in for tokenizer tokens; the point is that the
    count comes from the policy, which is the contract a real one must meet.
    """

    def factory(task, seed):
        policy = policy_factory(task, seed)

        def counted(messages):
            turn = policy(messages)
            reasoning, _ = split_reasoning(turn.text)
            return TurnResult(
                text=turn.text,
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=max(turn.completion_tokens, len(turn.text.split())),
                fault=turn.fault,
                reasoning_tokens=len(reasoning.split()) * scale,
            )

        return counted

    return factory


def verbose_on(turns: set[int], repeats: int = 12):
    """Gold actions with long reasoning on the given turns for seed 3407 only."""

    def factory(task, seed):
        def reasoning(index: int, text: str) -> str:
            if seed == 3407 and index in turns:
                return " ".join([text] * repeats)
            return text

        return scripted_policy(
            [
                tool_call_text("read_file", {"path": task.module_path}, reasoning(0, "I should read the implementation first.")),
                tool_call_text("apply_patch", {"patch": gold_patch(task)}, reasoning(1, "Patching the defect in place.")),
                tool_call_text("run_tests", {"profile": "unit"}, reasoning(2, "Verifying with the unit profile.")),
                answer_text("Fixed and verified."),
            ]
        )

    return factory


# --------------------------------------------------------------------------
# Measurement in the episode loop


def test_reasoning_tokens_are_accumulated_from_the_policy():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness, counting(policies.gold)(task, 0), request=task.request, developer=task.developer
        )
    assistant = [message for message in episode.messages if message["role"] == "assistant"]
    assert episode.reasoning_tokens_reported
    assert episode.reasoning_tokens == sum(len(message["reasoning_content"].split()) for message in assistant)
    assert episode.turn_reasoning_tokens == [len(message["reasoning_content"].split()) for message in assistant]
    assert episode.usage()["reasoning_tokens"] == episode.reasoning_tokens
    assert thinking.reasoning_length(episode) == (episode.reasoning_tokens, "tokens")


def test_unreported_reasoning_falls_back_to_characters():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness, policies.gold(task, 0), request=task.request, developer=task.developer
        )
    assistant = [message for message in episode.messages if message["role"] == "assistant"]
    assert not episode.reasoning_tokens_reported
    assert episode.reasoning_chars == sum(len(message["reasoning_content"]) for message in assistant) > 0
    assert thinking.reasoning_length(episode) == (episode.reasoning_chars, "chars")
    assert thinking.turn_reasoning_lengths(episode, "chars") == [
        len(message["reasoning_content"]) for message in assistant
    ]
    with pytest.raises(ValueError):
        thinking.turn_reasoning_lengths(episode, "tokens")


def test_truncation_inside_the_think_block_is_a_thinking_overrun():
    task = SMOKE_TASKS[0]
    with materialise(task) as workspace:
        overrun = run_episode(
            workspace.harness,
            lambda messages: TurnResult(
                text="<think>\nI keep going", completion_tokens=2048, reasoning_tokens=2048, fault="output_truncated"
            ),
            request=task.request,
            developer=task.developer,
        )
        cut_answer = run_episode(
            workspace.harness,
            lambda messages: TurnResult(
                text="<think>\nshort\n</think>\n\nThe answer is cut", completion_tokens=40, reasoning_tokens=3, fault="output_truncated"
            ),
            request=task.request,
            developer=task.developer,
        )
    assert overrun.termination == "output_truncated"
    assert overrun.thinking_overrun
    assert overrun.reasoning_tokens == 2048
    assert overrun.reasoning_chars == len("I keep going")
    assert not [message for message in overrun.messages if message["role"] == "assistant"]
    # Truncated after the think block closed: a long answer, not runaway thinking.
    assert cut_answer.termination == "output_truncated"
    assert not cut_answer.thinking_overrun
    assert cut_answer.reasoning_chars == len("short")


# --------------------------------------------------------------------------
# Scorecard, comparison and gate


def test_scorecard_reports_thinking_per_turn_and_horizon_bands():
    card = evaluation.evaluate(SMOKE_TASKS, counting(policies.gold), label="counted").scorecard()
    assert card["reasoning_tokens_reported"] is True
    assert card["reasoning_tokens_per_turn"] > 0
    assert 0 < card["reasoning_share_of_completion"] <= 1
    assert card["completion_tokens_per_turn"] >= card["reasoning_tokens_per_turn"]
    assert card["thinking_overrun_rate"] == 0.0
    # Six single-file tasks take three tool calls; the two multi-file tasks
    # take seven, which is the medium band the suite previously lacked.
    assert card["horizon_bands"] == {"medium": 2, "short": 6}
    assert card["success_by_horizon"] == {"medium": 1.0, "short": 1.0}


def test_thinking_budget_gate_fails_a_candidate_that_thinks_twice_as_much():
    baseline = evaluation.evaluate(SMOKE_TASKS[:3], counting(policies.gold), label="baseline")
    candidate = evaluation.evaluate(SMOKE_TASKS[:3], counting(policies.gold, scale=2), label="candidate")
    comparison = evaluation.compare(baseline, candidate)

    assert comparison["thinking"]["measured_in_tokens"] is True
    assert comparison["thinking"]["relative_change_tokens_per_turn"] == pytest.approx(1.0)
    checks = {check.name: check for check in evaluation.gate(comparison)}
    assert checks["thinking_budget"].passed is False
    assert "ceiling" in checks["thinking_budget"].detail
    # Quality is identical, so the only failing check is the budget.
    assert [name for name, check in checks.items() if not check.passed] == ["thinking_budget"]

    reverse = {check.name: check.passed for check in evaluation.gate(evaluation.compare(candidate, baseline))}
    assert reverse["thinking_budget"] is True
    generous = {check.name: check.passed for check in evaluation.gate(comparison, max_reasoning_growth=1.0)}
    assert generous["thinking_budget"] is True
    assert "thinking_budget" not in {check.name for check in evaluation.gate(comparison, max_reasoning_growth=None)}


def test_thinking_budget_is_unmeasured_rather_than_failed_for_uncounted_policies():
    baseline = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="scripted")
    candidate = evaluation.evaluate(SMOKE_TASKS[:1], counting(policies.gold), label="counted")
    comparison = evaluation.compare(baseline, candidate)
    assert comparison["thinking"]["measured_in_tokens"] is False
    assert comparison["thinking"]["relative_change_chars_per_turn"] == 0.0
    check = {check.name: check for check in evaluation.gate(comparison)}["thinking_budget"]
    assert check.passed is True
    assert "not measured" in check.detail


def test_thinking_overruns_fail_the_gate():
    def overrunning(task, seed):
        return lambda messages: TurnResult(
            text="<think>\nstill thinking", completion_tokens=2048, reasoning_tokens=2048, fault="output_truncated"
        )

    baseline = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="baseline")
    candidate = evaluation.evaluate(SMOKE_TASKS[:1], overrunning, label="overrun")
    assert candidate.scorecard()["thinking_overrun_rate"] == 1.0
    checks = {check.name: check.passed for check in evaluation.gate(evaluation.compare(baseline, candidate))}
    assert checks["thinking_overrun_no_worse"] is False


def test_reports_written_before_thinking_metrics_still_load():
    payload = evaluation.evaluate(SMOKE_TASKS[:1], policies.gold, label="old").as_dict()
    for row in payload["attempts"]:
        for key in ("turns", "reasoning_tokens", "reasoning_chars", "reasoning_tokens_reported", "thinking_overrun", "horizon_band"):
            row.pop(key)
    restored = evaluation.EvaluationReport.from_dict(payload)
    card = restored.scorecard()
    assert card["episode_success"] == 1.0
    assert card["reasoning_tokens_reported"] is False
    assert card["reasoning_tokens_per_turn"] == 0.0


# --------------------------------------------------------------------------
# Selection in the collector


def test_collection_keeps_the_verified_attempt_that_thought_least():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))

    shortest = collect([task], verbose_on({0}), attempts_per_task=2, seeds=(3407, 9176))
    assert [row["provenance"]["seed"] for row in shortest.rows] == [9176]
    assert shortest.rows[0]["provenance"]["selection"] == "shortest_reasoning"
    assert shortest.rows[0]["provenance"]["reasoning_unit"] == "chars"
    report = shortest.report()
    # Same actions, so the verbose copy is the duplicate, not the kept row.
    assert report["rejections"] == {"duplicate_actions": 1}
    assert report["thinking"]["per_turn"]["kept_rows"] < report["thinking"]["per_turn"]["verified"]
    assert report["thinking"]["unit"] == "chars"

    first = collect([task], verbose_on({0}), attempts_per_task=2, seeds=(3407, 9176), selection="first")
    assert [row["provenance"]["seed"] for row in first.rows] == [3407]

    with pytest.raises(ValueError, match="selection"):
        collect([task], verbose_on({0}), attempts_per_task=1, selection="longest")


# --------------------------------------------------------------------------
# Reasoning-length preference pairs


def test_reasoning_length_pairs_prefer_less_thinking_before_the_same_action():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    result = collect([task], verbose_on({0}), attempts_per_task=2, seeds=(3407, 9176))
    pairs = thinking.build_reasoning_length_pairs(result.attempts)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["contrast_type"] == "reasoning_length"
    assert pair["repo_family"] == "bounds"
    assert pair["prompt_messages"] == result.attempts[1].episode.messages[:2]
    assert pair["chosen_message"]["reasoning_content"] == "I should read the implementation first."
    assert pair["rejected_message"]["reasoning_content"].count("I should read") == 12
    # Same action on both sides: the pair is about thinking, nothing else.
    assert pair["chosen_message"]["tool_calls"] == pair["rejected_message"]["tool_calls"]
    assert pair["chosen_reward"] == 1.0 > pair["rejected_reward"] > 0.0
    assert pair["evidence"]["unit"] == "chars"
    assert pair["evidence"]["turn_index"] == 0
    assert pair["evidence"]["chosen_seed"] == 9176
    assert pair["evidence"]["rejected_seed"] == 3407
    assert pair["evidence"]["ratio"] > thinking.DEFAULT_MIN_LENGTH_RATIO
    assert pair["evidence"]["hidden_verified"] is False


def test_reasoning_length_pairs_walk_the_shared_action_prefix():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["config"](0))
    result = collect([task], verbose_on({0, 2}), attempts_per_task=2, seeds=(3407, 9176))
    pairs = thinking.build_reasoning_length_pairs(result.attempts)

    assert sorted(pair["evidence"]["turn_index"] for pair in pairs) == [0, 2]
    later = next(pair for pair in pairs if pair["evidence"]["turn_index"] == 2)
    # The prompt is the chosen trajectory's real prefix, observations included.
    assert [message["role"] for message in later["prompt_messages"]] == [
        "developer", "user", "assistant", "tool", "assistant", "tool",
    ]
    assert later["prompt_messages"][-1]["content"] == "patch applied"
    assert later["evidence"]["shared_action_turns"] == 2
    assert later["chosen_message"]["tool_calls"][0]["function"]["name"] == "run_tests"

    report = thinking.length_pairs_report(pairs)
    assert report["rows"] == 2
    assert report["pairs_beyond_first_turn"] == 1
    assert report["units"] == {"chars": 2}


def test_reasoning_length_pairs_exclude_unverified_attempts():
    def mixed(task, seed):
        if seed == 3407:
            return verbose_on({0})(task, seed)
        # Right patch, never verified: not a success, so never a pair member.
        return scripted_policy(
            [
                tool_call_text("read_file", {"path": task.module_path}, "Reading."),
                tool_call_text("apply_patch", {"patch": gold_patch(task)}, "Patching."),
                answer_text("Obviously correct, no need to run the tests."),
            ]
        )

    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    result = collect([task], mixed, attempts_per_task=2, seeds=(3407, 9176))
    assert result.report()["rejections"] == {"completed_without_verification": 1}
    assert thinking.build_reasoning_length_pairs(result.attempts) == []


def test_reasoning_length_pairs_use_tokens_when_the_policy_counted_them():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    result = collect([task], counting(verbose_on({0})), attempts_per_task=2, seeds=(3407, 9176))
    pairs = thinking.build_reasoning_length_pairs(result.attempts)
    assert len(pairs) == 1
    evidence = pairs[0]["evidence"]
    assert evidence["unit"] == "tokens"
    assert evidence["chosen_reasoning_length"] == len("I should read the implementation first.".split())
    assert evidence["rejected_reasoning_length"] == 12 * evidence["chosen_reasoning_length"]


def _assert_no_none_leaves(value, path="message"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert item is not None, f"null leaked into rendered {path}.{key}"
            _assert_no_none_leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_none_leaves(item, f"{path}[{index}]")


class _NullRejectingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        _assert_no_none_leaves(messages)
        return "<render>"


def test_length_pairs_satisfy_notebook_04_after_an_arrow_round_trip(tmp_path):
    """The pairs must load through the same cell the corpus pairs use."""
    tasks = [
        task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0)),
        task_from_fixture(fixtures.FAMILY_BUILDERS["csv_fields"](0)),
    ]
    result = collect(tasks, verbose_on({0, 1}), attempts_per_task=2, seeds=(3407, 9176))
    pairs = thinking.build_reasoning_length_pairs(result.attempts)
    assert len(pairs) == 4
    corpus, report_path = tmp_path / "length_pairs.jsonl", tmp_path / "report.json"
    report = thinking.write_length_pairs(pairs, corpus, report_path)
    assert report["families"] == {"bounds": 2, "csv_fields": 2}
    assert json.loads(report_path.read_text())["rows"] == 4

    generator = load_generator()
    notebook_04 = generator.build_04_dpo()
    namespace = {
        "json": json,
        "hashlib": __import__("hashlib"),
        "Dataset": Dataset,
        "load_dataset": load_dataset,
        "DEMO_MODE": False,
        "PREFERENCE_LOCAL_JSONL": str(corpus),
        "PREFERENCE_DATASET_ID": "unused",
        "PREFERENCE_DATASET_REVISION": "main",
        "hf_token": None,
        "tokenizer": _NullRejectingTokenizer(),
    }
    exec(generator.TOOLS_CELL, namespace)
    cell = next(
        cell.source
        for cell in notebook_04.cells
        if cell.cell_type == "code" and "demo_preferences = Dataset.from_list" in cell.source
    )
    exec(cell, namespace)
    assert len(namespace["preferences"]) == 4
    assert set(namespace["preferences"].column_names) == {"prompt", "chosen", "rejected"}
    assert len(namespace["split"]["train"]) + len(namespace["split"]["test"]) == 4


# --------------------------------------------------------------------------
# The RL brevity term


def test_length_rewards_are_gated_on_correctness():
    rewards = thinking.length_rewards([100, 300, 200, 50], [True, True, False, False], weight=0.1)
    # Correct: linear from +0.05 (shortest in group) to -0.05 (longest).
    assert rewards[0] == pytest.approx(0.1 * (0.5 - 50 / 250))
    assert rewards[1] == pytest.approx(-0.05)
    # Incorrect: never positive, even for the shortest sample in the group.
    assert rewards[2] == pytest.approx(0.1 * (0.5 - 150 / 250))
    assert rewards[3] == 0.0
    assert thinking.length_rewards([80, 80, 80], [True, True, False]) == [0.0, 0.0, 0.0]
    assert thinking.length_rewards([], []) == []
    with pytest.raises(ValueError):
        thinking.length_rewards([1, 2], [True])


def test_length_rewards_cannot_lift_an_incorrect_sample_over_a_correct_one():
    lengths = [50, 400, 90, 1000, 120]
    succeeded = [False, True, True, True, False]
    brevity = thinking.length_rewards(lengths, succeeded, weight=0.1)
    correctness = [0.8 if ok else 0.2 for ok in succeeded]
    totals = [c + b for c, b in zip(correctness, brevity)]
    correct = [total for total, ok in zip(totals, succeeded) if ok]
    incorrect = [total for total, ok in zip(totals, succeeded) if not ok]
    assert min(correct) > max(incorrect)
    # Among the correct samples the shortest is preferred.
    assert totals[2] > totals[1] > totals[3]


# --------------------------------------------------------------------------
# Notebook parity: the generated cells must carry the same contract


def code_cell_containing(notebook, marker: str) -> str:
    return next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    )


def test_notebook_05_brevity_reward_matches_the_package():
    generator = load_generator()
    cell = code_cell_containing(generator.build_05_grpo(), "def length_rewards")
    namespace: dict = {}
    exec(cell, namespace)  # the cell runs its own fixtures
    probes = [
        ([100, 300, 200, 50], [True, True, False, False]),
        ([7, 7, 7], [True, False, True]),
        ([], []),
        ([10, 20], [False, False]),
        ([1, 1000], [True, True]),
    ]
    for lengths, succeeded in probes:
        assert namespace["length_rewards"](lengths, succeeded) == thinking.length_rewards(lengths, succeeded)


def test_notebook_07_counts_reasoning_tokens_and_collects_for_brevity():
    generator = load_generator()
    notebook = generator.build_07_collect_and_evaluate()

    policy_cell = code_cell_containing(notebook, "def build_policy_factory")
    assert "think_end_id" in policy_cell
    assert "reasoning_tokens=reasoning_tokens" in policy_cell
    # A turn that never closes its think block is all reasoning, not zero.
    assert "reasoning_tokens = completion_tokens" in policy_cell

    collection_cell = code_cell_containing(notebook, "result = collect(")
    assert 'selection="shortest_reasoning"' in collection_cell
    assert "training_tasks(" in collection_cell
    assert "build_reasoning_length_pairs(" in collection_cell
    assert "evaluation_tasks(" not in collection_cell

    gate_cell = code_cell_containing(notebook, 'comparison["gate_passed"]')
    assert "max_reasoning_growth=MAX_REASONING_GROWTH" in gate_cell
    assert 'comparison["thinking"]' in gate_cell
