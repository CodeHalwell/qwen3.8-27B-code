"""Contracts for outcome pairs across policies and the attempt records behind them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from datasets import Dataset, load_dataset

from qwen3_8_27b_code import distillation, fixtures, policies
from qwen3_8_27b_code.collection import collect, read_attempts, write_attempts
from qwen3_8_27b_code.episodes import TurnResult, answer_text, scripted_policy, tool_call_text
from qwen3_8_27b_code.tasks import gold_patch, task_from_fixture

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cell_containing(notebook, marker: str) -> str:
    return next(cell.source for cell in notebook.cells if cell.cell_type == "code" and marker in cell.source)


def test_outcome_pairs_prefer_the_verified_continuation_at_the_divergence():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    teacher = collect([task], policies.gold, attempts_per_task=1, policy_label="teacher")
    student = collect([task], policies.failing, attempts_per_task=1, policy_label="student")

    pairs = distillation.build_outcome_pairs(teacher.attempts + student.attempts)
    assert len(pairs) == 1
    pair = pairs[0]
    # Both read the module first; they diverge at the second turn, where the
    # teacher patches and the student reruns the tests.
    assert pair["evidence"]["turn_index"] == 1
    assert pair["chosen_message"]["tool_calls"][0]["function"]["name"] == "apply_patch"
    assert pair["rejected_message"]["tool_calls"][0]["function"]["name"] == "run_tests"
    assert pair["evidence"]["form_matched"] is True
    assert pair["contrast_type"] == "outcome:verification_failed"
    assert (pair["evidence"]["chosen_policy"], pair["evidence"]["rejected_policy"]) == ("teacher", "student")
    assert pair["evidence"]["chosen_verdict"]["succeeded"] and not pair["evidence"]["rejected_verdict"]["succeeded"]
    assert [message["role"] for message in pair["prompt_messages"]] == ["developer", "user", "assistant", "tool"]
    assert pair["chosen_reward"] == 1.0 and pair["rejected_reward"] == 0.0
    assert pair["id"].startswith("pref/fixture-bounds-000-outcome-t01-teacher-s3407-vs-student-s3407")


def test_outcome_pairs_cover_test_deletion_and_unverified_claims():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["config"](0))
    gold = collect([task], policies.gold, attempts_per_task=1, policy_label="gold")
    hacker = collect([task], policies.test_deleting, attempts_per_task=1, policy_label="hacker")
    unverified = collect([task], policies.unverified, attempts_per_task=1, policy_label="unverified")

    pairs = distillation.build_outcome_pairs(gold.attempts + hacker.attempts + unverified.attempts)
    by_type = {pair["contrast_type"]: pair for pair in pairs}
    assert set(by_type) == {"outcome:protected_files_modified", "outcome:completed_without_verification"}
    # The hacker runs the tests before reading anything: divergence at turn 0.
    assert by_type["outcome:protected_files_modified"]["evidence"]["turn_index"] == 0
    # The unverified attempt answers where the gold one runs the tests.
    claim = by_type["outcome:completed_without_verification"]
    assert claim["evidence"]["turn_index"] == 2
    assert claim["chosen_message"]["tool_calls"][0]["function"]["name"] == "run_tests"
    assert not claim["rejected_message"].get("tool_calls")
    assert claim["evidence"]["form_matched"] is False

    report = distillation.outcome_pairs_report(pairs)
    assert report["rows"] == 2
    assert report["form_matched_pairs"] == 1 and report["prose_rejections"] == 1
    assert report["policy_pairings"] == {"gold > hacker": 1, "gold > unverified": 1}
    # Form-matched pairs are ranked first under the per-task cap.
    assert pairs[0]["evidence"]["form_matched"] is True


def test_outcome_pairs_skip_infrastructure_failures_and_undiverged_prefixes():
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))

    def broken(task, seed):
        def policy(messages):
            raise RuntimeError("endpoint down")

        return policy

    def cut_off(task, seed):
        # Same first action as gold, then truncated: no divergent turn exists.
        return scripted_policy(
            [
                tool_call_text("read_file", {"path": task.module_path}, "Reading."),
            ],
            on_exhaustion=None,
        ) if False else _truncating(task)

    gold = collect([task], policies.gold, attempts_per_task=1, policy_label="gold")
    down = collect([task], broken, attempts_per_task=1, policy_label="down")
    truncated = collect([task], cut_off, attempts_per_task=1, policy_label="cut")
    assert down.report()["rejections"] == {"infrastructure_failure": 1}
    assert truncated.report()["rejections"] == {"terminated_output_truncated": 1}
    assert distillation.build_outcome_pairs(gold.attempts + down.attempts + truncated.attempts) == []


def _truncating(task):
    turns = iter(
        [
            TurnResult(text=tool_call_text("read_file", {"path": task.module_path}, "Reading.")),
            TurnResult(text="<think>\nstill", fault="output_truncated"),
        ]
    )
    return lambda messages: next(turns)


def test_attempts_round_trip_through_jsonl(tmp_path):
    task = task_from_fixture(fixtures.FAMILY_BUILDERS["bounds"](0))
    teacher = collect([task], policies.gold, attempts_per_task=1, policy_label="teacher")
    student = collect([task], policies.failing, attempts_per_task=1, policy_label="student")
    path = tmp_path / "attempts.jsonl"
    assert write_attempts(student, path) == 1

    restored = read_attempts(path)
    assert restored[0].policy == "student"
    assert restored[0].rejection == "verification_failed"
    assert restored[0].verified_success is False
    assert restored[0].episode.messages == student.attempts[0].episode.messages
    assert restored[0].as_dict() == student.attempts[0].as_dict()
    assert distillation.build_outcome_pairs(teacher.attempts + restored) == distillation.build_outcome_pairs(
        teacher.attempts + student.attempts
    )


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


def test_outcome_pairs_satisfy_notebook_04_after_an_arrow_round_trip(tmp_path):
    tasks = [task_from_fixture(fixtures.FAMILY_BUILDERS[family](0)) for family in ("bounds", "csv_fields")]
    teacher = collect(tasks, policies.gold, attempts_per_task=1, policy_label="teacher")
    student = collect(tasks, policies.unverified, attempts_per_task=1, policy_label="student")
    pairs = distillation.build_outcome_pairs(teacher.attempts + student.attempts)
    assert len(pairs) == 2
    corpus = tmp_path / "outcome_pairs.jsonl"
    distillation.write_outcome_pairs(pairs, corpus, tmp_path / "report.json")

    generator = load_generator()
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
    exec(code_cell_containing(generator.build_04_dpo(), "demo_preferences = Dataset.from_list"), namespace)
    assert len(namespace["preferences"]) == 2


def test_collector_cli_writes_attempts_and_self_play_outcome_pairs(tmp_path):
    out = tmp_path / "collected"
    command = [
        sys.executable, str(ROOT / "scripts" / "collect_trajectories.py"),
        "--policy", "gold", "--suite", "training", "--variants-per-family", "1", "--attempts", "1",
        "--out", str(out / "rows.jsonl"), "--report", str(out / "report.json"),
        "--length-pairs-out", str(out / "length.jsonl"), "--length-pairs-report", str(out / "length.json"),
        "--attempts-out", str(out / "attempts.jsonl"),
        "--outcome-pairs-out", str(out / "outcome.jsonl"), "--outcome-pairs-report", str(out / "outcome.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr[-2000:]
    attempts = read_attempts(out / "attempts.jsonl")
    assert len(attempts) == 14 and all(attempt.policy == "gold" for attempt in attempts)
    assert json.loads((out / "outcome.json").read_text())["rows"] == 0
    assert json.loads((out / "report.json").read_text())["policy"] == "gold"


def test_teacher_cli_probe_path_is_wired(monkeypatch):
    spec = importlib.util.spec_from_file_location("collect_from_teacher", ROOT / "scripts" / "collect_from_teacher.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv",
        ["collect_from_teacher.py", "--preset", "moonshot", "--model", "kimi-test", "--reasoning-effort", "low", "--probe"],
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", "x")
    probed = {}

    def fake_probe(config):
        probed["config"] = config
        return {"usable": False, "tool_call_returned": False}

    monkeypatch.setattr(module, "probe_teacher", fake_probe)
    assert module.main() == 1
    assert probed["config"].base_url == "https://api.moonshot.ai/v1"
    assert probed["config"].reasoning_effort == "low"
    assert module.slug("moonshotai/Kimi-K3") == "moonshotai-kimi-k3"


def test_notebook_08_distils_through_the_shared_harness():
    generator = load_generator()
    notebook = generator.build_08_distil()
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "from qwen3_8_27b_code.teachers import" in source
    assert "from qwen3_8_27b_code.collection import" in source
    assert "probe_teacher(" in source and "require_reasoning" in source
    assert "FastLanguageModel" not in source and "torch" not in source
    collection_cell = code_cell_containing(notebook, "result = collect(")
    assert "training_tasks(" in collection_cell and "task_from_fixture" in collection_cell
    assert "evaluation_tasks(" not in collection_cell
    assert "policy_label=" in collection_cell and "write_attempts(" in collection_cell
    assert "build_outcome_pairs(" in source and "REPLACE_WITH_TEACHER_MODEL_ID" in source
    generator.validate_notebook(notebook, Path("08.ipynb"))
