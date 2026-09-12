"""Contracts for the multi-file task families: medium-band pairs and long-band pipelines."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

from qwen3_8_27b_code import evaluation, long_horizon, policies
from qwen3_8_27b_code.collection import collect
from qwen3_8_27b_code.episodes import EpisodeBudget, run_episode
from qwen3_8_27b_code.tasks import gold_patch, materialise
from qwen3_8_27b_code.trajectories import unified_patch

ROOT = Path(__file__).resolve().parents[1]

ALL_FAMILIES = {**long_horizon.TRAINING_FAMILY_BUILDERS, **long_horizon.EVALUATION_FAMILY_BUILDERS}


def test_every_multi_file_task_needs_every_file_fixed():
    for family, builder in ALL_FAMILIES.items():
        task = builder(0)
        changed = task.gold_files
        assert len(changed) in (2, 4), family
        assert len(task.test_paths) == len(changed), family
        assert task.horizon == ("medium" if len(changed) == 2 else "long"), family

        with materialise(task) as workspace:
            before = workspace.verify()
            assert before.visible_exit != 0, family
            if task.has_hidden_verification:
                assert before.hidden["contract"] is False, family
                assert before.hidden["no_regression"] is True, family

        # Each file's fix on its own leaves the visible suite red.
        for path, content in changed.items():
            with materialise(task) as workspace:
                partial = unified_patch(path, task.files[path], content)
                assert workspace.harness.execute("apply_patch", {"patch": partial}) == "patch applied"
                half = workspace.verify()
                assert half.visible_exit != 0, (family, path)
                assert not half.succeeded, (family, path)

        with materialise(task) as workspace:
            assert workspace.harness.execute("apply_patch", {"patch": gold_patch(task)}) == "patch applied"
            after = workspace.verify()
            assert after.succeeded, (family, after.as_dict())
            assert not after.regression, family


def test_variants_change_names_but_not_the_bug_class():
    for family, builder in ALL_FAMILIES.items():
        first, second = builder(0), builder(1)
        assert first.task_id != second.task_id
        assert first.family == second.family == family
        # Deterministic: the same variant is the same task every time.
        assert builder(1).files == second.files


def test_hidden_verifiers_import_the_repository_from_outside_it():
    task = long_horizon.EVALUATION_FAMILY_BUILDERS["ledger"](0)
    for check in task.hidden_checks:
        assert "sys.path.insert(0, os.getcwd())" in check.source
    with materialise(task) as workspace:
        listing = workspace.harness.execute("list_files", {"path": "."})
        assert "contract" not in listing and "hidden" not in listing
        assert sorted(path.name for path in workspace.hidden_dir.iterdir()) == ["contract.py", "no_regression.py"]


def test_gold_walks_the_medium_and_long_bands_on_multi_file_tasks():
    suite = [builder(0) for builder in long_horizon.EVALUATION_FAMILY_BUILDERS.values()]
    report = evaluation.evaluate(suite, policies.gold, label="gold")
    card = report.scorecard()
    assert card["episode_success"] == 1.0
    # Two coupled-module tasks at seven calls, one four-stage pipeline at
    # seventeen: the band by calls made and the band each task was
    # designed for agree for the gold policy.
    assert card["horizon_bands"] == {"long": 1, "medium": 2}
    assert card["task_horizon_bands"] == {"long": 1, "medium": 2}
    assert card["success_by_task_horizon"] == {"long": 1.0, "medium": 1.0}
    calls = {record.family: record.tool_calls for record in report.records}
    assert calls == {"ledger": 7, "word_stats": 7, "invoice_pipeline": 17}


def test_pipeline_gold_repairs_a_stage_at_a_time_and_watches_the_suite():
    task = long_horizon.EVALUATION_FAMILY_BUILDERS["invoice_pipeline"](0)
    with materialise(task) as workspace:
        episode = run_episode(
            workspace.harness, policies.gold(task, 0), request=task.request, developer=task.developer
        )
        verdict = workspace.verify()
    assert verdict.succeeded
    assert episode.termination == "assistant_complete"
    assert episode.tool_calls == 17
    runs = [message["content"] for message in episode.messages if message.get("name") == "run_tests"]
    # Four verification runs; only the last is green, and the failure count
    # falls after every stage, which is the evidence a long-horizon policy
    # has to read rather than assume.
    assert len(runs) == 4
    assert [run.startswith("exit=0") for run in runs] == [False, False, False, True]
    failures = [int(re.search(r"(\d+) failed", run).group(1)) for run in runs[:-1]]
    assert failures == sorted(failures, reverse=True) and failures[0] > failures[-1]


def test_training_multi_file_tasks_collect_without_hidden_verification():
    result = collect(long_horizon.training_tasks(1), policies.gold, attempts_per_task=1)
    assert len(result.rows) == 3
    assert result.report()["rejections"] == {}
    calls = {}
    for row in result.rows:
        assert row["verification"]["hidden_verified"] is False
        assert row["provenance"]["task_id"].startswith("train/")
        calls[row["repo_family"]] = row["provenance"]["usage"]["tool_calls"]
    assert calls == {"quantity_pipeline": 7, "inventory": 7, "readings_pipeline": 17}


def test_default_budget_admits_the_long_band():
    # A budget below the long band excludes long tasks by construction.
    budget = EpisodeBudget()
    assert budget.tool_calls >= 30
    assert budget.wall_seconds >= 900
    assert evaluation.horizon_band(budget.tool_calls) == "long"


def test_deleting_one_of_several_test_files_is_still_tampering():
    task = long_horizon.training_tasks(1)[0]
    result = collect([task], policies.test_deleting, attempts_per_task=1)
    assert result.rows == []
    assert result.report()["rejections"] == {"protected_files_modified": 1}


def test_collector_training_suite_includes_the_multi_file_tasks():
    spec = importlib.util.spec_from_file_location("collect_trajectories", ROOT / "scripts" / "collect_trajectories.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [task.task_id for task in module.build_tasks("training", 1)]
    assert "train/quantity_pipeline-000" in ids
    assert "train/inventory-000" in ids
    assert "train/readings_pipeline-000" in ids
    assert any(task_id.startswith("fixture/") for task_id in ids)
    assert not any(task_id.startswith("eval/") for task_id in ids)
