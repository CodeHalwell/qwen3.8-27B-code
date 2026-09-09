"""Contracts for the multi-file, medium-horizon task families."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from qwen3_8_27b_code import evaluation, long_horizon, policies
from qwen3_8_27b_code.collection import collect
from qwen3_8_27b_code.tasks import gold_patch, materialise
from qwen3_8_27b_code.trajectories import unified_patch

ROOT = Path(__file__).resolve().parents[1]

ALL_FAMILIES = {**long_horizon.TRAINING_FAMILY_BUILDERS, **long_horizon.EVALUATION_FAMILY_BUILDERS}


def test_every_multi_file_task_needs_both_files_fixed():
    for family, builder in ALL_FAMILIES.items():
        task = builder(0)
        changed = task.gold_files
        assert len(changed) == 2, family
        assert len(task.test_paths) == 2, family

        with materialise(task) as workspace:
            before = workspace.verify()
            assert before.visible_exit != 0, family
            if task.has_hidden_verification:
                assert before.hidden["contract"] is False, family
                assert before.hidden["no_regression"] is True, family

        # Each half of the fix on its own leaves the visible suite red.
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


def test_gold_walks_the_medium_band_on_multi_file_tasks():
    suite = [builder(0) for builder in long_horizon.EVALUATION_FAMILY_BUILDERS.values()]
    card = evaluation.evaluate(suite, policies.gold, label="gold").scorecard()
    assert card["episode_success"] == 1.0
    assert card["horizon_bands"] == {"medium": 2}
    assert card["calls_per_success"] == 7.0


def test_training_multi_file_tasks_collect_without_hidden_verification():
    result = collect(long_horizon.training_tasks(1), policies.gold, attempts_per_task=1)
    assert len(result.rows) == 2
    assert result.report()["rejections"] == {}
    for row in result.rows:
        assert row["verification"]["hidden_verified"] is False
        assert row["provenance"]["task_id"].startswith("train/")
        assert row["provenance"]["usage"]["tool_calls"] == 7


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
    assert any(task_id.startswith("fixture/") for task_id in ids)
    assert not any(task_id.startswith("eval/") for task_id in ids)
