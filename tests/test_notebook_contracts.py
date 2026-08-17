"""Regression tests for contracts embedded in the generated Colab notebooks."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote

from datasets import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cell_containing(notebook, marker: str) -> str:
    return next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    )


class FakeTokenizer:
    """Minimal tokenizer surface for data-contract cells that do not need a model."""

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "<tool_call>fixture</tool_call>\n<tool_response>ok</tool_response>"

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split())))}


def test_arrow_round_trip_preserves_semantic_tool_schema():
    generator = load_generator()
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)

    expected_tools = namespace["TOOLS"]
    round_tripped = Dataset.from_list([{"tools": expected_tools}])[0]["tools"]

    assert namespace["canonical_tool_schema"](round_tripped) == namespace["TOOL_SCHEMA_JSON"]
    changed = deepcopy(round_tripped)
    changed[0]["function"]["description"] = "Different semantics"
    assert namespace["canonical_tool_schema"](changed) != namespace["TOOL_SCHEMA_JSON"]


def test_rendered_tool_block_preserves_semantic_tool_schema():
    generator = load_generator()
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)

    rendered = (
        "<|im_start|>system\n<tools>\n"
        + "\n".join(json.dumps(tool) for tool in namespace["TOOLS"])
        + "\n</tools><|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert namespace["rendered_tool_schema"](rendered) == namespace["TOOL_SCHEMA_JSON"]


def test_repository_family_split_always_has_two_nonempty_partitions():
    generator = load_generator()
    split_cell = code_cell_containing(generator.build_02_data(), "validation_family_count")
    namespace = {
        "prepared": Dataset.from_list(
            [
                {"repo_family": "family-a", "value": 1},
                {"repo_family": "family-b", "value": 2},
            ]
        ),
        "Counter": Counter,
        "hashlib": hashlib,
        "PUSH_DATASET": False,
        "DEMO_MODE": True,
    }
    exec(split_cell, namespace)

    dataset_dict = namespace["dataset_dict"]
    assert set(dataset_dict) == {"train", "validation"}
    assert len(dataset_dict["train"]) == 1
    assert len(dataset_dict["validation"]) == 1
    assert set(dataset_dict["train"]["repo_family"]).isdisjoint(
        dataset_dict["validation"]["repo_family"]
    )


def test_notebooks_02_and_03_demo_data_execute_after_arrow_round_trip():
    generator = load_generator()
    notebook_02 = generator.build_02_data()
    namespace_02 = {
        "json": json,
        "tokenizer": FakeTokenizer(),
        "Dataset": Dataset,
        "DEMO_MODE": True,
        "SOURCE_DATASET_IDS": [],
        "Counter": Counter,
        "hashlib": hashlib,
        "np": np,
    }
    exec(generator.TOOLS_CELL, namespace_02)
    exec(code_cell_containing(notebook_02, "demo_rows = ["), namespace_02)
    exec(code_cell_containing(notebook_02, "def validate_row"), namespace_02)
    exec(code_cell_containing(notebook_02, "def render_row(row: dict)"), namespace_02)
    exec(code_cell_containing(notebook_02, "validation_family_count"), namespace_02)
    assert len(namespace_02["dataset_dict"]["train"]) == 1
    assert len(namespace_02["dataset_dict"]["validation"]) == 1

    notebook_03 = generator.build_03_sft()
    namespace_03 = {
        "json": json,
        "tokenizer": FakeTokenizer(),
        "Dataset": Dataset,
        "DEMO_MODE": True,
    }
    exec(generator.TOOLS_CELL, namespace_03)
    exec(code_cell_containing(notebook_03, "def demo_rows()"), namespace_03)
    assert len(namespace_03["train_dataset"]) == 1
    assert len(namespace_03["eval_dataset"]) == 1


def test_baseline_search_uses_bounded_python_fallback(tmp_path):
    generator = load_generator()
    executor_cell = code_cell_containing(generator.build_01_baseline(), "class PilotTask")
    namespace = {
        "dataclass": dataclass,
        "Path": Path,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "json": json,
        "re": re,
        "DEMO_MODE": False,
        "PILOT_MANIFEST": tmp_path / "unused.jsonl",
        "TASK_ENV": {},
    }
    exec(executor_cell, namespace)

    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("def clamp(value):\n    return value\n")
    task = namespace["PilotTask"](
        task_id="search-test",
        repo_path=str(tmp_path),
        request="Find clamp",
        visible_test_command=[sys.executable, "-V"],
        hidden_test_command=[sys.executable, "-V"],
    )

    result = namespace["execute_tool"](task, "search", {"query": "clamp"})
    assert "src/example.py:1:def clamp(value):" in result
    assert namespace["execute_tool"](task, "search", {"query": "["}).startswith(
        "invalid regular expression:"
    )


def test_reward_fixture_rejects_deleted_visible_tests():
    generator = load_generator()
    environment_cell = code_cell_containing(generator.build_05_grpo(), "class ToyCodingEnv")
    namespace = {
        "hashlib": hashlib,
        "Path": Path,
        "re": re,
        "shutil": shutil,
        "tempfile": tempfile,
    }
    exec(environment_cell, namespace)

    environment = namespace["ToyCodingEnv"]()
    environment.reset()
    assert environment.apply_patch(
        "--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n"
        " def clamp(value, low, high):\n-    return value\n"
        "+    return max(low, min(high, value))\n"
    ) == "Done!"
    (environment.root / "tests" / "test_clamp.py").unlink()

    assert environment.run_tests("unit").startswith("Test integrity failure")
    assert environment.get_reward() == 0.0


def test_generated_notebooks_have_restart_and_schema_guards():
    generator = load_generator()
    notebooks = [
        generator.build_00_preflight(),
        generator.build_01_baseline(),
        generator.build_02_data(),
        generator.build_03_sft(),
        generator.build_04_dpo(),
        generator.build_05_grpo(),
        generator.build_06_qat_export(),
    ]

    all_source = "\n".join(cell.source for notebook in notebooks for cell in notebook.cells)
    assert "row.get(\"tools\") != TOOLS" not in all_source
    assert "then resume here" not in all_source
    assert "then continue from the runtime/authentication cell" not in all_source
    assert "This runtime was restarted. Rerun the notebook from the first cell" in all_source
    assert "TOOL_SCHEMA_JSON" in all_source
    assert 'assert "<function=read_file>" in rendered_probe' not in all_source
    assert "rendered_tool_schema(rendered_probe) == TOOL_SCHEMA_JSON" in all_source

    for notebook in notebooks:
        generator.validate_notebook(notebook, Path("generated.ipynb"))


def test_core_install_uses_a_resolvable_unsloth_compatibility_set():
    generator = load_generator()
    install = generator.INSTALL_CORE

    for package, expected in {
        "transformers": "5.3.0",
        "trl": "0.22.2",
        "datasets": "4.3.0",
        "peft": "0.19.0",
    }.items():
        assert f'"{package}": "{expected}"' in install
    assert "c49429ed1f8b89749de77c0ec930ef19685c9ae5" not in install
    assert "b39c2276567639b93ca5b53658751e0f9c09b92f" not in install
    assert "--quiet" not in install
    assert "/content/qwen38_pip_install.log" in install


def test_g4_preflight_accepts_decimal_96_gb_and_rejects_decimal_48_gb():
    generator = load_generator()
    auth = generator.AUTH_AND_RUNTIME

    decimal_96_gb_in_gib = 96_000_000_000 / 1024**3
    decimal_48_gb_in_gib = 48_000_000_000 / 1024**3
    assert decimal_96_gb_in_gib < 90
    assert decimal_96_gb_in_gib >= 85
    assert decimal_48_gb_in_gib < 85

    assert "MIN_G4_TOTAL_GIB = 85.0" in auth
    assert "gpu_total_gib < MIN_G4_TOTAL_GIB" in auth
    assert "gpu_gib < 90" not in auth
    assert '"gpu_total_gib": round(gpu_total_gib, 2)' in auth


def test_generated_cell_ids_are_deterministic():
    generator = load_generator()
    first = generator.build_02_data()
    second = generator.build_02_data()
    first_ids = [cell.id for cell in first.cells]
    second_ids = [cell.id for cell in second.cells]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_local_markdown_links_resolve():
    markdown_files = [ROOT / "README.md", ROOT / "notebooks" / "README.md"]
    markdown_files.extend(sorted((ROOT / "docs").glob("*.md")))
    failures = []
    for markdown_file in markdown_files:
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown_file.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if local_target and not (markdown_file.parent / local_target).resolve().exists():
                failures.append(f"{markdown_file.relative_to(ROOT)} -> {target}")
    assert not failures, failures
