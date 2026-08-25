"""Contracts for the bootstrap corpus and its package/notebook parity.

The notebooks stay self-contained for Colab, so the tool schema, parser and
executor exist both as generated cells and as package modules. These tests
pin the two copies together and verify the committed corpus artifacts still
satisfy the exact validation the notebooks will run.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from datasets import Dataset, load_dataset

from qwen3_8_27b_code import fixtures, harness, parsing, preferences, schema, trajectories

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"
SFT_CORPUS = ROOT / "data" / "native_sft" / "trajectories.jsonl"
SFT_REPORT = ROOT / "data" / "native_sft" / "quality_report.json"
PREF_CORPUS = ROOT / "data" / "preferences" / "pairs.jsonl"
PREF_REPORT = ROOT / "data" / "preferences" / "quality_report.json"


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_package_schema_matches_notebook_tools_cell():
    generator = load_generator()
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)

    assert namespace["TOOL_SCHEMA_JSON"] == schema.TOOL_SCHEMA_JSON
    assert namespace["canonical_tool_schema"](schema.TOOLS) == schema.TOOL_SCHEMA_JSON
    probe = [
        {"role": "developer", "content": "Policy."},
        {"role": "system", "content": "More policy."},
        {"role": "user", "content": "Task."},
        {"role": "assistant", "content": "Done."},
    ]
    assert namespace["canonical_to_qwen"](probe) == schema.canonical_to_qwen(probe)


def test_package_parser_matches_notebook_parser_cell():
    generator = load_generator()
    parser_cell = code_cell_containing(generator.build_01_baseline(), "def parse_tool_calls")
    namespace = {"re": re}
    exec(parser_cell, namespace)

    trailing_space_probe = (
        "<tool_call>\n<function=apply_patch>\n<parameter=patch>\n"
        "--- a/src/x.py\n+++ b/src/x.py\n@@ -1,1 +1,1 @@\n-old\n+new \n"
        "</parameter>\n</function>\n</tool_call>"
    )
    probes = [
        "<think>plan</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/cache.py\n</parameter>\n</function>\n</tool_call>",
        (
            "<tool_call>\n<function=apply_patch>\n<parameter=patch>\n"
            "--- a/src/x.py\n+++ b/src/x.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
            "</parameter>\n</function>\n</tool_call>"
        ),
        trailing_space_probe,
        "plain final answer with no calls",
    ]
    for probe in probes:
        assert namespace["parse_tool_calls"](probe) == parsing.parse_tool_calls(probe)

    # A diff line that intentionally ends in whitespace must survive parsing
    # (parity alone would also pass if both copies were wrong).
    patch = parsing.parse_tool_calls(trailing_space_probe)[1][0]["function"]["arguments"]["patch"]
    assert patch.endswith("+new "), repr(patch[-8:])


def test_package_harness_matches_notebook_executor(tmp_path):
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
        "TASK_ENV": harness.filtered_environment(),
    }
    exec(executor_cell, namespace)

    # Same stateful call sequence against two identical repository copies:
    # one driven by the notebook executor, one by the package harness.
    task = fixtures.FAMILY_BUILDERS["bounds"](3)
    repos = {}
    for owner in ("notebook", "package"):
        repo = tmp_path / owner
        for path, content in {task.module_path: task.buggy_module, task.tests_path: task.strong_tests}.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        repos[owner] = repo
    pilot = namespace["PilotTask"](
        task_id="parity",
        repo_path=str(repos["notebook"]),
        request=task.request,
        visible_test_command=harness.default_test_command(),
        hidden_test_command=[sys.executable, "-V"],
    )
    package = harness.RepoHarness(repos["package"])

    calls = [
        ("list_files", {"path": "."}),
        ("read_file", {"path": task.module_path}),
        ("search", {"query": task.search_query}),
        ("search", {"query": "["}),
        ("apply_patch", {"patch": trajectories.unified_patch(task.module_path, task.buggy_module, task.fixed_module)}),
        ("shell", {"command": "rm -rf /"}),
    ]
    for name, arguments in calls:
        assert namespace["execute_tool"](pilot, name, arguments) == package.execute(name, arguments), name
    notebook_run = namespace["execute_tool"](pilot, "run_tests", {"profile": "unit"})
    package_run = package.execute("run_tests", {"profile": "unit"})
    assert notebook_run.splitlines()[0] == package_run.splitlines()[0] == "exit=0"


def test_committed_corpus_passes_notebook_02_validation_after_arrow_round_trip():
    generator = load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(code_cell_containing(generator.build_02_data(), "def validate_row"), namespace)

    dataset = load_dataset("json", data_files=str(SFT_CORPUS), split="train")
    errors = [(row["id"], problems) for row in dataset if (problems := namespace["validate_row"](row))]
    assert not errors, errors[:5]
    assert len(set(dataset["repo_family"])) >= 2


def test_committed_corpus_matches_quality_report():
    rows = load_jsonl(SFT_CORPUS)
    report = json.loads(SFT_REPORT.read_text())
    assert report["rows"] == len(rows)
    assert sum(report["families"].values()) == len(rows)
    assert sum(report["shapes"].values()) == len(rows)
    assert {row["reasoning_effort"] for row in rows} == {"low", "medium", "xhigh"}
    assert all(row["verification"]["all_required_tests_pass"] for row in rows)
    assert all(row["tool_schema_json"] == schema.TOOL_SCHEMA_JSON for row in rows)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))
    # The rendered section is appended by scripts/validate_dataset_rendering.py
    # against the real tokenizer; its presence proves the corpus was checked.
    assert report["rendered"]["rows"] == len(rows)
    assert report["rendered"]["tokens"]["max"] <= 4096


def test_committed_preference_pairs_satisfy_notebook_04_contract():
    pairs = load_jsonl(PREF_CORPUS)
    report = json.loads(PREF_REPORT.read_text())
    assert report["rows"] == len(pairs)
    assert report["rendered"]["rows"] == len(pairs)
    for pair in pairs:
        assert pair["infra_status"] == "ok"
        assert pair["chosen_reward"] > pair["rejected_reward"]
        assert {message["role"] for message in pair["prompt_messages"]} <= {"developer", "system", "user", "assistant", "tool"}
        assert pair["chosen_message"]["role"] == "assistant"
        assert pair["rejected_message"]["role"] == "assistant"
        assert pair["chosen_message"] != pair["rejected_message"]
        assert pair["evidence"]["basis"]
        # The behaviour DPO rewards must be visible to the trainer: the chosen
        # continuation acts (a tool call), never claims untraced completed work.
        assert pair["chosen_message"].get("tool_calls"), pair["id"]

    by_contrast = {}
    for pair in pairs:
        by_contrast.setdefault(pair["contrast_type"], []).append(pair)

    for pair in by_contrast["patch_outcome"]:
        assert pair["evidence"]["chosen_run"] == "exit=0"
        assert pair["evidence"]["rejected_run"] != "exit=0"
        assert pair["rejected_message"]["tool_calls"][0]["function"]["name"] == "apply_patch"
        assert any(message["role"] == "tool" for message in pair["prompt_messages"])
    for pair in by_contrast["verification_claim"]:
        assert pair["evidence"]["chosen_run"] == "exit=0"
        assert pair["chosen_message"]["tool_calls"][0]["function"]["name"] == "run_tests"
        # State: patch applied in-transcript, tests not yet run.
        assert any(message.get("content") == "patch applied" for message in pair["prompt_messages"])
    for pair in by_contrast["test_integrity"]:
        assert pair["evidence"]["failing_run"] != "exit=0"
        assert pair["evidence"]["chosen_run"] == "exit=0"
        # The failing run the rejected message wants to hide is in the transcript.
        assert any(
            message["role"] == "tool" and message["content"].startswith("exit=") and not message["content"].startswith("exit=0")
            for message in pair["prompt_messages"]
        )
    for pair in by_contrast["inspect_first"]:
        assert pair["chosen_message"]["tool_calls"][0]["function"]["name"] == "read_file"


def test_synthesis_still_produces_valid_rows_end_to_end():
    """Tiny live check: real harness, real pytest, notebook validation."""
    generator = load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(code_cell_containing(generator.build_02_data(), "def validate_row"), namespace)

    for family, shape in [("intervals", "recovery"), ("config", "test_author")]:
        row = trajectories.synthesise(fixtures.FAMILY_BUILDERS[family](1), shape, "medium")
        assert not namespace["validate_row"](row)
        assert row["shape"] == shape


def test_preference_generation_grounds_patch_contrasts():
    task = fixtures.FAMILY_BUILDERS["stats"](preferences.PREFERENCE_VARIANT_BASE)
    pair = preferences._build_pair(task, "patch_outcome", "pref/stats-test")
    assert pair["evidence"]["chosen_run"] == "exit=0"
    assert pair["evidence"]["rejected_run"].startswith("exit=") and pair["evidence"]["rejected_run"] != "exit=0"
    chosen_patch = pair["chosen_message"]["tool_calls"][0]["function"]["arguments"]["patch"]
    assert chosen_patch == trajectories.unified_patch(task.module_path, task.buggy_module, task.fixed_module)
    # The prompt prefix carries the real read observation the decision is based on.
    assert pair["prompt_messages"][-1]["role"] == "tool"
    assert pair["prompt_messages"][-1]["content"].startswith("def ") or "class " in pair["prompt_messages"][-1]["content"]


def _assert_no_none_leaves(value, path="message"):
    if isinstance(value, dict):
        for key, item in value.items():
            assert item is not None, f"null leaked into rendered {path}.{key}"
            _assert_no_none_leaves(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_none_leaves(item, f"{path}[{index}]")


class _NullRejectingTokenizer:
    """Fails the render if Arrow-injected nulls reach the chat template."""

    def apply_chat_template(self, messages, **kwargs):
        _assert_no_none_leaves(messages)
        return "<render>"


def test_notebook_04_render_cell_strips_arrow_nulls_from_completions():
    """A tool-call chosen next to a content-only rejected unions their struct
    keys through Arrow; the render cell must strip the injected nulls before
    the template sees them."""
    generator = load_generator()
    notebook_04 = generator.build_04_dpo()
    namespace = {"json": json, "Dataset": Dataset, "DEMO_MODE": True}
    exec(generator.TOOLS_CELL, namespace)
    namespace["tokenizer"] = _NullRejectingTokenizer()
    exec(code_cell_containing(notebook_04, "demo_preferences = Dataset.from_list"), namespace)
    preferences_dataset = namespace["preferences"]
    assert len(preferences_dataset) == 2
    assert set(preferences_dataset.column_names) == {"prompt", "chosen", "rejected"}


def test_corpus_sha_matches_report():
    for corpus, report_path in [(SFT_CORPUS, SFT_REPORT), (PREF_CORPUS, PREF_REPORT)]:
        report = json.loads(report_path.read_text())
        assert report["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
