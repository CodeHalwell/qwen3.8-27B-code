"""Public Hub datasets converted into native-schema rows."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from qwen3_8_27b_code import public_sources as ps
from qwen3_8_27b_code.schema import TOOL_SCHEMA_JSON, TOOL_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def _swe_row(resolved: int = 1, turns: int = 2, tool: str = "bash", output_chars: int = 20) -> dict:
    messages = [{"role": "system", "content": "shell agent"}, {"role": "user", "content": "<pr_description>fix</pr_description>"}]
    for index in range(turns):
        messages.append({
            "role": "assistant",
            "content": f"step {index}",
            "reasoning_content": f"thinking {index}",
            "tool_calls": [
                {"function": {"name": tool, "arguments": json.dumps({"command": f"cat file{index}.py"})}, "type": "function"},
            ],
        })
        messages.append({"role": "tool", "content": json.dumps({"returncode": 0, "output": "x" * output_chars})})
    messages.append({"role": "assistant", "content": "Submitted."})
    return {"resolved": resolved, "repo": "owner/repo", "trajectory_id": f"traj-{turns}", "messages": messages}


def test_open_swe_bash_becomes_the_native_shell_tool():
    rows = ps.convert_open_swe_row(_swe_row(), budget_tokens=10_000)
    assert len(rows) == 1
    row = rows[0]
    assert row["lane"] == "agentic" and row["repo_family"] == "owner/repo" and row["reasoning_effort"] == "xhigh"
    assert row["tool_schema_version"] == TOOL_SCHEMA_VERSION and row["tool_schema_json"] == TOOL_SCHEMA_JSON
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["developer", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    call = row["messages"][2]["tool_calls"][0]
    assert call == {"type": "function", "function": {"name": "shell", "arguments": {"command": "cat file0.py"}}}
    assert row["messages"][2]["reasoning_content"] == "thinking 0"
    assert row["messages"][3] == {"role": "tool", "name": "shell", "content": "x" * 20}
    assert row["verification"]["window"] == "whole"


def test_open_swe_rows_that_did_not_resolve_or_use_other_tools_are_dropped():
    assert ps.convert_open_swe_row(_swe_row(resolved=0), budget_tokens=10_000) == []
    assert ps.convert_open_swe_row(_swe_row(tool="str_replace_editor"), budget_tokens=10_000) == []
    orphan = _swe_row()
    orphan["messages"].insert(2, {"role": "tool", "content": "no call"})
    assert ps.convert_open_swe_row(orphan, budget_tokens=10_000) == []


def test_tool_observations_keep_exit_codes_and_are_trimmed():
    assert ps._tool_content(json.dumps({"returncode": 2, "output": "boom"})) == "[exit code 2]\nboom"
    long = ps._tool_content(json.dumps({"returncode": 0, "output": "a" * 10_000}))
    assert len(long) <= ps.MAX_TOOL_OUTPUT_CHARS and ps.TRUNCATION_MARKER in long
    assert ps._tool_content("plain text") == "plain text"


def test_long_trajectories_are_windowed_into_head_and_tail():
    count = lambda text: len(text)  # noqa: E731 - one token per character keeps the arithmetic visible
    row = _swe_row(turns=6, output_chars=100)
    messages = ps.convert_open_swe_messages(row["messages"])
    whole = ps.window_trajectory(messages, budget_tokens=100_000, count=count)
    assert [kind for kind, _ in whole] == ["whole"]
    windows = ps.window_trajectory(messages, budget_tokens=420, count=count)
    kinds = [kind for kind, _ in windows]
    assert kinds == ["head", "tail"]
    head, tail = (window for _, window in windows)
    # Both keep the prefix; the head starts at the first turn, the tail ends at the last.
    assert head[:2] == messages[:2] and tail[:2] == messages[:2]
    assert head[2] is messages[2] and tail[-1] is messages[-1]
    # Turns stay with their observations: no window starts or ends mid-pair.
    for window in (head, tail):
        pending = 0
        for message in window[2:]:
            if message["role"] == "assistant":
                assert pending == 0
                pending = len(message.get("tool_calls") or [])
            else:
                pending -= 1
        assert pending == 0
    # Two rows come out of one long trajectory, each within budget.
    rows = ps.convert_open_swe_row(row, budget_tokens=420, count=count)
    assert [r["verification"]["window"] for r in rows] == ["head", "tail"]
    assert all(ps.converted_tokens(r, count) <= 420 for r in rows)
    assert ps.window_trajectory(messages, budget_tokens=10, count=count) == []


def test_open_code_instruct_keeps_only_fully_verified_answers():
    row = {"id": "1", "input": "Write f.", "output": "def f(): pass", "domain": "algorithmic", "average_test_score": "1.0"}
    (converted,) = ps.convert_open_code_instruct_row(row)
    assert converted["lane"] == "non_agentic" and converted["repo_family"] == "opencodeinstruct:algorithmic"
    assert converted["reasoning_effort"] == "low"
    assert converted["messages"] == [{"role": "user", "content": "Write f."}, {"role": "assistant", "content": "def f(): pass"}]
    assert ps.convert_open_code_instruct_row(dict(row, average_test_score="0.9")) == []
    assert ps.convert_open_code_instruct_row(dict(row, average_test_score=None)) == []


def test_open_code_reasoning_moves_the_think_block_into_the_reasoning_field():
    row = {"id": "7", "source": "codeforces", "input": "Balance brackets.", "output": "<think>\nuse a stack\n</think>\n```python\nprint(1)\n```"}
    (converted,) = ps.convert_open_code_reasoning_row(row)
    assert converted["messages"][1] == {"role": "assistant", "content": "```python\nprint(1)\n```", "reasoning_content": "use a stack"}
    assert converted["reasoning_effort"] == "xhigh" and converted["repo_family"] == "opencodereasoning:codeforces"
    # split_1 rows carry "-" and need an external lookup: skipped.
    assert ps.convert_open_code_reasoning_row(dict(row, input="-")) == []
    assert ps.split_think("no tags") == ("", "no tags")


def test_convert_rows_stops_at_the_limit_and_drops_rows_over_budget():
    rows = [_swe_row(turns=1), _swe_row(turns=1), _swe_row(turns=40, output_chars=2_000)]
    out = list(ps.convert_rows(ps.SOURCE_OPEN_SWE, rows, limit=2, budget_tokens=10_000))
    assert len(out) == 2
    with pytest.raises(ValueError, match="No converter"):
        list(ps.convert_rows("nobody/nothing", rows, limit=1, budget_tokens=10))


def test_converted_rows_pass_notebook_02_validation():
    """The notebook's validator, not a re-statement of it."""
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    validate_cell = next(
        cell.source for cell in generator.build_02_data().cells
        if cell.cell_type == "code" and "def validate_row" in cell.source
    )
    exec(validate_cell, namespace)
    validate_row = namespace["validate_row"]

    for row in ps.convert_open_swe_row(_swe_row(turns=3), budget_tokens=10_000):
        assert validate_row(row) == [], row["id"]
    for row in ps.convert_open_code_instruct_row(
        {"id": "1", "input": "q", "output": "a", "domain": "generic", "average_test_score": "1.0"}
    ):
        assert validate_row(row) == []
    for row in ps.convert_open_code_reasoning_row({"id": "2", "source": "s", "input": "q", "output": "<think>t</think>a"}):
        assert validate_row(row) == []
