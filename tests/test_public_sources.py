"""Public Hub datasets converted into native-schema rows."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from qwen3_8_27b_code import harness, public_sources as ps
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


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
    assert spec and spec.loader
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator


def _validate_cell(generator) -> str:
    return next(
        cell.source for cell in generator.build_02_data().cells
        if cell.cell_type == "code" and "def validate_row" in cell.source
    )


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


def test_open_swe_rows_end_before_the_unanswered_submit_command():
    # Real trajectories end on a bash call to the submit command that no
    # tool message answers; the row stops at the last observation.
    row = _swe_row(turns=2)
    row["messages"][-1] = {
        "role": "assistant", "content": "Patch verified.", "reasoning_content": "done",
        "tool_calls": [{"function": {"name": "bash", "arguments": json.dumps(
            {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"}
        )}, "type": "function"}],
    }
    (converted,) = ps.convert_open_swe_row(row, budget_tokens=10_000)
    assert [m["role"] for m in converted["messages"]] == ["developer", "user", "assistant", "tool", "assistant", "tool"]
    assert "COMPLETE_TASK" not in json.dumps(converted["messages"])
    # Any other unanswered call is a truncated trace, not an ending.
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": "pytest"})
    assert ps.convert_open_swe_row(row, budget_tokens=10_000) == []


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


def test_long_trajectories_are_cut_to_a_faithful_head_window():
    count = lambda text: len(text)  # noqa: E731 - one token per character keeps the arithmetic visible
    row = _swe_row(turns=6, output_chars=100)
    messages = ps.convert_open_swe_messages(row["messages"])
    whole = ps.window_trajectory(messages, budget_tokens=100_000, count=count)
    assert [kind for kind, _ in whole] == ["whole"]
    windows = ps.window_trajectory(messages, budget_tokens=420, count=count)
    # One head window and never a tail: the last turns depend on edits the
    # omitted middle made, so a tail would show tests passing for a patch
    # the transcript never contains.
    assert [kind for kind, _ in windows] == ["head"]
    ((_, head),) = windows
    assert head[:2] == messages[:2] and head[2] is messages[2]
    assert head == messages[: len(head)]
    assert len(head) < len(messages)
    # Turns stay with their observations: the window does not end mid-pair.
    pending = 0
    for message in head[2:]:
        if message["role"] == "assistant":
            assert pending == 0
            pending = len(message.get("tool_calls") or [])
        else:
            pending -= 1
    assert pending == 0
    rows = ps.convert_open_swe_row(row, budget_tokens=420, count=count)
    assert [r["verification"]["window"] for r in rows] == ["head"]
    assert all(ps.converted_tokens(r, count) <= 420 for r in rows)
    assert ps.window_trajectory(messages, budget_tokens=10, count=count) == []


def test_converted_shell_observations_match_what_the_executor_returns(tmp_path):
    """The rows teach the shape the harness produces, exit code included."""
    executor = harness.RepoHarness(tmp_path)
    observed = executor.execute("shell", {"command": "printf boom; exit 2"})
    assert observed == ps._tool_content(json.dumps({"returncode": 2, "output": "boom"})) == "[exit code 2]\nboom"
    assert executor.execute("shell", {"command": "printf fine"}) == "fine"
    # Fifty megabytes of output: only the head and tail are ever held.
    long = executor.execute("shell", {"command": "head -c 50000000 /dev/zero | tr '\\0' a"})
    assert len(long) <= harness.SHELL_OUTPUT_LIMIT and harness.TRUNCATION_MARKER in long
    assert long.startswith("a" * 100) and long.endswith("a" * 100)


def test_shell_timeout_kills_the_whole_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "SHELL_TIMEOUT_SECONDS", 1)
    executor = harness.RepoHarness(tmp_path)
    started = time.monotonic()
    # The background sleep holds the output pipe; if it survived the
    # timeout, reading the pipe would wait for it, not for the 1 s limit.
    observed = executor.execute("shell", {"command": "sleep 30 & sleep 30"})
    assert observed == "[timed out after 1s]"
    assert time.monotonic() - started < 10


def test_sources_are_pinned_and_the_report_records_the_commit(monkeypatch):
    assert all(len(spec["revision"]) == 40 for spec in ps.SOURCE_LOADERS.values())
    monkeypatch.setattr(ps, "stream_source", lambda source, token=None: iter([_swe_row(), _swe_row(resolved=0)]))
    rows, report = ps.collect_public_rows({ps.SOURCE_OPEN_SWE: 5, ps.SOURCE_OPEN_CODE_REASONING: 0}, budget_tokens=10_000)
    assert len(rows) == 1 and report["converter"] == ps.CONVERTER_VERSION
    entry = report["sources"][ps.SOURCE_OPEN_SWE]
    assert entry["revision"] == ps.SOURCE_LOADERS[ps.SOURCE_OPEN_SWE]["revision"]
    assert entry["rows"] == 1 and entry["unverified"] == 0 and entry["windows"] == {"whole": 1}
    assert entry["scanned"] == 2
    assert ps.SOURCE_OPEN_CODE_REASONING not in report["sources"]


def test_open_code_instruct_keeps_only_fully_verified_answers():
    row = {"id": "1", "input": "Write f.", "output": "def f(): pass", "domain": "algorithmic", "average_test_score": "1.0"}
    (converted,) = ps.convert_open_code_instruct_row(row)
    assert converted["lane"] == "non_agentic"
    # One domain covers every row of this source, so the family is bucketed
    # by row id: the split can hold out part of it instead of all or none.
    assert converted["repo_family"].startswith("opencodeinstruct:algorithmic/")
    assert converted["reasoning_effort"] == "low"
    assert converted["messages"] == [{"role": "user", "content": "Write f."}, {"role": "assistant", "content": "def f(): pass"}]
    assert ps.convert_open_code_instruct_row(dict(row, average_test_score="0.9")) == []
    assert ps.convert_open_code_instruct_row(dict(row, average_test_score=None)) == []


def test_open_code_reasoning_moves_the_think_block_into_the_reasoning_field():
    row = {"id": "7", "source": "codeforces", "input": "Balance brackets.", "output": "<think>\nuse a stack\n</think>\n```python\nprint(1)\n```"}
    (converted,) = ps.convert_open_code_reasoning_row(row)
    assert converted["messages"][1] == {"role": "assistant", "content": "```python\nprint(1)\n```", "reasoning_content": "use a stack"}
    assert converted["reasoning_effort"] == "xhigh"
    assert converted["repo_family"].startswith("opencodereasoning:codeforces/")
    # Nothing ran these answers, and the row says so rather than claiming a pass.
    assert converted["verification"] == {"all_required_tests_pass": None, "runner": "none"}
    # split_1 rows carry "-" and need an external lookup: skipped.
    assert ps.convert_open_code_reasoning_row(dict(row, input="-")) == []
    assert ps.split_think("no tags") == ("", "no tags")


def test_non_agentic_families_are_bucketed_but_stay_stable():
    rows = [
        ps.convert_open_code_instruct_row(
            {"id": str(i), "input": "q", "output": "a", "domain": "generic", "average_test_score": "1.0"}
        )[0]
        for i in range(400)
    ]
    families = {row["repo_family"] for row in rows}
    # Enough families for the split to take a share, few enough to stay whole.
    assert len(families) == ps.FAMILY_BUCKETS
    assert all(family.startswith("opencodeinstruct:generic/") for family in families)
    # The same row lands in the same family on every rebuild, so a corpus
    # rebuilt later splits the same way. These are the values this hash
    # produces; changing them changes which rows are held out.
    assert [ps.bucketed_family("opencodeinstruct:generic", row_id) for row_id in ("1", "7", "abc", "12345")] == [
        "opencodeinstruct:generic/11",
        "opencodeinstruct:generic/17",
        "opencodeinstruct:generic/13",
        "opencodeinstruct:generic/05",
    ]
    # The prefix is carried through and an integer id reads like its string.
    assert ps.bucketed_family("other:prefix", "7").startswith("other:prefix/")
    assert ps.bucketed_family("p", 7) == ps.bucketed_family("p", "7")


def test_convert_rows_stops_at_the_limit_and_drops_rows_over_budget():
    rows = [_swe_row(turns=1), _swe_row(turns=1), _swe_row(turns=40, output_chars=2_000)]
    out = list(ps.convert_rows(ps.SOURCE_OPEN_SWE, rows, limit=2, budget_tokens=10_000))
    assert len(out) == 2
    # A source that yields nothing is given up on after a bounded number
    # of rows, not streamed to its end.
    consumed = 0

    def endless():
        nonlocal consumed
        while True:
            consumed += 1
            yield _swe_row(resolved=0)

    assert list(ps.convert_rows(ps.SOURCE_OPEN_SWE, endless(), limit=2, budget_tokens=10_000)) == []
    assert consumed == 2 * ps.SCAN_ROWS_PER_NATIVE_ROW
    with pytest.raises(ValueError, match="No converter"):
        list(ps.convert_rows("nobody/nothing", rows, limit=1, budget_tokens=10))


def test_public_rows_keep_their_lane_behind_a_bootstrap_row():
    """The corpus notebook 02 actually builds: bootstrap rows first."""
    from datasets import Dataset

    generator = _load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(_validate_cell(generator), namespace)
    validate_row = namespace["validate_row"]

    # A row in the committed bootstrap corpus shape: no lane column at all.
    bootstrap = json.loads((ROOT / "data" / "native_sft" / "trajectories.jsonl").read_text().splitlines()[0])
    assert "lane" not in bootstrap
    public = ps.convert_open_code_instruct_row(
        {"id": "1", "input": "q", "output": "a", "domain": "generic", "average_test_score": "1.0"}
    ) + ps.convert_open_code_reasoning_row({"id": "2", "source": "s", "input": "q", "output": "<think>t</think>a"})

    # Dataset.from_list names its columns from the first row, so without the
    # helper the lane of every public row is dropped and the non-agentic
    # rows read as agentic trajectories with no tool call.
    naive = Dataset.from_list([bootstrap] + public)
    assert "lane" not in naive.column_names
    assert [validate_row(row) for row in naive][1:] != [[], []]

    unified = Dataset.from_list(namespace["unify_columns"]([bootstrap] + public))
    assert [row["lane"] for row in unified] == [None, "non_agentic", "non_agentic"]
    assert [validate_row(row) for row in unified] == [[], [], []]


def test_converted_rows_pass_notebook_02_validation():
    """The notebook's validator, not a re-statement of it."""
    generator = _load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(_validate_cell(generator), namespace)
    validate_row = namespace["validate_row"]

    for row in ps.convert_open_swe_row(_swe_row(turns=3), budget_tokens=10_000):
        assert validate_row(row) == [], row["id"]
    for row in ps.convert_open_code_instruct_row(
        {"id": "1", "input": "q", "output": "a", "domain": "generic", "average_test_score": "1.0"}
    ):
        assert validate_row(row) == []
    for row in ps.convert_open_code_reasoning_row({"id": "2", "source": "s", "input": "q", "output": "<think>t</think>a"}):
        assert validate_row(row) == []
