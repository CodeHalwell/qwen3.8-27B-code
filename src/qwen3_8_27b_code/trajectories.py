"""Synthesise execution-verified, native-schema SFT trajectories.

Every trajectory is produced by driving :class:`RepoHarness` against a
materialised fixture repository: tool observations are recorded verbatim from
real executions (including real pytest runs), never written by hand, and a
trajectory is only emitted when its verification actually passed. A violated
expectation raises :class:`GenerationError` instead of emitting a bad row.

Corpus caveat: pytest output embeds wall-clock timings, so regenerating the
corpus reproduces identical structure, ids and patches but not identical
observation bytes. Validation therefore checks structure, not file hashes of
observations.
"""

from __future__ import annotations

from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from .fixtures import FixtureTask, VARIANTS_PER_FAMILY, iter_tasks
from .harness import RepoHarness
from .schema import TOOL_SCHEMA_JSON, TOOL_SCHEMA_VERSION, TOOLS

GENERATOR_VERSION = "scripted-gold-v1"

# 20-slot cycle: 11 standard, 3 search_first, 3 recovery, 2 test_author,
# 1 inspect_answer. Families that cannot support a shape fall back to standard.
SHAPE_CYCLE = [
    "standard", "standard", "search_first", "standard", "recovery",
    "standard", "standard", "test_author", "standard", "standard",
    "recovery", "standard", "search_first", "standard", "standard",
    "inspect_answer", "standard", "recovery", "search_first", "test_author",
]

# 10-slot cycle: 50% medium, 30% low, 20% xhigh per docs/data-strategy.md.
EFFORT_CYCLE = [
    "medium", "low", "medium", "xhigh", "medium",
    "low", "medium", "medium", "low", "xhigh",
]


class GenerationError(RuntimeError):
    """A trajectory violated an execution expectation and was not emitted."""


def unified_patch(path: str, before: str, after: str) -> str:
    """Build a git-apply-compatible unified diff for one repository file."""
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


class _EpisodeBuilder:
    """Accumulate one trajectory while executing its tools for real."""

    def __init__(self, task: FixtureTask, files: dict[str, str], developer: str, request: str):
        self.task = task
        self._tmp = Path(tempfile.mkdtemp(prefix="qwen38_traj_"))
        self.root = self._tmp / "repo"
        for path, content in files.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        self.harness = RepoHarness(self.root)
        self.messages = [
            {"role": "developer", "content": developer},
            {"role": "user", "content": request},
        ]

    def call(self, name: str, arguments: dict, reasoning: str) -> str:
        observation = self.harness.execute(name, arguments)
        self.messages.append({
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
        })
        self.messages.append({"role": "tool", "name": name, "content": observation})
        return observation

    def expect(self, condition: bool, detail: str, observation: str) -> None:
        if not condition:
            raise GenerationError(
                f"{self.task.family}/{self.task.variant}: {detail}\n--- observation ---\n{observation[:2000]}"
            )

    def finish(self, content: str) -> None:
        self.messages.append({"role": "assistant", "reasoning_content": "", "content": content})

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


def _run_and_expect(builder: _EpisodeBuilder, reasoning: str, passing: bool) -> str:
    observation = builder.call("run_tests", {"profile": "unit"}, reasoning)
    if passing:
        builder.expect(observation.startswith("exit=0"), "expected passing tests", observation)
    else:
        builder.expect(
            observation.startswith("exit=") and not observation.startswith("exit=0"),
            "expected failing tests",
            observation,
        )
    return observation


def _patch_and_expect(builder: _EpisodeBuilder, path: str, before: str, after: str, reasoning: str) -> None:
    observation = builder.call("apply_patch", {"patch": unified_patch(path, before, after)}, reasoning)
    builder.expect(observation == "patch applied", "patch was rejected", observation)


def synthesise(task: FixtureTask, shape: str, reasoning_effort: str) -> dict:
    """Produce one validated trajectory row for the requested shape."""
    if shape == "test_author" and not (task.supports_test_author and task.weak_tests and task.test_addition):
        shape = "standard"
    if shape == "recovery" and task.partial_module is None:
        shape = "standard"

    tests = task.weak_tests if shape == "test_author" else task.strong_tests
    module = task.fixed_module if shape == "inspect_answer" else task.buggy_module
    files = {task.module_path: module, task.tests_path: tests}
    developer = task.developer
    request = task.answer_question if shape == "inspect_answer" else task.request
    builder = _EpisodeBuilder(task, files, developer, request)
    try:
        if shape == "inspect_answer":
            listing = builder.call("list_files", {"path": "."}, "I should see the repository layout before answering.")
            builder.expect(task.module_path in listing, "module missing from listing", listing)
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            builder.finish(task.answer_text)
            # The answer is only trustworthy against a healthy repository.
            verification_observation = builder.harness.execute("run_tests", {"profile": "unit"})
            if not verification_observation.startswith("exit=0"):
                raise GenerationError(
                    f"{task.family}/{task.variant}: inspect_answer fixture is not green\n{verification_observation[:2000]}"
                )
        elif shape == "test_author":
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            builder.call(
                "read_file", {"path": task.tests_path},
                "I should check what the existing tests cover before changing anything.",
            )
            _patch_and_expect(
                builder, task.tests_path, task.weak_tests, task.weak_tests + task.test_addition,
                "The suite never exercises the broken case; a regression test will make the bug visible before I fix it.",
            )
            _run_and_expect(builder, "The new regression test should fail against the current implementation.", passing=False)
            _patch_and_expect(builder, task.module_path, task.buggy_module, task.fixed_module, task.bug_reasoning)
            _run_and_expect(builder, task.verify_reasoning, passing=True)
            builder.finish(task.summary + " The new regression test now guards the previously untested case.")
        elif shape == "recovery":
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            _patch_and_expect(builder, task.module_path, task.buggy_module, task.partial_module, task.partial_reasoning)
            _run_and_expect(builder, task.verify_reasoning, passing=False)
            _patch_and_expect(builder, task.module_path, task.partial_module, task.fixed_module, task.recovery_reasoning)
            _run_and_expect(builder, "The corrected change should pass the full unit profile now.", passing=True)
            builder.finish(task.summary)
        elif shape == "search_first":
            hits = builder.call("search", {"query": task.search_query}, f"I should locate the implementation with a search for {task.search_query!r}.")
            builder.expect(
                task.module_path in hits and not hits.startswith("invalid regular expression"),
                "search did not locate the module", hits,
            )
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            _patch_and_expect(builder, task.module_path, task.buggy_module, task.fixed_module, task.bug_reasoning)
            _run_and_expect(builder, task.verify_reasoning, passing=True)
            builder.finish(task.summary)
        elif shape == "standard":
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            _patch_and_expect(builder, task.module_path, task.buggy_module, task.fixed_module, task.bug_reasoning)
            _run_and_expect(builder, task.verify_reasoning, passing=True)
            builder.finish(task.summary)
        else:
            raise GenerationError(f"unknown shape {shape!r}")
        return {
            "id": f"sft/{task.family}-{task.variant:03d}",
            "source": GENERATOR_VERSION,
            "repo_family": task.family,
            "shape": shape,
            "reasoning_effort": reasoning_effort,
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "tool_schema_json": TOOL_SCHEMA_JSON,
            "tools": TOOLS,
            "messages": builder.messages,
            "verification": {"all_required_tests_pass": True, "runner": "python -m pytest -q"},
        }
    finally:
        builder.cleanup()


def generate_rows(variants_per_family: int = VARIANTS_PER_FAMILY) -> list[dict]:
    rows = []
    for index, task in enumerate(iter_tasks(variants_per_family)):
        shape = SHAPE_CYCLE[index % len(SHAPE_CYCLE)]
        effort = EFFORT_CYCLE[index % len(EFFORT_CYCLE)]
        rows.append(synthesise(task, shape, effort))
    return rows


def _approx_tokens(text: str) -> int:
    """Whitespace proxy used only until the real tokenizer measures the corpus."""
    return len(text.split())


def quality_report(rows: list[dict], corpus_path: Path) -> dict:
    tool_calls = [
        sum(1 for message in row["messages"] if message["role"] == "assistant" and message.get("tool_calls"))
        for row in rows
    ]
    approx_lengths = sorted(
        _approx_tokens(json.dumps(row["messages"], ensure_ascii=False)) for row in rows
    )

    def percentile(ordered: list[int], fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {
        "generator_version": GENERATOR_VERSION,
        "rows": len(rows),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "families": dict(sorted(Counter(row["repo_family"] for row in rows).items())),
        "shapes": dict(sorted(Counter(row["shape"] for row in rows).items())),
        "reasoning_effort": dict(sorted(Counter(row["reasoning_effort"] for row in rows).items())),
        "tool_calls": {
            "total": sum(tool_calls),
            "mean_per_row": round(statistics.mean(tool_calls), 3),
            "max_per_row": max(tool_calls),
        },
        "approx_whitespace_tokens": {
            "note": "whitespace proxy; run scripts/validate_dataset_rendering.py for real token counts",
            "p50": percentile(approx_lengths, 0.50),
            "p90": percentile(approx_lengths, 0.90),
            "p99": percentile(approx_lengths, 0.99),
            "max": approx_lengths[-1],
        },
        "limits": [
            "Python-only fixtures with pytest as the sole build system",
            "single-file gold fixes; multi-file work needs real repositories",
            "scripted gold trajectories: no model-generated exploration variance",
            "pytest observations embed wall-clock timings, so regeneration is structurally but not byte identical",
        ],
        "execution": "every observation recorded from a real harness run; run_tests invoked pytest in the fixture repository",
    }


def write_corpus(out_path: Path, report_path: Path, variants_per_family: int = VARIANTS_PER_FAMILY) -> dict:
    rows = generate_rows(variants_per_family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = quality_report(rows, out_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
