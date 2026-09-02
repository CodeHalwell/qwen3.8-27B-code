"""Agent tasks: a materialisable repository plus verification the model cannot read.

Two things separate a task here from the scripted SFT fixtures in
``fixtures.py``. A task carries *hidden* checks, executed outside the model's
workspace after the episode ends, so an attempt cannot pass by satisfying only
the tests it can see. And it names the files the agent must not edit, so an
attempt that deletes the failing test is detected rather than scored green.

The evaluation families below are deliberately disjoint from the training
families: different bug classes, different modules, never generated into the
SFT corpus. ``tests/test_agent_tasks.py`` enforces that separation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
from typing import Iterator

from .fixtures import FixtureTask
from .harness import RepoHarness, default_test_command, filtered_environment

MODULE_DIR = "src"
TESTS_DIR = "tests"
HIDDEN_CHECK_TIMEOUT = 120

DEVELOPER_POOL = [
    "Inspect before editing and run tests before completing.",
    "Work only in the provided repository. Verify changes with the unit test profile.",
    "Make the smallest correct change and verify it with the unit tests.",
]


@dataclass(frozen=True)
class HiddenCheck:
    """A verifier script executed outside the repository the model can see."""

    name: str
    source: str


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    family: str
    variant: int
    request: str
    developer: str
    files: dict[str, str]
    hidden_checks: tuple[HiddenCheck, ...] = ()
    protected_paths: tuple[str, ...] = ()
    module_path: str = ""
    tests_path: str = ""
    reference_module: str | None = None
    baseline_should_fail: bool = True

    @property
    def has_hidden_verification(self) -> bool:
        return bool(self.hidden_checks)


@dataclass
class Verdict:
    """What actually happened to a repository after an attempt."""

    visible_exit: int | None = None
    hidden: dict[str, bool] = field(default_factory=dict)
    hidden_output: dict[str, str] = field(default_factory=dict)
    tampered_paths: list[str] = field(default_factory=list)

    @property
    def visible_passed(self) -> bool:
        return self.visible_exit == 0

    @property
    def hidden_passed(self) -> bool:
        return bool(self.hidden) and all(self.hidden.values())

    @property
    def regression(self) -> bool:
        """A check named ``no_regression`` failing means working code broke."""
        return any(
            name.startswith("no_regression") and not passed
            for name, passed in self.hidden.items()
        )

    @property
    def succeeded(self) -> bool:
        """Success requires untampered tests and every available verifier.

        Without hidden checks the visible suite is all there is, which is why
        the collector records whether a task had hidden verification at all.
        """
        if self.tampered_paths:
            return False
        if self.hidden:
            return self.hidden_passed and self.visible_exit in (None, 0)
        return self.visible_passed

    def as_dict(self) -> dict:
        return {
            "visible_exit": self.visible_exit,
            "hidden": dict(sorted(self.hidden.items())),
            "tampered_paths": sorted(self.tampered_paths),
            "succeeded": self.succeeded,
            "regression": self.regression,
        }


class TaskWorkspace:
    """A disposable checkout of one task, with its verifiers kept outside it."""

    def __init__(self, task: AgentTask, base: Path):
        self.task = task
        self.base = base
        self.root = base / "repo"
        self.hidden_dir = base / "hidden"
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        for relative, content in task.files.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        for check in task.hidden_checks:
            (self.hidden_dir / f"{check.name}.py").write_text(check.source)
        # The scratch home keeps credential caches out of reach of task code.
        self.environment = filtered_environment(home=base / "task_home")
        self.harness = RepoHarness(self.root, environment=self.environment)

    def run_visible_tests(self) -> int:
        result = subprocess.run(
            default_test_command(),
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=HIDDEN_CHECK_TIMEOUT,
        )
        return result.returncode

    def tampered_paths(self) -> list[str]:
        tampered = []
        for relative in self.task.protected_paths:
            target = self.root / relative
            if not target.is_file():
                tampered.append(relative)
            elif target.read_text() != self.task.files[relative]:
                tampered.append(relative)
        return tampered

    def verify(self) -> Verdict:
        verdict = Verdict(tampered_paths=self.tampered_paths())
        verdict.visible_exit = self.run_visible_tests()
        for check in self.task.hidden_checks:
            result = subprocess.run(
                [sys.executable, str(self.hidden_dir / f"{check.name}.py")],
                cwd=self.root,
                env=self.environment,
                text=True,
                capture_output=True,
                timeout=HIDDEN_CHECK_TIMEOUT,
            )
            verdict.hidden[check.name] = result.returncode == 0
            verdict.hidden_output[check.name] = (result.stdout + result.stderr)[-2000:]
        return verdict


@contextmanager
def materialise(task: AgentTask) -> Iterator[TaskWorkspace]:
    base = Path(tempfile.mkdtemp(prefix="qwen38_task_"))
    try:
        yield TaskWorkspace(task, base)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def hidden_script(module_path: str, symbol: str, assertions: str) -> str:
    """Build a verifier that loads the module by source, not by import.

    Importing would let a stale ``.pyc`` or an ``__init__`` side effect decide
    the result; reading the file is what the grader is actually asserting on.
    """
    statements = textwrap.dedent(assertions).strip().splitlines()
    body = "\n".join(f"    {line}" if line.strip() else "" for line in statements)
    return (
        "import pathlib\n"
        "namespace = {}\n"
        f"exec((pathlib.Path.cwd() / {module_path!r}).read_text(), namespace)\n"
        f"{symbol} = namespace[{symbol!r}]\n"
        "def check():\n"
        f"{body}\n"
        "check()\n"
        "print('ok')\n"
    )


def _rng(family: str, variant: int) -> random.Random:
    seed = int(hashlib.sha256(f"eval:{family}:{variant}".encode()).hexdigest(), 16) % 2**32
    return random.Random(seed)


def _tests(module_name: str, symbol: str, functions: list[str]) -> str:
    return f"from {MODULE_DIR}.{module_name} import {symbol}\n\n" + "\n".join(functions) + "\n"


def _build(
    family: str,
    variant: int,
    module_name: str,
    symbol: str,
    buggy: str,
    fixed: str,
    visible: list[str],
    request: str,
    contract: str,
    regression: str,
    rng: random.Random,
) -> AgentTask:
    module_path = f"{MODULE_DIR}/{module_name}.py"
    tests_path = f"{TESTS_DIR}/test_{module_name}.py"
    return AgentTask(
        task_id=f"eval/{family}-{variant:03d}",
        family=family,
        variant=variant,
        request=request,
        developer=rng.choice(DEVELOPER_POOL),
        files={module_path: buggy, tests_path: _tests(module_name, symbol, visible)},
        hidden_checks=(
            HiddenCheck("contract", hidden_script(module_path, symbol, contract)),
            HiddenCheck("no_regression", hidden_script(module_path, symbol, regression)),
        ),
        protected_paths=(tests_path,),
        module_path=module_path,
        tests_path=tests_path,
        reference_module=fixed,
    )


def _family_word_wrap(variant: int) -> AgentTask:
    rng = _rng("word_wrap", variant)
    symbol = rng.choice(["wrap", "wrap_text", "fill"])
    module_name = rng.choice(["wrapping", "textflow", "layout"])
    header = (
        f'def {symbol}(text, width):\n'
        f'    """Wrap text into lines of at most width characters, breaking at spaces."""\n'
    )
    buggy = header + "    return [text[index:index + width] for index in range(0, len(text), width)]\n"
    fixed = header + (
        "    lines = []\n"
        '    current = ""\n'
        "    for word in text.split():\n"
        '        candidate = f"{current} {word}" if current else word\n'
        "        if len(candidate) > width and current:\n"
        "            lines.append(current)\n"
        "            current = word\n"
        "        else:\n"
        "            current = candidate\n"
        "    if current:\n"
        "        lines.append(current)\n"
        "    return lines\n"
    )
    visible = [
        f'def test_short_text_is_one_line():\n    assert {symbol}("the quick", 10) == ["the quick"]\n',
        (
            "def test_wraps_at_a_space_not_mid_word():\n"
            f'    assert {symbol}("the quick brown fox", 10) == ["the quick", "brown fox"]\n'
        ),
    ]
    contract = f'''
    sentence = "alpha beta gamma delta epsilon"
    lines = {symbol}(sentence, 11)
    assert all(len(line) <= 11 for line in lines), lines
    assert " ".join(lines).split() == sentence.split(), lines
    assert {symbol}("", 8) == []
    assert {symbol}("supercalifragilistic", 5) == ["supercalifragilistic"]
    '''
    regression = f'''
    assert {symbol}("alpha beta", 20) == ["alpha beta"]
    assert {symbol}("", 5) == []
    '''
    request = rng.choice([
        f"{symbol}() chops text every width characters instead of wrapping at spaces, so words are split. Fix it and run the unit tests.",
        f"Wrapping with {symbol}() breaks words in half rather than moving them to the next line. Fix it minimally and verify with the tests.",
    ])
    return _build("word_wrap", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng)


def _family_duration_format(variant: int) -> AgentTask:
    rng = _rng("duration_format", variant)
    symbol = rng.choice(["format_duration", "as_mmss", "render_duration"])
    module_name = rng.choice(["duration", "timing", "clockfmt"])
    header = (
        f'def {symbol}(seconds):\n'
        f'    """Render whole seconds as M:SS, with the seconds zero-padded to two digits."""\n'
    )
    buggy = header + '    return f"{seconds // 60}:{seconds % 60}"\n'
    fixed = header + '    return f"{seconds // 60}:{seconds % 60:02d}"\n'
    visible = [
        f'def test_two_digit_seconds_are_unchanged():\n    assert {symbol}(130) == "2:10"\n',
        f'def test_single_digit_seconds_are_padded():\n    assert {symbol}(125) == "2:05"\n',
    ]
    contract = f'''
    assert {symbol}(0) == "0:00"
    assert {symbol}(9) == "0:09"
    assert {symbol}(60) == "1:00"
    assert {symbol}(3725) == "62:05"
    '''
    regression = f'''
    assert {symbol}(130) == "2:10"
    assert {symbol}(599) == "9:59"
    '''
    request = rng.choice([
        f"{symbol}() renders 125 seconds as \"2:5\" instead of \"2:05\". Pad the seconds and run the unit tests.",
        f"Durations from {symbol}() lose the zero padding on single-digit seconds. Fix it minimally and verify with the tests.",
    ])
    return _build("duration_format", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng)


def _family_matrix_transpose(variant: int) -> AgentTask:
    rng = _rng("matrix_transpose", variant)
    symbol = rng.choice(["transpose", "swap_axes", "columns_of"])
    module_name = rng.choice(["matrix", "grids", "tabular"])
    header = (
        f'def {symbol}(rows):\n'
        f'    """Return the matrix with rows and columns swapped, for any shape."""\n'
    )
    buggy = header + (
        "    size = len(rows)\n"
        "    return [[rows[index][column] for index in range(size)] for column in range(size)]\n"
    )
    fixed = header + "    return [list(column) for column in zip(*rows)]\n"
    visible = [
        (
            "def test_square_matrix():\n"
            f"    assert {symbol}([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n"
        ),
        (
            "def test_wide_matrix_keeps_every_column():\n"
            f"    assert {symbol}([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]\n"
        ),
    ]
    contract = f'''
    assert {symbol}([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert {symbol}([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
    assert {symbol}([[7, 8, 9]]) == [[7], [8], [9]]
    assert {symbol}([]) == []
    '''
    regression = f'''
    assert {symbol}([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert {symbol}([]) == []
    '''
    request = rng.choice([
        f"{symbol}() only works on square matrices; a 2x3 input silently loses a column. Make it shape-agnostic and run the unit tests.",
        f"Transposing a non-square matrix with {symbol}() drops data or raises. Fix it minimally and verify with the tests.",
    ])
    return _build("matrix_transpose", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng)


def _family_path_join(variant: int) -> AgentTask:
    rng = _rng("path_join", variant)
    symbol = rng.choice(["join_path", "resolve_path", "under"])
    module_name = rng.choice(["paths", "locations", "fsutil"])
    header = (
        f'def {symbol}(base, segment):\n'
        f'    """Join segment onto base; an absolute segment replaces the base entirely."""\n'
    )
    buggy = header + '    return base.rstrip("/") + "/" + segment\n'
    fixed = header + (
        '    if segment.startswith("/"):\n'
        "        return segment\n"
        "    if not segment:\n"
        '        return base.rstrip("/") or "/"\n'
        '    return base.rstrip("/") + "/" + segment\n'
    )
    visible = [
        (
            "def test_relative_segment_is_appended():\n"
            f'    assert {symbol}("/srv/app", "static") == "/srv/app/static"\n'
        ),
        (
            "def test_absolute_segment_replaces_the_base():\n"
            f'    assert {symbol}("/srv/app", "/etc/hosts") == "/etc/hosts"\n'
        ),
    ]
    contract = f'''
    assert {symbol}("/srv/app", "/etc/hosts") == "/etc/hosts"
    assert {symbol}("/srv/app/", "static") == "/srv/app/static"
    assert {symbol}("/srv/app/", "") == "/srv/app"
    assert {symbol}("/srv/app", "static/css") == "/srv/app/static/css"
    '''
    regression = f'''
    assert {symbol}("/srv/app/", "static") == "/srv/app/static"
    assert {symbol}("/srv/app", "static") == "/srv/app/static"
    '''
    request = rng.choice([
        f"{symbol}() concatenates an absolute segment onto the base instead of letting it replace the base. Fix it and run the unit tests.",
        f"An absolute second argument to {symbol}() should win outright, but it is appended. Fix it minimally and verify with the tests.",
    ])
    return _build("path_join", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng)


def _family_case_insensitive_lookup(variant: int) -> AgentTask:
    rng = _rng("case_insensitive_lookup", variant)
    symbol = rng.choice(["lookup", "find_value", "get_setting"])
    module_name = rng.choice(["registry", "headers", "lookup_table"])
    header = (
        f'def {symbol}(mapping, key):\n'
        f'    """Find a value by key, ignoring case on the query and on stored keys."""\n'
    )
    buggy = header + "    return mapping.get(key.lower())\n"
    fixed = header + (
        "    lowered = {stored.lower(): value for stored, value in mapping.items()}\n"
        "    return lowered.get(key.lower())\n"
    )
    visible = [
        (
            "def test_uppercase_query_finds_lowercase_key():\n"
            f'    assert {symbol}({{"host": "example"}}, "HOST") == "example"\n'
        ),
        (
            "def test_lowercase_query_finds_mixed_case_key():\n"
            f'    assert {symbol}({{"Host": "example"}}, "host") == "example"\n'
        ),
    ]
    contract = f'''
    assert {symbol}({{"Content-Type": "json"}}, "content-type") == "json"
    assert {symbol}({{"RETRIES": 3}}, "Retries") == 3
    assert {symbol}({{"host": "example"}}, "HOST") == "example"
    assert {symbol}({{"host": "example"}}, "missing") is None
    '''
    regression = f'''
    assert {symbol}({{"host": "example"}}, "host") == "example"
    assert {symbol}({{}}, "anything") is None
    '''
    request = rng.choice([
        f"{symbol}() lower-cases the query but not the stored keys, so a mixed-case key is never found. Fix it and run the unit tests.",
        f"Case-insensitive lookup in {symbol}() only works when the stored key is already lower case. Fix it minimally and verify with the tests.",
    ])
    return _build(
        "case_insensitive_lookup", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng
    )


def _family_histogram(variant: int) -> AgentTask:
    rng = _rng("histogram", variant)
    symbol = rng.choice(["bucket_index", "bucket_of", "index_for"])
    module_name = rng.choice(["histogram", "buckets", "binning"])
    header = (
        f'def {symbol}(value, low, high, buckets):\n'
        f'    """Return the 0-based bucket for value in [low, high]; high lands in the last bucket."""\n'
    )
    buggy = header + "    return int((value - low) / (high - low) * buckets)\n"
    fixed = header + (
        "    index = int((value - low) / (high - low) * buckets)\n"
        "    return min(index, buckets - 1)\n"
    )
    visible = [
        f"def test_interior_value():\n    assert {symbol}(5, 0, 10, 5) == 2\n",
        f"def test_maximum_lands_in_the_last_bucket():\n    assert {symbol}(10, 0, 10, 5) == 4\n",
    ]
    contract = f'''
    assert {symbol}(0, 0, 10, 5) == 0
    assert {symbol}(10, 0, 10, 5) == 4
    assert {symbol}(9.9, 0, 10, 5) == 4
    assert {symbol}(4, 0, 8, 4) == 2
    '''
    regression = f'''
    assert {symbol}(5, 0, 10, 5) == 2
    assert {symbol}(0, 0, 10, 5) == 0
    '''
    request = rng.choice([
        f"{symbol}() returns an out-of-range index when the value equals the upper bound. Clamp it into the last bucket and run the unit tests.",
        f"The maximum value handed to {symbol}() produces a bucket index one past the end. Fix it minimally and verify with the tests.",
    ])
    return _build("histogram", variant, module_name, symbol, buggy, fixed, visible, request, contract, regression, rng)


EVALUATION_FAMILY_BUILDERS = {
    "word_wrap": _family_word_wrap,
    "duration_format": _family_duration_format,
    "matrix_transpose": _family_matrix_transpose,
    "path_join": _family_path_join,
    "case_insensitive_lookup": _family_case_insensitive_lookup,
    "histogram": _family_histogram,
}

EVALUATION_VARIANTS_PER_FAMILY = 2


def evaluation_tasks(variants_per_family: int = EVALUATION_VARIANTS_PER_FAMILY) -> list[AgentTask]:
    """The frozen held-out suite: six families, deterministic variants."""
    return [
        EVALUATION_FAMILY_BUILDERS[family](variant)
        for family in EVALUATION_FAMILY_BUILDERS
        for variant in range(variants_per_family)
    ]


def task_from_fixture(fixture: FixtureTask) -> AgentTask:
    """Adapt a training fixture into a task, for collection smoke runs.

    These have no hidden verifier: the strong tests are the only grader, and
    they are visible. Collection reports say so per task, because a corpus
    verified only by tests the agent could edit is weaker evidence than one
    graded from outside the workspace.
    """
    return AgentTask(
        task_id=f"fixture/{fixture.family}-{fixture.variant:03d}",
        family=fixture.family,
        variant=fixture.variant,
        request=fixture.request,
        developer=fixture.developer,
        files={fixture.module_path: fixture.buggy_module, fixture.tests_path: fixture.strong_tests},
        protected_paths=(fixture.tests_path,),
        module_path=fixture.module_path,
        tests_path=fixture.tests_path,
        reference_module=fixture.fixed_module,
    )
