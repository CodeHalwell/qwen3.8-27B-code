"""Rooted six-tool executor over a repository directory.

Importable twin of notebook 01's ``execute_tool``: same tool surface, path
containment, output bounds, patch normalisation and test invocation, so that
trajectories generated through this class match what the deployed agent will
observe. Behavioural parity with the notebook cell is exercised in
``tests/test_bootstrap_corpus.py``.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path

_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")

# Every variable that can lead task code back to a credential cache on disk.
_HOME_KEYS = ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE")

# Repository code under test has no reason to reach the Hub.
_OFFLINE_FLAGS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

READ_LIMIT_BYTES = 20_000
SEARCH_SCAN_LIMIT_BYTES = 5_000_000
SEARCH_MATCH_LIMIT = 200
TEST_OUTPUT_TAIL = 12_000
# A shell observation keeps its head and tail inside this bound.
SHELL_OUTPUT_LIMIT = 12_000
SHELL_TIMEOUT_SECONDS = 120
TRUNCATION_MARKER = "\n... [output trimmed] ...\n"


def trim_output(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` inside ``limit`` characters."""
    if len(text) <= limit:
        return text
    keep = (limit - len(TRUNCATION_MARKER)) // 2
    return text[:keep] + TRUNCATION_MARKER + text[-keep:]


def format_command_observation(returncode: int | None, output: str, limit: int = SHELL_OUTPUT_LIMIT) -> str:
    """The ``shell`` observation: a non-zero exit code first, then the output.

    Training rows converted from third-party shell transcripts go through
    the same function, so what the model learns to expect is what this
    executor returns.
    """
    prefix = "" if returncode in (None, 0) else f"[exit code {returncode}]\n"
    return trim_output(prefix + output, limit)


def default_test_command() -> list[str]:
    return [sys.executable, "-m", "pytest", "-q"]


def filtered_environment(home: str | Path | None = None) -> dict[str, str]:
    """Build the environment repository code runs under.

    Dropping variables whose names look secret is not sufficient on its own:
    ``huggingface_hub.login`` writes the Hub token to a file under ``HOME``,
    so any test the model runs could read the token straight off disk. Point
    the cache-bearing variables at a scratch home and disable Hub network
    access for task subprocesses.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }
    # A patched file with unchanged size and same-second mtime revalidates a
    # stale .pyc (the bytecode header stores whole-second mtimes), so two test
    # runs bracketing a fast patch can execute the pre-patch module. Disable
    # bytecode caching inside fixture repositories entirely.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if home is not None:
        scratch_home = Path(home).resolve()
        scratch_home.mkdir(parents=True, exist_ok=True)
        environment.update({key: str(scratch_home) for key in _HOME_KEYS})
    environment.update(_OFFLINE_FLAGS)
    return environment


class RepoHarness:
    """Execute the six deployment tools against one repository root."""

    def __init__(
        self,
        root: str | Path,
        visible_test_command: list[str] | None = None,
        environment: dict[str, str] | None = None,
    ):
        self.root = Path(root).resolve()
        self.visible_test_command = list(visible_test_command or default_test_command())
        if environment is not None:
            self.environment = dict(environment)
        else:
            self.environment = filtered_environment(home=self.root.parent / "task_home")

    def _rooted(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes repository root")
        return candidate

    def execute(self, name: str, arguments: dict) -> str:
        if name == "list_files":
            base = self._rooted(arguments["path"])
            files = [
                str(path.relative_to(self.root))
                for path in base.rglob("*")
                if path.is_file() and ".git" not in path.parts
            ]
            return "\n".join(files[:SEARCH_MATCH_LIMIT]) or "[no files]"
        if name == "read_file":
            return self._rooted(arguments["path"]).read_text(errors="replace")[:READ_LIMIT_BYTES]
        if name == "search":
            return self._search(arguments["query"])
        if name == "apply_patch":
            return self._apply_patch(arguments["patch"])
        if name == "run_tests":
            if arguments["profile"] != "unit":
                return "unknown test profile"
            return self._run_tests()
        if name == "shell":
            return self._shell(arguments["command"])
        return f"unknown tool: {name}"

    def _search(self, query: str) -> str:
        try:
            regex = re.compile(query)
        except re.error as exc:
            return f"invalid regular expression: {exc}"
        hits: list[str] = []
        scanned_bytes = 0
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                payload = path.read_bytes()
            except OSError as exc:
                hits.append(f"{path.relative_to(self.root)}:read_error:{exc}")
                continue
            if b"\x00" in payload:
                continue
            scanned_bytes += len(payload)
            if scanned_bytes > SEARCH_SCAN_LIMIT_BYTES:
                hits.append("[search truncated after 5 MB]")
                break
            for line_no, line in enumerate(payload.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{path.relative_to(self.root)}:{line_no}:{line}")
                    if len(hits) >= SEARCH_MATCH_LIMIT:
                        hits.append("[search truncated after 200 matches]")
                        return "\n".join(hits)[:READ_LIMIT_BYTES]
        return "\n".join(hits)[:READ_LIMIT_BYTES] or "[no matches]"

    def _apply_patch(self, patch: str) -> str:
        # The XML parameter parser cannot distinguish a patch's final newline
        # from tag whitespace, and git apply calls a patch whose last line
        # lacks one "corrupt". Normalise to exactly one trailing newline.
        normalized = patch.rstrip("\r\n") + "\n"
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=self.root,
            input=normalized,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return "patch applied" if result.returncode == 0 else f"patch rejected: {result.stderr[:4000]}"

    def _run_tests(self) -> str:
        result = subprocess.run(
            self.visible_test_command,
            cwd=self.root,
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        return f"exit={result.returncode}\n{(result.stdout + result.stderr)[-TEST_OUTPUT_TAIL:]}"

    def _shell(self, command: str) -> str:
        # The same scrubbed environment, working directory, time limit and
        # bounded observation as run_tests. run_tests already executes
        # whatever the repository and the model's patches contain, so a
        # command here adds no exposure a test file could not. The command
        # runs in its own process group so that a timeout kills whatever
        # it started, not only bash.
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=self.root,
            env=self.environment,
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            output, _ = process.communicate(timeout=SHELL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            return f"[timed out after {SHELL_TIMEOUT_SECONDS}s]"
        return format_command_observation(process.returncode, output)
