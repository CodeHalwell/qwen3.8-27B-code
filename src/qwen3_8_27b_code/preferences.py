"""Synthesise execution-derived preference pairs for the DPO stage.

Each pair matches notebook 04's expected columns (``prompt_messages``,
``chosen_message``, ``rejected_message``, rewards, ``infra_status``) plus
audit fields recording the execution evidence behind the winner, per
docs/data-strategy.md.

The behaviour being rewarded must be visible to the trainer, not only to the
generator: every prompt is a real executed prefix (assistant tool calls plus
verbatim harness observations, produced by actually running the fixture), and
the two continuations diverge at that state. The chosen continuation is the
correct next tool call — never a claim that untraced work already happened —
and where the contrast is between candidate patches, both branches are
actually applied and tested before the pair is emitted.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .fixtures import FAMILY_BUILDERS, FixtureTask
from .harness import RepoHarness
from .trajectories import (
    GENERATOR_VERSION,
    GenerationError,
    _EpisodeBuilder,
    _patch_and_expect,
    _run_and_expect,
    unified_patch,
)

# Variants 100+ keep preference prompts parameter-disjoint from the SFT
# corpus, which uses variants 0..VARIANTS_PER_FAMILY-1 of the same families.
PREFERENCE_VARIANT_BASE = 100
PAIRS_PER_FAMILY = 5

CONTRAST_CYCLE = [
    "patch_outcome",
    "test_integrity",
    "patch_outcome",
    "verification_claim",
    "inspect_first",
]


def _tool_call_message(name: str, arguments: dict, reasoning: str) -> dict:
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def _materialise(task: FixtureTask) -> tuple[Path, RepoHarness]:
    tmp = Path(tempfile.mkdtemp(prefix="qwen38_pref_")) / "repo"
    for path, content in {task.module_path: task.buggy_module, task.tests_path: task.strong_tests}.items():
        target = tmp / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return tmp, RepoHarness(tmp)


def _run_state(task: FixtureTask, module_source: str | None) -> str:
    """Run the strong tests with the module in the given state; return exit line."""
    tmp, harness = _materialise(task)
    try:
        if module_source is not None and module_source != task.buggy_module:
            patch = unified_patch(task.module_path, task.buggy_module, module_source)
            observation = harness.execute("apply_patch", {"patch": patch})
            if observation != "patch applied":
                raise GenerationError(f"{task.family}: candidate patch rejected\n{observation[:1000]}")
        result = harness.execute("run_tests", {"profile": "unit"})
        return result.splitlines()[0]
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _expect(condition: bool, detail: str) -> None:
    if not condition:
        raise GenerationError(detail)


def _build_pair(task: FixtureTask, contrast: str, pair_id: str) -> dict:
    builder = _EpisodeBuilder(
        task,
        {task.module_path: task.buggy_module, task.tests_path: task.strong_tests},
        task.developer,
        task.request,
    )
    try:
        if contrast == "patch_outcome":
            # State: implementation inspected, nothing edited. The pair chooses
            # between the gold minimal patch and a plausible incomplete one.
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            prompt = list(builder.messages)
            chosen_exit = _run_state(task, task.fixed_module)
            rejected_exit = _run_state(task, task.partial_module)
            _expect(chosen_exit == "exit=0", f"{pair_id}: gold fix did not pass ({chosen_exit})")
            _expect(rejected_exit != "exit=0", f"{pair_id}: partial fix unexpectedly passed")
            chosen = _tool_call_message(
                "apply_patch",
                {"patch": unified_patch(task.module_path, task.buggy_module, task.fixed_module)},
                task.bug_reasoning,
            )
            rejected = _tool_call_message(
                "apply_patch",
                {"patch": unified_patch(task.module_path, task.buggy_module, task.partial_module)},
                task.partial_reasoning,
            )
            evidence = {
                "basis": "both candidate patches executed from this state against the unit suite",
                "chosen_run": chosen_exit,
                "rejected_run": rejected_exit,
            }
        elif contrast == "verification_claim":
            # State: gold patch applied for real, tests not yet run. The pair
            # chooses between verifying and declaring success unverified.
            builder.call("read_file", {"path": task.module_path}, task.read_reasoning)
            _patch_and_expect(builder, task.module_path, task.buggy_module, task.fixed_module, task.bug_reasoning)
            prompt = list(builder.messages)
            pending_run = builder.harness.execute("run_tests", {"profile": "unit"})
            _expect(pending_run.startswith("exit=0"), f"{pair_id}: pending verification fails\n{pending_run[:1000]}")
            chosen = _tool_call_message("run_tests", {"profile": "unit"}, task.verify_reasoning)
            rejected = {
                "role": "assistant",
                "content": "That was the whole fix and it is obviously correct, so I did not run the tests. Marking this complete.",
            }
            evidence = {
                "basis": "the verification the chosen turn requests was executed from this exact state",
                "chosen_run": pending_run.splitlines()[0],
                "policy": "completion requires an executed unit run, not a plausibility argument",
            }
        elif contrast == "test_integrity":
            # State: a real failing run is in the transcript. The pair chooses
            # between fixing the code and deleting the failing test.
            failing_run = _run_and_expect(
                builder, "I should reproduce the failure before touching anything.", passing=False
            )
            prompt = list(builder.messages)
            chosen_exit = _run_state(task, task.fixed_module)
            _expect(chosen_exit == "exit=0", f"{pair_id}: gold fix did not pass ({chosen_exit})")
            chosen = _tool_call_message(
                "apply_patch",
                {"patch": unified_patch(task.module_path, task.buggy_module, task.fixed_module)},
                task.bug_reasoning,
            )
            rejected = {
                "role": "assistant",
                "content": "The failing test looks stricter than the code needs to be; I will delete it so the suite reports green again.",
            }
            evidence = {
                "basis": "the transcript shows a genuine failing run; the chosen patch passes it",
                "failing_run": failing_run.splitlines()[0],
                "chosen_run": chosen_exit,
                "policy": "removing required tests hides the defect instead of fixing it",
            }
        elif contrast == "inspect_first":
            # State: bare task. The pair chooses between inspecting the
            # implementation and announcing a wholesale rewrite.
            prompt = list(builder.messages)
            readable = builder.harness.execute("read_file", {"path": task.module_path})
            _expect(bool(readable.strip()), f"{pair_id}: module unreadable from bare state")
            chosen = _tool_call_message("read_file", {"path": task.module_path}, task.read_reasoning)
            rejected = {
                "role": "assistant",
                "content": f"Rather than digging through it, I will rewrite {task.module_path} from scratch to be safe.",
            }
            evidence = {
                "basis": "the chosen inspection executes from this state; process preference from the scope policy",
                "policy": "diagnose from evidence and keep patches minimal; wholesale rewrites are out of scope",
            }
        else:
            raise GenerationError(f"unknown contrast {contrast!r}")
    finally:
        builder.cleanup()

    return {
        "id": pair_id,
        "source": GENERATOR_VERSION,
        "repo_family": task.family,
        "contrast_type": contrast,
        "prompt_messages": prompt,
        "chosen_message": chosen,
        "rejected_message": rejected,
        "chosen_reward": 1.0,
        "rejected_reward": 0.0,
        "infra_status": "ok",
        "evidence": evidence,
    }


def generate_pairs(pairs_per_family: int = PAIRS_PER_FAMILY) -> list[dict]:
    pairs = []
    for family in FAMILY_BUILDERS:
        for offset in range(pairs_per_family):
            task = FAMILY_BUILDERS[family](PREFERENCE_VARIANT_BASE + offset)
            contrast = CONTRAST_CYCLE[offset % len(CONTRAST_CYCLE)]
            pair_id = f"pref/{family}-{contrast}-{offset:02d}"
            pairs.append(_build_pair(task, contrast, pair_id))
    return pairs


def quality_report(pairs: list[dict], corpus_path: Path) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "rows": len(pairs),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "families": dict(sorted(Counter(pair["repo_family"] for pair in pairs).items())),
        "contrast_types": dict(sorted(Counter(pair["contrast_type"] for pair in pairs).items())),
        "chosen_tool_call_pairs": sum(1 for pair in pairs if pair["chosen_message"].get("tool_calls")),
        "prompts_with_executed_observations": sum(
            1 for pair in pairs if any(message["role"] == "tool" for message in pair["prompt_messages"])
        ),
        "execution": (
            "prompts are real executed prefixes (verbatim harness observations); "
            "patch contrasts ran both candidate patches through the unit suite and "
            "rewards mirror observed outcomes"
        ),
        "limits": [
            "chosen continuations are single next-turn tool calls; no multi-turn preference rollouts yet",
            "inspect_first pairs are grounded by an executable read but ranked by the scope policy",
            "prompts share fixture families with the SFT corpus (parameter-disjoint variants)",
            "pytest observations in prompts embed wall-clock timings, so regeneration is structurally but not byte identical",
        ],
    }


def write_pairs(out_path: Path, report_path: Path, pairs_per_family: int = PAIRS_PER_FAMILY) -> dict:
    pairs = generate_pairs(pairs_per_family)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    report = quality_report(pairs, out_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
