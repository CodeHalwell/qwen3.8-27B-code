"""Synthesise execution-derived preference pairs for the DPO stage.

Each pair matches notebook 04's expected columns (``prompt_messages``,
``chosen_message``, ``rejected_message``, rewards, ``infra_status``) plus
audit fields recording the execution evidence behind the winner, per
docs/data-strategy.md. Where a pair contrasts patches, both candidate patches
are actually applied and tested in a fresh fixture; a pair is only emitted
when the executions come out the way the rewards claim.
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
from .trajectories import GENERATOR_VERSION, GenerationError, unified_patch

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
    prompt = [
        {"role": "developer", "content": task.developer},
        {"role": "user", "content": task.request},
    ]
    if contrast == "patch_outcome":
        chosen_exit = _run_state(task, task.fixed_module)
        rejected_exit = _run_state(task, task.partial_module)
        _expect(chosen_exit == "exit=0", f"{pair_id}: gold fix did not pass ({chosen_exit})")
        _expect(rejected_exit != "exit=0", f"{pair_id}: partial fix unexpectedly passed")
        chosen = task.summary
        rejected = f"{task.partial_reasoning} I applied exactly that change and consider the task complete."
        evidence = {
            "basis": "both candidate patches executed against the unit suite",
            "chosen_run": chosen_exit,
            "rejected_run": rejected_exit,
        }
    elif contrast == "test_integrity":
        buggy_exit = _run_state(task, None)
        _expect(buggy_exit != "exit=0", f"{pair_id}: fixture bug is not observable")
        prompt = [
            {"role": "developer", "content": task.developer},
            {"role": "user", "content": task.request + " The unit suite is currently failing."},
        ]
        chosen = task.summary
        rejected = "The failing test looks stricter than the code needs to be; I deleted it so the suite is green again."
        evidence = {
            "basis": "the failing test detects a real bug",
            "buggy_run": buggy_exit,
            "policy": "removing required tests hides the defect instead of fixing it",
        }
    elif contrast == "verification_claim":
        chosen_exit = _run_state(task, task.fixed_module)
        _expect(chosen_exit == "exit=0", f"{pair_id}: gold fix did not pass ({chosen_exit})")
        chosen = task.summary
        rejected = f"{task.bug_reasoning} I made that change; it is obviously correct, so I did not run the tests."
        evidence = {
            "basis": "identical edit; only the chosen answer verified it",
            "chosen_run": chosen_exit,
            "policy": "completion requires an executed unit run, not a plausibility argument",
        }
    elif contrast == "inspect_first":
        chosen = f"I would first read {task.module_path} and the failing test to confirm the cause before editing anything."
        rejected = f"Rather than digging through it, I would rewrite {task.module_path} from scratch to be safe."
        evidence = {
            "basis": "process preference from the scope policy",
            "policy": "diagnose from evidence and keep patches minimal; wholesale rewrites are out of scope",
        }
    else:
        raise GenerationError(f"unknown contrast {contrast!r}")

    return {
        "id": pair_id,
        "source": GENERATOR_VERSION,
        "repo_family": task.family,
        "contrast_type": contrast,
        "prompt_messages": prompt,
        "chosen_message": {"role": "assistant", "content": chosen},
        "rejected_message": {"role": "assistant", "content": rejected},
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
        "execution": "patch-based contrasts ran both candidate patches through the unit suite; rewards mirror observed outcomes",
        "limits": [
            "single-turn assistant continuations; no multi-turn trajectory preferences yet",
            "inspect_first pairs are policy-based rather than execution-derived",
            "prompts share fixture families with the SFT corpus (parameter-disjoint variants)",
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
