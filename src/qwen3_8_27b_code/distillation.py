"""Outcome preference pairs across policies: what a teacher did that the student did not.

A teacher's verified trajectories feed SFT through the collector. The
attempts that did *not* verify are still evidence, and so are the student's
own: at the first turn where a verified attempt and a failed attempt at the
same task diverge, the verified continuation is preferred. Both continuations
were executed and graded from outside the workspace, so the pair is
execution-derived in the sense docs/data-strategy.md requires, whichever
policy produced each side.

That makes the same builder serve three uses. Teacher against student
(distillation proper), student against itself (the candidate-generation mode
of docs/agentic-harness.md), and teacher against teacher across seeds. The
evidence names the policy on each side so a DPO mixture can be audited for
how much of it is one or the other.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .thinking import _action_key, assistant_turns

OUTCOME_PAIRS_VERSION = "outcome-pairs-v1"
DEFAULT_MAX_OUTCOME_PAIRS_PER_TASK = 4


def _divergence(success, failure) -> tuple[int, int, int] | None:
    """(turn index, chosen message index, rejected message index) at the first
    assistant turn where the two attempts act differently, or None."""
    turns_a, turns_b = assistant_turns(success.episode), assistant_turns(failure.episode)
    for turn_index, ((index_a, message_a), (index_b, message_b)) in enumerate(zip(turns_a, turns_b)):
        if _action_key(message_a) != _action_key(message_b):
            return turn_index, index_a, index_b
    return None


def build_outcome_pairs(attempts, *, max_pairs_per_task: int = DEFAULT_MAX_OUTCOME_PAIRS_PER_TASK) -> list[dict]:
    """Verified-versus-failed pairs at the same task from any mix of attempts.

    Infrastructure failures are never a rejected side: the harness, not the
    model, produced them. Every other non-verified attempt is eligible,
    including the ones that edited the tests or claimed success unverified,
    because those are precisely the behaviours the preference stage exists to
    move away from.
    """
    from .collection import rejection_reason  # collection imports thinking, not this module

    by_task: dict[str, list] = {}
    for attempt in attempts:
        by_task.setdefault(attempt.task_id, []).append(attempt)

    pairs = []
    for _, group in sorted(by_task.items()):
        successes, failures = [], []
        for attempt in group:
            reason = rejection_reason(attempt.episode, attempt.verdict)
            if reason is None:
                successes.append(attempt)
            elif reason != "infrastructure_failure":
                failures.append((attempt, reason))
        task_pairs = []
        for success in successes:
            for failure, reason in failures:
                located = _divergence(success, failure)
                if located is None:
                    continue
                turn_index, chosen_index, rejected_index = located
                chosen = dict(success.episode.messages[chosen_index])
                rejected = dict(failure.episode.messages[rejected_index])
                task_slug = success.task_id.replace("/", "-")
                task_pairs.append(
                    {
                        "id": (
                            f"pref/{task_slug}-outcome-t{turn_index:02d}"
                            f"-{_policy_slug(success)}-s{success.seed}-vs-{_policy_slug(failure)}-s{failure.seed}"
                        ),
                        "source": OUTCOME_PAIRS_VERSION,
                        "repo_family": success.family,
                        "contrast_type": f"outcome:{reason}",
                        "prompt_messages": [dict(message) for message in success.episode.messages[:chosen_index]],
                        "chosen_message": chosen,
                        "rejected_message": rejected,
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "infra_status": "ok",
                        "evidence": {
                            "basis": (
                                "both attempts executed from the same task; identical actions up to "
                                "this turn, then the chosen continuation led to a verified success and "
                                "the rejected one did not"
                            ),
                            "task_id": success.task_id,
                            "turn_index": turn_index,
                            "shared_action_turns": turn_index,
                            "prefix_source": "chosen",
                            "chosen_policy": success.policy,
                            "rejected_policy": failure.policy,
                            "chosen_seed": success.seed,
                            "rejected_seed": failure.seed,
                            "chosen_verdict": success.verdict.as_dict(),
                            "rejected_verdict": failure.verdict.as_dict(),
                            "rejected_termination": failure.episode.termination,
                            "rejection": reason,
                            "hidden_verified": bool(success.verdict.hidden),
                            # Form-matched pairs (a tool call on both sides) are
                            # the ones DPO cannot win on message shape alone.
                            "form_matched": bool(chosen.get("tool_calls")) == bool(rejected.get("tool_calls")),
                        },
                    }
                )
        # Form-matched and earlier-diverging pairs first when a task has many.
        task_pairs.sort(
            key=lambda pair: (not pair["evidence"]["form_matched"], pair["evidence"]["turn_index"], pair["id"])
        )
        pairs.extend(task_pairs[:max_pairs_per_task])
    return pairs


def _policy_slug(attempt) -> str:
    return (attempt.policy or "policy").replace("/", "-").replace(":", "-").replace(" ", "")


def outcome_pairs_report(pairs: list[dict], corpus_path: Path | None = None) -> dict:
    sides = Counter()
    for pair in pairs:
        sides[f"{pair['evidence']['chosen_policy'] or '?'} > {pair['evidence']['rejected_policy'] or '?'}"] += 1
    return {
        "generator_version": OUTCOME_PAIRS_VERSION,
        "rows": len(pairs),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest() if corpus_path else None,
        "families": dict(sorted(Counter(pair["repo_family"] for pair in pairs).items())),
        "contrast_types": dict(sorted(Counter(pair["contrast_type"] for pair in pairs).items())),
        "policy_pairings": dict(sorted(sides.items())),
        "form_matched_pairs": sum(1 for pair in pairs if pair["evidence"]["form_matched"]),
        "prose_rejections": sum(1 for pair in pairs if not pair["rejected_message"].get("tool_calls")),
        "hidden_verified_pairs": sum(1 for pair in pairs if pair["evidence"]["hidden_verified"]),
        "turn_index": {
            str(index): count
            for index, count in sorted(Counter(pair["evidence"]["turn_index"] for pair in pairs).items())
        },
        "execution": (
            "both continuations were executed in the harness and graded from outside the workspace; "
            "the chosen one verified, the rejected one was refused for the recorded reason"
        ),
        "limits": [
            "for turn_index > 0 the rejected turn was generated under its own prefix; the actions "
            "match up to the divergence, the token-level context does not",
            "a prose rejection (the failed attempt answered where the verified one acted) is "
            "form-asymmetric; keep such pairs a minority, as docs/data-strategy.md requires",
            "the rejected side is not sampled from the policy under training unless it is: read "
            "policy_pairings before assuming the pairs cover the student's own error distribution",
        ],
    }


def write_outcome_pairs(pairs: list[dict], out_path: Path, report_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    report = outcome_pairs_report(pairs, out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
