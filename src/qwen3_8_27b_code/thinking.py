"""Thinking budget: measure, select and train for shorter reasoning at equal quality.

The objective this module serves is stated in docs/thinking-budget.md: a
deployable coding agent should spend as few reasoning tokens as it can while
staying exactly as reliable, and "as reliable" is decided by the held-out gate,
never here. Everything below is therefore conditioned on verified success:

- ``reasoning_length`` is the per-episode length the collector sorts by when
  it keeps the shortest verified attempt at a task (shortest rejection
  sampling);
- ``build_reasoning_length_pairs`` turns several verified attempts at one task
  into DPO pairs whose two continuations take the *same* action and differ
  only in how much they think first; and
- ``length_rewards`` is the group-relative brevity term for GRPO, bounded so
  it can reorder correct samples among themselves but never lift an incorrect
  sample above a correct one.

Reasoning is measured in tokens when the policy counted them and in
characters otherwise; every artifact records which. Tokens are what the user
pays for, so the gate uses tokens; characters are the CPU-side proxy that keeps
the selection and the pair builder testable without a tokenizer.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Sequence

from .episodes import Episode

LENGTH_PAIRS_VERSION = "reasoning-length-v1"

# A pair is only informative when the two continuations think materially
# differently. The ratio guards against pairs that differ by a few words; the
# absolute gap guards against ratios inflated by a near-empty think block.
DEFAULT_MIN_LENGTH_RATIO = 1.5
DEFAULT_MIN_LENGTH_GAP = {"tokens": 32, "chars": 120}
DEFAULT_MAX_PAIRS_PER_TASK = 4


def reasoning_length(episode: Episode) -> tuple[int, str]:
    """Total reasoning spent in the episode, as ``(length, unit)``."""
    if episode.reasoning_tokens_reported:
        return episode.reasoning_tokens, "tokens"
    return episode.reasoning_chars, "chars"


def assistant_turns(episode: Episode) -> list[tuple[int, dict]]:
    """``(index in messages, message)`` for every assistant turn, in order."""
    return [
        (index, message)
        for index, message in enumerate(episode.messages)
        if message.get("role") == "assistant"
    ]


def turn_reasoning_lengths(episode: Episode, unit: str) -> list[int]:
    """Reasoning length of each completed assistant turn, in ``unit``."""
    turns = assistant_turns(episode)
    if unit == "tokens":
        counts = episode.turn_reasoning_tokens
        if len(counts) != len(turns) or any(count is None for count in counts):
            raise ValueError("per-turn reasoning tokens were not reported for every turn")
        return [int(count) for count in counts]
    if unit != "chars":
        raise ValueError(f"unknown reasoning unit {unit!r}")
    return [len(message.get("reasoning_content") or "") for _, message in turns]


def _action_key(message: dict) -> str | None:
    """Canonical form of the tool calls in a turn; None for a final answer."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    return json.dumps(
        [[call["function"]["name"], call["function"].get("arguments", {})] for call in calls],
        sort_keys=True,
        ensure_ascii=False,
    )


def _pairs_for_couple(short_first, second, unit: str, min_ratio: float, min_gap: int) -> list[dict]:
    """Turn-level pairs between two verified attempts at the same task.

    The two trajectories are walked in step while they take identical actions.
    At each such turn the continuation that reached the action with less
    reasoning is preferred. The walk stops at the first divergence (or at a
    final answer), because beyond it the two attempts no longer share a state
    to prefer between.
    """
    pairs = []
    turns_a, turns_b = assistant_turns(short_first.episode), assistant_turns(second.episode)
    lengths_a = turn_reasoning_lengths(short_first.episode, unit)
    lengths_b = turn_reasoning_lengths(second.episode, unit)
    for turn_index, ((index_a, message_a), (index_b, message_b)) in enumerate(zip(turns_a, turns_b)):
        action = _action_key(message_a)
        if action is None or action != _action_key(message_b):
            break
        if lengths_a[turn_index] <= lengths_b[turn_index]:
            chosen_attempt, chosen_index, chosen_length = short_first, index_a, lengths_a[turn_index]
            rejected_attempt, rejected_index, rejected_length = second, index_b, lengths_b[turn_index]
        else:
            chosen_attempt, chosen_index, chosen_length = second, index_b, lengths_b[turn_index]
            rejected_attempt, rejected_index, rejected_length = short_first, index_a, lengths_a[turn_index]
        if rejected_length < chosen_length * min_ratio or rejected_length - chosen_length < min_gap:
            continue

        task_slug = chosen_attempt.task_id.replace("/", "-")
        prompt = [dict(message) for message in chosen_attempt.episode.messages[:chosen_index]]
        pairs.append(
            {
                "id": (
                    f"pref/{task_slug}-reasoning_length-t{turn_index:02d}"
                    f"-s{chosen_attempt.seed}-vs-s{rejected_attempt.seed}"
                ),
                "source": LENGTH_PAIRS_VERSION,
                "repo_family": chosen_attempt.family,
                "contrast_type": "reasoning_length",
                "prompt_messages": prompt,
                "chosen_message": dict(chosen_attempt.episode.messages[chosen_index]),
                "rejected_message": dict(rejected_attempt.episode.messages[rejected_index]),
                "chosen_reward": 1.0,
                # Both continuations succeeded; the rejected one is only worse
                # by how much longer it thought. Notebook 04 requires strict
                # inequality, which the gap guarantees.
                "rejected_reward": round(chosen_length / rejected_length, 4),
                "infra_status": "ok",
                "evidence": {
                    "basis": (
                        "both attempts are verified successes that took identical actions "
                        "through this turn; the chosen turn reaches the same action with "
                        "less reasoning"
                    ),
                    "unit": unit,
                    "chosen_reasoning_length": chosen_length,
                    "rejected_reasoning_length": rejected_length,
                    "ratio": round(rejected_length / chosen_length, 3) if chosen_length else None,
                    "turn_index": turn_index,
                    "shared_action_turns": turn_index,
                    "prefix_source": "chosen",
                    "task_id": chosen_attempt.task_id,
                    "chosen_seed": chosen_attempt.seed,
                    "rejected_seed": rejected_attempt.seed,
                    "chosen_total_reasoning": reasoning_length(chosen_attempt.episode)[0],
                    "rejected_total_reasoning": reasoning_length(rejected_attempt.episode)[0],
                    "hidden_verified": bool(chosen_attempt.verdict.hidden) and bool(rejected_attempt.verdict.hidden),
                },
            }
        )
    return pairs


def build_reasoning_length_pairs(
    attempts,
    *,
    min_ratio: float = DEFAULT_MIN_LENGTH_RATIO,
    min_gap: int | None = None,
    max_pairs_per_task: int = DEFAULT_MAX_PAIRS_PER_TASK,
) -> list[dict]:
    """Reasoning-length preference pairs from a collection's attempts.

    Only verified successes take part, re-derived here from the episode and
    verdict rather than trusted from the caller, so a pair can never prefer a
    shorter *wrong* answer. Attempts that were deduplicated as rows are still
    eligible: two attempts with the same actions and different reasoning are
    exactly the contrast this data exists to capture.
    """
    from .collection import rejection_reason  # collection imports this module

    verified = [attempt for attempt in attempts if rejection_reason(attempt.episode, attempt.verdict) is None]
    by_task: dict[str, list] = {}
    for attempt in verified:
        by_task.setdefault(attempt.task_id, []).append(attempt)

    pairs = []
    for _, group in sorted(by_task.items()):
        if len(group) < 2:
            continue
        unit = "tokens" if all(attempt.episode.reasoning_tokens_reported for attempt in group) else "chars"
        gap = DEFAULT_MIN_LENGTH_GAP[unit] if min_gap is None else min_gap
        ordered = sorted(group, key=lambda attempt: (reasoning_length(attempt.episode)[0], attempt.seed))
        task_pairs = []
        for first in range(len(ordered)):
            for second in range(first + 1, len(ordered)):
                task_pairs.extend(_pairs_for_couple(ordered[first], ordered[second], unit, min_ratio, gap))
        # Keep the most contrasting pairs when a task has many.
        task_pairs.sort(key=lambda pair: (-(pair["evidence"]["ratio"] or float("inf")), pair["id"]))
        pairs.extend(task_pairs[:max_pairs_per_task])
    return pairs


def length_pairs_report(pairs: list[dict], corpus_path: Path | None = None) -> dict:
    ratios = [pair["evidence"]["ratio"] for pair in pairs if pair["evidence"]["ratio"] is not None]
    turn_indices = Counter(pair["evidence"]["turn_index"] for pair in pairs)
    return {
        "generator_version": LENGTH_PAIRS_VERSION,
        "rows": len(pairs),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest() if corpus_path else None,
        "families": dict(sorted(Counter(pair["repo_family"] for pair in pairs).items())),
        "units": dict(sorted(Counter(pair["evidence"]["unit"] for pair in pairs).items())),
        "turn_index": {str(index): count for index, count in sorted(turn_indices.items())},
        "pairs_beyond_first_turn": sum(1 for pair in pairs if pair["evidence"]["turn_index"] > 0),
        "hidden_verified_pairs": sum(1 for pair in pairs if pair["evidence"]["hidden_verified"]),
        "ratio": {
            "median": round(statistics.median(ratios), 3) if ratios else None,
            "min": round(min(ratios), 3) if ratios else None,
            "max": round(max(ratios), 3) if ratios else None,
        },
        "chosen_reasoning_median": (
            statistics.median(pair["evidence"]["chosen_reasoning_length"] for pair in pairs) if pairs else None
        ),
        "rejected_reasoning_median": (
            statistics.median(pair["evidence"]["rejected_reasoning_length"] for pair in pairs) if pairs else None
        ),
        "execution": (
            "both continuations come from attempts that were executed and verified from outside "
            "the workspace; the pair prefers the one that reached the identical action with less reasoning"
        ),
        "limits": [
            "both sides succeeded, so this data teaches brevity only; correctness contrasts come from "
            "the execution-derived pairs in data/preferences",
            "for turn_index > 0 the rejected turn was generated under its own prefix, whose earlier "
            "reasoning and observation text differ from the chosen prefix the pair renders; the "
            "actions are identical, the token-level context is not",
            "pairs stop at the first divergent action or final answer, so late-episode thinking is "
            "under-represented relative to opening turns",
            "measured in characters when the policy did not report reasoning tokens",
        ],
    }


def write_length_pairs(pairs: list[dict], out_path: Path, report_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    report = length_pairs_report(pairs, out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def length_rewards(lengths: Sequence[int], succeeded: Sequence[bool], weight: float = 0.1) -> list[float]:
    """Group-relative brevity reward, gated on correctness.

    Within one GRPO group the shortest sample earns ``+weight/2`` and the
    longest ``-weight/2``, linearly in between (the long2short length reward
    of Kimi k1.5). An incorrect sample is clipped to at most zero, so brevity
    never rewards giving up early. With a correctness reward on ``[0, 1]`` and
    ``weight <= 1`` a correct sample always outranks an incorrect one: the
    term reorders correct samples among themselves and nothing else.
    """
    if len(lengths) != len(succeeded):
        raise ValueError("lengths and succeeded must align")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if not lengths:
        return []
    shortest, longest = min(lengths), max(lengths)
    rewards = []
    for length, ok in zip(lengths, succeeded):
        scaled = 0.0 if longest == shortest else 0.5 - (length - shortest) / (longest - shortest)
        reward = weight * scaled
        rewards.append(round(reward if ok else min(0.0, reward), 6))
    return rewards
