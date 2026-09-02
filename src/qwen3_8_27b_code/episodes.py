"""One bounded agent episode, with the model call injected.

Collection and evaluation are the same loop with different bookkeeping, so the
loop lives here once and takes a ``Policy``: a callable that turns the message
history into one assistant turn. Everything except that callable runs on CPU,
which is what makes the collector and the evaluator testable without a GPU.

The policy is also the only place the GPU-bound notebook has to supply code,
so the loop cannot drift between the baseline run, trace collection and the
evaluation gate the way three hand-copied cells would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Callable, Protocol

from .harness import RepoHarness
from .parsing import parse_tool_calls, split_reasoning
from .schema import validate_tool_call

# Every way an episode can stop. Budget exhaustion and truncation are
# first-class outcomes, not generic exceptions: docs/agentic-harness.md
# requires them to be distinguishable from a wrong patch.
TERMINATIONS = frozenset(
    {
        "assistant_complete",
        "tool_budget",
        "timeout",
        "output_truncated",
        "context_budget",
        "policy_error",
    }
)

# Outcomes caused by the harness rather than the model. They must never become
# negative rewards or rejected preference examples.
INFRASTRUCTURE_TERMINATIONS = frozenset({"policy_error"})


@dataclass(frozen=True)
class TurnResult:
    """One assistant turn, plus how generation ended.

    ``fault`` is ``output_truncated`` when the turn hit the token cap without
    an end-of-turn token, and ``context_budget`` when the prompt no longer
    leaves room to generate. A truncated turn parses as "no tool calls", so a
    loop that ignores this scores a cut-off turn as a finished answer.
    """

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fault: str | None = None


class Policy(Protocol):
    def __call__(self, messages: list[dict]) -> TurnResult: ...


@dataclass(frozen=True)
class EpisodeBudget:
    tool_calls: int = 10
    wall_seconds: float = 480.0


@dataclass
class Episode:
    """The full record of one attempt."""

    messages: list[dict] = field(default_factory=list)
    termination: str = "assistant_complete"
    final_text: str = ""
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    repeated_calls: int = 0
    tool_errors: int = 0
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_seconds: float = 0.0

    @property
    def called_tools(self) -> list[str]:
        return [
            call["function"]["name"]
            for message in self.messages
            if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        ]

    @property
    def ran_tests(self) -> bool:
        return "run_tests" in self.called_tools

    @property
    def valid_tool_call_rate(self) -> float:
        attempted = self.tool_calls + self.invalid_tool_calls
        return 1.0 if not attempted else self.tool_calls / attempted

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.termination in INFRASTRUCTURE_TERMINATIONS

    def usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "repeated_calls": self.repeated_calls,
            "tool_errors": self.tool_errors,
            "turns": self.turns,
            "wall_seconds": round(self.wall_seconds, 3),
        }


def _call_fingerprint(name: str, arguments: dict) -> str:
    return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)


def run_episode(
    harness: RepoHarness,
    policy: Policy,
    request: str,
    developer: str,
    budget: EpisodeBudget | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Episode:
    """Drive one episode against an already-materialised repository."""
    budget = budget or EpisodeBudget()
    started = clock()
    episode = Episode(
        messages=[
            {"role": "developer", "content": developer},
            {"role": "user", "content": request},
        ]
    )
    previous_fingerprint: str | None = None

    while True:
        if clock() - started > budget.wall_seconds:
            episode.termination = "timeout"
            break
        if episode.tool_calls + episode.invalid_tool_calls >= budget.tool_calls:
            episode.termination = "tool_budget"
            break

        try:
            turn = policy(episode.messages)
        except Exception as exc:  # noqa: BLE001 - recorded, never scored
            episode.termination = "policy_error"
            episode.final_text = f"{type(exc).__name__}: {exc}"
            break

        episode.turns += 1
        episode.prompt_tokens += turn.prompt_tokens
        episode.completion_tokens += turn.completion_tokens
        if turn.fault is not None:
            episode.termination = turn.fault
            break

        reasoning, calls = parse_tool_calls(turn.text)
        if not calls:
            _, final_text = split_reasoning(turn.text)
            episode.messages.append(
                {"role": "assistant", "reasoning_content": reasoning, "content": final_text}
            )
            episode.final_text = final_text
            episode.termination = "assistant_complete"
            break

        episode.messages.append(
            {"role": "assistant", "reasoning_content": reasoning, "content": "", "tool_calls": calls}
        )
        for call in calls:
            function = call["function"]
            name, arguments = function["name"], function.get("arguments", {})

            fingerprint = _call_fingerprint(name, arguments)
            if fingerprint == previous_fingerprint:
                episode.repeated_calls += 1
            previous_fingerprint = fingerprint

            # A malformed call becomes a typed observation so the episode can
            # show recovery, which is the behaviour worth training.
            schema_error = validate_tool_call(name, arguments)
            if schema_error is not None:
                episode.invalid_tool_calls += 1
                episode.messages.append(
                    {"role": "tool", "name": name, "content": f"invalid_tool_call: {schema_error}"}
                )
                continue

            try:
                observation = harness.execute(name, arguments)
            except Exception as exc:  # noqa: BLE001 - a tool fault, not a crash
                episode.tool_errors += 1
                observation = f"tool_error: {type(exc).__name__}: {exc}"
            episode.tool_calls += 1
            episode.messages.append({"role": "tool", "name": name, "content": observation})

    episode.wall_seconds = clock() - started
    return episode


def scripted_policy(turns: list[str], on_exhaustion: str | None = None) -> Policy:
    """A policy that replays fixed assistant turns.

    Used by the tests and by ``--policy scripted-gold`` to exercise the whole
    collection and evaluation path without a GPU.
    """
    remaining = list(turns)

    def policy(messages: list[dict]) -> TurnResult:
        del messages
        if remaining:
            return TurnResult(text=remaining.pop(0))
        if on_exhaustion is None:
            raise RuntimeError("scripted policy ran out of turns")
        return TurnResult(text=on_exhaustion)

    return policy


def tool_call_text(name: str, arguments: dict, reasoning: str = "") -> str:
    """Render a turn in Qwen3.8's native XML tool-call syntax."""
    parameters = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments.items()
    )
    return (
        f"<think>\n{reasoning}\n</think>\n\n"
        f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call><|im_end|>"
    )


def answer_text(content: str, reasoning: str = "") -> str:
    return f"<think>\n{reasoning}\n</think>\n\n{content}<|im_end|>"
