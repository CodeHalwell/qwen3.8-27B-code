"""Convert public Hub datasets into native-schema training rows.

Three sources, two lanes (docs/data-strategy.md):

* ``nvidia/Open-SWE-Traces`` — repository-agent trajectories that use one
  ``bash`` tool, which maps onto this harness's ``shell`` tool without
  loss, observation format included. Only resolved trajectories are
  kept, and each stops before the harness's final submit command, which
  nothing answers. They are long (a median of tens of thousands of
  tokens), so one over the token budget is cut into windows that together
  cover every turn once: the opening, then later runs of turns, each
  carrying an elision note naming the turns and commands that came before
  it so nothing vouches for edits it does not show. Tool outputs are
  trimmed.
* ``nvidia/OpenCodeInstruct`` — instruction and answer pairs with unit
  tests; only rows whose tests all passed are kept, as non-agentic rows.
* ``nvidia/OpenCodeReasoning`` — competitive-programming answers with the
  reasoning in ``<think>`` tags, split into the native reasoning field.

Every row carries the deployment tool schema so notebook 02's validator
treats it like a bootstrap row, a ``repo_family`` for the family-disjoint
split, and a ``reasoning_effort`` matching how the text was produced.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import json
import hashlib
import re
from collections import Counter

from .harness import TRUNCATION_MARKER as TRUNCATION_MARKER  # re-exported for callers
from .harness import format_command_observation, trim_output
from .schema import TOOL_SCHEMA_JSON, TOOL_SCHEMA_VERSION, TOOLS

CONVERTER_VERSION = "public-sources-v4"
TokenCounter = Callable[[str], int]

# Roughly one token per 3.6 characters of mixed code and prose; used only
# when no tokenizer is supplied.
CHARS_PER_TOKEN = 3.6
# A tool observation longer than this keeps its head and tail, in the
# executor's own format (harness.format_command_observation) at a training
# budget rather than the executor's bound.
MAX_TOOL_OUTPUT_CHARS = 3_000
# A non-agentic source names one domain for all of its rows, so its whole
# share of the corpus was a single repo_family and notebook 02's
# family-disjoint split could only take all of it or none. These rows are
# independent problems with no repository to leak, so the family is split
# into buckets by row id: disjoint between the splits, and small enough
# that holding some out is a choice rather than an avalanche.
FAMILY_BUCKETS = 32

SOURCE_OPEN_SWE = "nvidia/Open-SWE-Traces"
# mini-swe-agent ends an episode with this command; its observation is the
# end of the episode and is never recorded.
SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SOURCE_OPEN_CODE_INSTRUCT = "nvidia/OpenCodeInstruct"
SOURCE_OPEN_CODE_REASONING = "nvidia/OpenCodeReasoning"

# How each source is loaded: the ``datasets`` config and split to stream,
# at a pinned commit so one configuration always yields one corpus. The
# commit is recorded with the corpus (notebook 02 uploads the report as
# public_sources.json beside the dataset).
SOURCE_LOADERS = {
    SOURCE_OPEN_SWE: {
        "name": "v1.2", "split": "minisweagent",
        "revision": "31cfd32021f674a1bbd5ff9f56a2151436fe2be3", "reasoning_effort": "xhigh",
    },
    SOURCE_OPEN_CODE_INSTRUCT: {
        "name": None, "split": "train",
        "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "reasoning_effort": "low",
    },
    SOURCE_OPEN_CODE_REASONING: {
        "name": "split_0", "split": "split_0",
        "revision": "20a1ca19c0d050fe9057fc08339d6b370ec1c67a", "reasoning_effort": "xhigh",
    },
}


def approximate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _base_row(row_id: str, source: str, repo_family: str, lane: str, reasoning_effort: str) -> dict:
    return {
        "id": row_id,
        "source": f"{CONVERTER_VERSION}:{source}",
        "lane": lane,
        "repo_family": repo_family,
        "shape": "public",
        "reasoning_effort": reasoning_effort,
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "tool_schema_json": TOOL_SCHEMA_JSON,
        "tools": TOOLS,
    }


def bucketed_family(prefix: str, row_id: str) -> str:
    bucket = int(hashlib.sha256(str(row_id).encode()).hexdigest(), 16) % FAMILY_BUCKETS
    return f"{prefix}/{bucket:02d}"


def trim_tool_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return trim_output(text, limit)


def _tool_content(raw: str | None) -> str:
    """mini-swe-agent observations are JSON with returncode and output."""
    text = raw or ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return trim_tool_output(text)
    if isinstance(payload, dict) and "output" in payload:
        return format_command_observation(
            payload.get("returncode"), str(payload.get("output") or ""), MAX_TOOL_OUTPUT_CHARS
        )
    return trim_tool_output(text)


def convert_open_swe_messages(messages: list[dict]) -> list[dict] | None:
    """Map a mini-swe-agent conversation onto the native roles and tools.

    Returns ``None`` when a turn cannot be expressed: a tool other than
    ``bash``, arguments that are not a JSON object with a command, a tool
    response that answers no call, or a call left unanswered other than
    the final submit. The submit turn itself is dropped: nothing answers
    it, and it is the harness's protocol, not the repair. What remains is
    a faithful prefix of the episode, ending on the last observation.
    """
    converted: list[dict] = []
    pending = 0
    for message in messages:
        role = message.get("role")
        if role == "system":
            converted.append({"role": "developer", "content": message.get("content") or ""})
        elif role == "user":
            converted.append({"role": "user", "content": message.get("content") or ""})
        elif role == "assistant":
            calls = []
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                if function.get("name") != "bash":
                    return None
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        return None
                if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
                    return None
                calls.append({
                    "type": "function",
                    "function": {"name": "shell", "arguments": {"command": arguments["command"]}},
                })
            entry = {"role": "assistant", "content": message.get("content") or ""}
            if message.get("reasoning_content"):
                entry["reasoning_content"] = message["reasoning_content"]
            if calls:
                entry["tool_calls"] = calls
            converted.append(entry)
            pending = len(calls)
        elif role == "tool":
            if pending <= 0:
                return None
            pending -= 1
            converted.append({"role": "tool", "name": "shell", "content": _tool_content(message.get("content"))})
        else:
            return None
    if pending:
        last = converted[-1]
        calls = last.get("tool_calls") or []
        submit = bool(calls) and pending == len(calls) and all(
            SUBMIT_MARKER in call["function"]["arguments"]["command"] for call in calls
        )
        if not submit:
            return None
        converted.pop()
    return converted


def _message_tokens(message: dict, count: TokenCounter) -> int:
    total = count(message.get("content") or "") + count(message.get("reasoning_content") or "")
    for call in message.get("tool_calls") or []:
        total += count(json.dumps(call["function"]["arguments"]))
    return total + 8  # role and framing tokens


def _turn_groups(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """Split into the prefix (developer and user turns) and assistant turns,
    each grouped with the tool responses that answer it."""
    prefix: list[dict] = []
    groups: list[list[dict]] = []
    for message in messages:
        if message["role"] in ("developer", "system", "user") and not groups:
            prefix.append(message)
        elif message["role"] == "assistant":
            groups.append([message])
        else:
            if not groups:
                return prefix, []
            groups[-1].append(message)
    return prefix, groups


# An elided run of turns is announced in place of the turns themselves,
# with the commands they ran, so a later turn's claim about the repository
# is backed by something in its own context. The note is a user turn, which
# assistant-only masking excludes from the loss: it is context, never a
# target.
ELIDED_COMMAND_CHARS = 160
# The note competes with the turns it is introducing, and it grows as the
# episode does. Capped at this share of the budget it cannot crowd them out;
# the oldest commands are dropped first, since the newest describe the state
# the next turn is about to act on.
ELIDED_NOTE_BUDGET_SHARE = 0.15


def _elision_note(elided: list[list[dict]], commands: list[str], dropped: int) -> dict:
    listing = [f"  {command}" for command in commands]
    if dropped:
        listing.insert(0, f"  ... {dropped} earlier commands omitted ...")
    body = "\n".join(listing) or "  (no commands)"
    return {
        "role": "user",
        "content": (
            f"[Earlier turns of this session are omitted. {len(elided)} assistant "
            f"turns ran before this point and the repository already reflects "
            f"them. The commands they ran, in order:\n{body}\n]"
        ),
    }


def _elision_message(
    elided: list[list[dict]], max_tokens: int, count: TokenCounter = approximate_tokens
) -> dict:
    """Announce a run of omitted turns and the commands they ran, within
    ``max_tokens``. Oldest commands go first when it does not fit."""
    commands = []
    for group in elided:
        for call in group[0].get("tool_calls") or []:
            function = call.get("function") or {}
            argument = (function.get("arguments") or {}).get("command")
            if isinstance(argument, str):
                commands.append(f"{function.get('name')}: {trim_output(argument, ELIDED_COMMAND_CHARS)}")
    kept = list(commands)
    while True:
        note = _elision_note(elided, kept, len(commands) - len(kept))
        if not kept or _message_tokens(note, count) <= max_tokens:
            return note
        kept = kept[1:]


def window_trajectory(
    messages: list[dict], budget_tokens: int, count: TokenCounter = approximate_tokens
) -> list[tuple[str, list[dict]]]:
    """Cut a trajectory into windows that each fit ``budget_tokens``.

    A trajectory that fits is one ``whole`` window. A longer one is cut into
    a ``head`` window and then ``segment`` windows which together cover every
    turn exactly once, so the middle and the end of an episode are trained on
    rather than only its opening. Head windows alone taught the model the
    exploratory turns, where an agent reasons least, and a gate run measured
    what that cost: reasoning fell by two thirds and success with it.

    A segment carries an elision note in place of the turns before it, naming
    how many ran and which commands they ran. That is what keeps a window
    that starts mid-episode honest: a later turn reporting a passing test has
    the patch that made it pass somewhere in its own context, rather than
    vouching for edits the transcript never shows.

    Turns are never split from the tool responses that answer them. Nothing
    is returned when even one turn does not fit beside the prefix.
    """
    prefix, groups = _turn_groups(messages)
    if not groups:
        return []
    prefix_tokens = sum(_message_tokens(m, count) for m in prefix)
    group_tokens = [sum(_message_tokens(m, count) for m in group) for group in groups]
    if prefix_tokens + sum(group_tokens) <= budget_tokens:
        return [("whole", prefix + [m for group in groups for m in group])]

    windows: list[tuple[str, list[dict]]] = []
    start = 0
    while start < len(groups):
        available = budget_tokens - prefix_tokens
        elision = None
        if start:
            elision = _elision_message(
                groups[:start], int(budget_tokens * ELIDED_NOTE_BUDGET_SHARE), count
            )
            available -= _message_tokens(elision, count)
        taken, used = 0, 0
        for size in group_tokens[start:]:
            if used + size > available:
                break
            used += size
            taken += 1
        # The opening turn not fitting beside the prefix drops the
        # trajectory, as before. Later, it means the elision note has grown
        # past what the budget leaves; the turns covered so far still stand.
        if taken == 0:
            break
        head = prefix if elision is None else prefix + [elision]
        body = [message for group in groups[start:start + taken] for message in group]
        windows.append(("head" if start == 0 else "segment", head + body))
        start += taken
    return windows


def convert_open_swe_row(row: dict, budget_tokens: int, count: TokenCounter = approximate_tokens) -> list[dict]:
    """Native rows from one Open-SWE-Traces row: none unless it resolved."""
    if row.get("resolved") != 1:
        return []
    messages = convert_open_swe_messages(row.get("messages") or [])
    if not messages:
        return []
    repo = str(row.get("repo") or "unknown/unknown")
    effort = SOURCE_LOADERS[SOURCE_OPEN_SWE]["reasoning_effort"]
    rows = []
    trajectory = row.get("trajectory_id") or row.get("instance_id")
    for position, (kind, window) in enumerate(window_trajectory(messages, budget_tokens, count)):
        # One trajectory now yields several rows, so the position goes in the
        # id: "head" is always position 0 and the segments follow it.
        suffix = kind if kind in ("whole", "head") else f"{kind}-{position}"
        base = _base_row(
            f"open-swe-traces/{trajectory}/{suffix}", SOURCE_OPEN_SWE, repo, "agentic", effort,
        )
        base["messages"] = window
        # The flag records the episode's outcome; ``window`` says whether the
        # row is that whole episode, its opening, or a later run of its turns.
        base["verification"] = {"all_required_tests_pass": True, "runner": "swe-rebench hidden tests", "window": kind}
        rows.append(base)
    return rows


def convert_open_code_instruct_row(row: dict) -> list[dict]:
    """A non-agentic row when every unit test passed."""
    try:
        score = float(row.get("average_test_score"))
    except (TypeError, ValueError):
        return []
    if score < 1.0:
        return []
    prompt, answer = row.get("input") or "", row.get("output") or ""
    if not prompt.strip() or not answer.strip():
        return []
    base = _base_row(
        f"opencodeinstruct/{row.get('id')}", SOURCE_OPEN_CODE_INSTRUCT,
        bucketed_family(f"opencodeinstruct:{row.get('domain') or 'generic'}", row.get("id")),
        "non_agentic", SOURCE_LOADERS[SOURCE_OPEN_CODE_INSTRUCT]["reasoning_effort"],
    )
    base["messages"] = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    base["verification"] = {"all_required_tests_pass": True, "runner": "opencodeinstruct unit tests"}
    return [base]


_THINK = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)


def split_think(output: str) -> tuple[str, str]:
    match = _THINK.match(output or "")
    if not match:
        return "", (output or "").strip()
    return match.group(1).strip(), match.group(2).strip()


def convert_open_code_reasoning_row(row: dict) -> list[dict]:
    """A non-agentic row with the reasoning moved out of the answer."""
    prompt = row.get("input") or ""
    if not prompt.strip() or prompt.strip() == "-":
        return []
    reasoning, answer = split_think(row.get("output") or "")
    if not answer:
        return []
    base = _base_row(
        f"opencodereasoning/{row.get('id')}", SOURCE_OPEN_CODE_REASONING,
        bucketed_family(f"opencodereasoning:{row.get('source') or 'unknown'}", row.get("id")),
        "non_agentic", SOURCE_LOADERS[SOURCE_OPEN_CODE_REASONING]["reasoning_effort"],
    )
    assistant = {"role": "assistant", "content": answer}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    base["messages"] = [{"role": "user", "content": prompt}, assistant]
    # Nothing executed these answers. The row says so, and notebook 02
    # admits an unverified answer to the non-agentic lane only.
    base["verification"] = {"all_required_tests_pass": None, "runner": "none"}
    return [base]


def convert_row(source: str, row: dict, budget_tokens: int, count: TokenCounter = approximate_tokens) -> list[dict]:
    if source == SOURCE_OPEN_SWE:
        return convert_open_swe_row(row, budget_tokens, count)
    if source == SOURCE_OPEN_CODE_INSTRUCT:
        return convert_open_code_instruct_row(row)
    if source == SOURCE_OPEN_CODE_REASONING:
        return convert_open_code_reasoning_row(row)
    raise ValueError(f"No converter for {source!r}; known: {sorted(SOURCE_LOADERS)}")


# Source rows read per native row wanted before a stream is given up on,
# so a source that yields nothing cannot be streamed to its end.
SCAN_ROWS_PER_NATIVE_ROW = 25


def convert_rows(
    source: str,
    rows: Iterable[dict],
    limit: int,
    budget_tokens: int,
    count: TokenCounter = approximate_tokens,
    max_scanned: int | None = None,
) -> Iterator[dict]:
    """Convert streamed rows until ``limit`` native rows have been yielded
    or ``max_scanned`` source rows have been read (default: 25 per native
    row wanted)."""
    bound = limit * SCAN_ROWS_PER_NATIVE_ROW if max_scanned is None else max_scanned
    produced = 0
    scanned = 0
    for row in rows:
        scanned += 1
        for converted in convert_row(source, row, budget_tokens, count):
            if converted_tokens(converted, count) > budget_tokens:
                continue
            yield converted
            produced += 1
            if produced >= limit:
                return
        if scanned >= bound:
            return


def converted_tokens(row: dict, count: TokenCounter = approximate_tokens) -> int:
    return sum(_message_tokens(m, count) for m in row["messages"])


def stream_source(source: str, token: str | None = None):
    """The ``datasets`` streaming iterator for a known source."""
    from datasets import load_dataset

    spec = SOURCE_LOADERS[source]
    return load_dataset(
        source, spec["name"], split=spec["split"], revision=spec["revision"], streaming=True, token=token
    )


def collect_public_rows(
    caps: dict[str, int],
    budget_tokens: int,
    count: TokenCounter = approximate_tokens,
    token: str | None = None,
) -> tuple[list[dict], dict]:
    """Stream each source in ``caps`` and convert up to its cap of rows.

    Returns the rows and a report of what each source yielded.
    """
    rows: list[dict] = []
    report: dict = {"converter": CONVERTER_VERSION, "budget_tokens": budget_tokens, "sources": {}}
    for source, cap in caps.items():
        if cap <= 0:
            continue
        before = len(rows)
        scanned = 0

        def counted(stream=stream_source(source, token)):
            nonlocal scanned
            for row in stream:
                scanned += 1
                yield row

        rows.extend(convert_rows(source, counted(), cap, budget_tokens, count))
        kept = rows[before:]
        report["sources"][source] = {
            "revision": SOURCE_LOADERS[source]["revision"],
            "scanned": scanned,
            "rows": len(kept),
            "unverified": sum(r["verification"].get("all_required_tests_pass") is not True for r in kept),
            "families": len({r["repo_family"] for r in kept}),
            "windows": dict(sorted(Counter(r["verification"].get("window", "n/a") for r in kept).items())),
        }
    return rows, report
