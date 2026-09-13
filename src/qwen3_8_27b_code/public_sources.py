"""Convert public Hub datasets into native-schema training rows.

Three sources, two lanes (docs/data-strategy.md):

* ``nvidia/Open-SWE-Traces`` — repository-agent trajectories that use one
  ``bash`` tool, which maps onto this harness's ``shell`` tool without
  loss. Only resolved trajectories are kept. They are long (a median of
  tens of thousands of tokens), so each is cut into windows that fit a
  token budget: the head, which teaches exploration, and the tail, which
  teaches editing, testing and finishing. Tool outputs are trimmed.
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
import re

from .schema import TOOL_SCHEMA_JSON, TOOL_SCHEMA_VERSION, TOOLS

CONVERTER_VERSION = "public-sources-v1"
TokenCounter = Callable[[str], int]

# Roughly one token per 3.6 characters of mixed code and prose; used only
# when no tokenizer is supplied.
CHARS_PER_TOKEN = 3.6
# A tool observation longer than this keeps its head and tail.
MAX_TOOL_OUTPUT_CHARS = 3_000
TRUNCATION_MARKER = "\n... [output trimmed] ...\n"

SOURCE_OPEN_SWE = "nvidia/Open-SWE-Traces"
SOURCE_OPEN_CODE_INSTRUCT = "nvidia/OpenCodeInstruct"
SOURCE_OPEN_CODE_REASONING = "nvidia/OpenCodeReasoning"

# How each source is loaded: the ``datasets`` config and split to stream.
SOURCE_LOADERS = {
    SOURCE_OPEN_SWE: {"name": "v1.2", "split": "minisweagent", "reasoning_effort": "xhigh"},
    SOURCE_OPEN_CODE_INSTRUCT: {"name": None, "split": "train", "reasoning_effort": "low"},
    SOURCE_OPEN_CODE_REASONING: {"name": "split_0", "split": "split_0", "reasoning_effort": "xhigh"},
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


def trim_tool_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    keep = (limit - len(TRUNCATION_MARKER)) // 2
    return text[:keep] + TRUNCATION_MARKER + text[-keep:]


def _tool_content(raw: str | None) -> str:
    """mini-swe-agent observations are JSON with returncode and output."""
    text = raw or ""
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return trim_tool_output(text)
    if isinstance(payload, dict) and "output" in payload:
        output = str(payload.get("output") or "")
        code = payload.get("returncode")
        prefix = "" if code in (None, 0) else f"[exit code {code}]\n"
        return trim_tool_output(prefix + output)
    return trim_tool_output(text)


def convert_open_swe_messages(messages: list[dict]) -> list[dict] | None:
    """Map a mini-swe-agent conversation onto the native roles and tools.

    Returns ``None`` when a turn cannot be expressed: a tool other than
    ``bash``, arguments that are not a JSON object with a command, or a
    tool response that answers no call.
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
        return None
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


def window_trajectory(
    messages: list[dict], budget_tokens: int, count: TokenCounter = approximate_tokens
) -> list[tuple[str, list[dict]]]:
    """Cut a trajectory into windows that fit ``budget_tokens``.

    A trajectory that fits is one ``whole`` window. A longer one yields a
    ``head`` window (the prefix and the first turns that fit) and a
    ``tail`` window (the prefix and the last turns that fit), so both the
    exploration and the finishing behaviour are represented. Turns are
    never split from the tool responses that answer them. Nothing is
    returned when even one turn does not fit beside the prefix.
    """
    prefix, groups = _turn_groups(messages)
    if not groups:
        return []
    prefix_tokens = sum(_message_tokens(m, count) for m in prefix)
    group_tokens = [sum(_message_tokens(m, count) for m in group) for group in groups]
    if prefix_tokens + sum(group_tokens) <= budget_tokens:
        return [("whole", prefix + [m for group in groups for m in group])]
    available = budget_tokens - prefix_tokens

    head: list[list[dict]] = []
    used = 0
    for group, size in zip(groups, group_tokens):
        if used + size > available:
            break
        head.append(group)
        used += size
    tail: list[list[dict]] = []
    used = 0
    for group, size in zip(reversed(groups), reversed(group_tokens)):
        if used + size > available:
            break
        tail.insert(0, group)
        used += size

    windows: list[tuple[str, list[dict]]] = []
    if head:
        windows.append(("head", prefix + [m for group in head for m in group]))
    if tail and tail != head:
        windows.append(("tail", prefix + [m for group in tail for m in group]))
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
    for kind, window in window_trajectory(messages, budget_tokens, count):
        base = _base_row(
            f"open-swe-traces/{row.get('trajectory_id') or row.get('instance_id')}/{kind}",
            SOURCE_OPEN_SWE, repo, "agentic", effort,
        )
        base["messages"] = window
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
        f"opencodeinstruct:{row.get('domain') or 'generic'}", "non_agentic",
        SOURCE_LOADERS[SOURCE_OPEN_CODE_INSTRUCT]["reasoning_effort"],
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
        f"opencodereasoning:{row.get('source') or 'unknown'}", "non_agentic",
        SOURCE_LOADERS[SOURCE_OPEN_CODE_REASONING]["reasoning_effort"],
    )
    assistant = {"role": "assistant", "content": answer}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    base["messages"] = [{"role": "user", "content": prompt}, assistant]
    base["verification"] = {"all_required_tests_pass": True, "runner": "opencodereasoning (unverified answer)"}
    return [base]


def convert_row(source: str, row: dict, budget_tokens: int, count: TokenCounter = approximate_tokens) -> list[dict]:
    if source == SOURCE_OPEN_SWE:
        return convert_open_swe_row(row, budget_tokens, count)
    if source == SOURCE_OPEN_CODE_INSTRUCT:
        return convert_open_code_instruct_row(row)
    if source == SOURCE_OPEN_CODE_REASONING:
        return convert_open_code_reasoning_row(row)
    raise ValueError(f"No converter for {source!r}; known: {sorted(SOURCE_LOADERS)}")


def convert_rows(
    source: str,
    rows: Iterable[dict],
    limit: int,
    budget_tokens: int,
    count: TokenCounter = approximate_tokens,
) -> Iterator[dict]:
    """Convert streamed rows until ``limit`` native rows have been yielded."""
    produced = 0
    for row in rows:
        for converted in convert_row(source, row, budget_tokens, count):
            if converted_tokens(converted, count) > budget_tokens:
                continue
            yield converted
            produced += 1
            if produced >= limit:
                return


def converted_tokens(row: dict, count: TokenCounter = approximate_tokens) -> int:
    return sum(_message_tokens(m, count) for m in row["messages"])


def stream_source(source: str, token: str | None = None):
    """The ``datasets`` streaming iterator for a known source."""
    from datasets import load_dataset

    spec = SOURCE_LOADERS[source]
    return load_dataset(source, spec["name"], split=spec["split"], streaming=True, token=token)


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
        rows.extend(convert_rows(source, stream_source(source, token), cap, budget_tokens, count))
        kept = rows[before:]
        report["sources"][source] = {
            "rows": len(kept),
            "families": len({r["repo_family"] for r in kept}),
            "windows": dict(sorted(
                __import__("collections").Counter(r["verification"].get("window", "n/a") for r in kept).items()
            )),
        }
    return rows, report
