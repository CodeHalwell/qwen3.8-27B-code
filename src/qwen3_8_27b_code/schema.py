"""Deployment tool schema and chat-normalisation helpers.

This module is the importable twin of the generated notebooks' TOOLS cell.
The notebooks stay self-contained for Colab, so the definitions exist twice;
``tests/test_bootstrap_corpus.py`` asserts the canonical fingerprints match so
the copies cannot drift apart silently.
"""

from __future__ import annotations

import json

TOOL_SCHEMA_VERSION = "qwen38-six-tools-v2"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files below a repository-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 repository file with bounded output.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search repository text using a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff to files inside the repository.",
            "parameters": {
                "type": "object",
                "properties": {"patch": {"type": "string"}},
                "required": ["patch"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run an allow-listed repository test profile.",
            "parameters": {
                "type": "object",
                "properties": {"profile": {"type": "string", "enum": ["unit"]}},
                "required": ["profile"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command from the harness allow-list.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def without_arrow_nulls(value):
    """Remove null struct fields inserted by a Datasets/Arrow round trip."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = without_arrow_nulls(item)
            if normalized is not None:
                cleaned[key] = normalized
        return cleaned
    if isinstance(value, list):
        return [without_arrow_nulls(item) for item in value]
    return value


def canonical_tool_schema(tools: list[dict]) -> str:
    """Return a stable semantic fingerprint while retaining tool order."""
    return json.dumps(
        without_arrow_nulls(tools),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


TOOL_SCHEMA_JSON = canonical_tool_schema(TOOLS)


def canonical_to_qwen(messages: list[dict]) -> list[dict]:
    """Merge the leading policy messages into one system message.

    The Qwen3.8 template accepts ``developer`` natively and merges any run of
    leading system/developer messages itself. This fold is therefore not a
    compatibility shim; it exists so that training and deployment both send
    the template a single, deterministically joined policy message instead of
    relying on the template's own join order.
    """
    converted = []
    pending_system = []
    for stored_message in messages:
        message = without_arrow_nulls(stored_message)
        role = message["role"]
        if role in {"system", "developer"} and not converted:
            pending_system.append(str(message.get("content", "")))
            continue
        if pending_system:
            converted.append({"role": "system", "content": "\n\n".join(pending_system)})
            pending_system = []
        converted.append(message)
    if pending_system:
        converted.append({"role": "system", "content": "\n\n".join(pending_system)})
    return converted


def validate_tool_call(name: str, arguments: dict) -> str | None:
    """Check one call against the deployment schema; return an error or None.

    The harness needs this before executing anything so that a malformed call
    becomes a typed observation the model can recover from, rather than a
    Python exception or a silently wrong execution.
    """
    specification = next(
        (tool["function"] for tool in TOOLS if tool["function"]["name"] == name),
        None,
    )
    if specification is None:
        known = ", ".join(sorted(tool["function"]["name"] for tool in TOOLS))
        return f"unknown tool {name!r}; available tools are {known}"
    if not isinstance(arguments, dict):
        return f"{name} arguments must be an object"
    parameters = specification["parameters"]
    required = set(parameters.get("required", []))
    properties = parameters.get("properties", {})

    # Report every problem at once. A model that mistyped one argument name
    # usually needs to see the whole correction, not one round per mistake.
    problems = []
    missing = sorted(required - set(arguments))
    if missing:
        problems.append(f"missing required argument(s): {', '.join(missing)}")
    if parameters.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            problems.append(f"unknown argument(s): {', '.join(unexpected)}")
    for argument in sorted(set(arguments) & set(properties)):
        allowed = properties[argument].get("enum")
        if allowed is not None and arguments[argument] not in allowed:
            problems.append(f"{argument} must be one of {', '.join(map(str, allowed))}")
    return f"{name}: " + "; ".join(problems) if problems else None
