"""Parser for Qwen3.8's native XML tool-call syntax.

Importable twin of the parser cell in notebook 01; parity is enforced by
``tests/test_bootstrap_corpus.py`` against the generated cell source.
"""

from __future__ import annotations

import re

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
# Capture parameter values verbatim up to the closing tag; trimming happens
# per-parameter in code so a patch's trailing-whitespace diff lines survive.
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>(?:\r?\n)?(.*?)</parameter>",
    re.DOTALL,
)


# Generation is decoded with special tokens visible so the tool-call markup
# survives; the turn terminators must not leak into a stored final answer.
TURN_TERMINATORS = ("<|im_end|>", "<|endoftext|>")


def strip_turn_terminators(text: str) -> str:
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        for terminator in TURN_TERMINATORS:
            if stripped.endswith(terminator):
                stripped = stripped[: -len(terminator)].rstrip()
                changed = True
    return stripped


def split_reasoning(text: str) -> tuple[str, str]:
    if "</think>" in text:
        reasoning, content = text.split("</think>", 1)
        return (
            reasoning.removeprefix("<think>").strip(),
            strip_turn_terminators(content),
        )
    return "", strip_turn_terminators(text)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    reasoning, content = split_reasoning(text)
    calls = []
    for function_name, body in TOOL_CALL_RE.findall(content):
        arguments = {
            name.strip(): (
                value.rstrip("\r\n")
                if name.strip() == "patch"
                else value.strip()
            )
            for name, value in PARAM_RE.findall(body)
        }
        calls.append({
            "type": "function",
            "function": {"name": function_name.strip(), "arguments": arguments},
        })
    return reasoning, calls
