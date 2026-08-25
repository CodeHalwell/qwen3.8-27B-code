#!/usr/bin/env python3
"""Render every bootstrap corpus row through the real Qwen3.8 chat template.

Runs entirely on CPU: only the tokenizer is downloaded, under the reviewed
transformers pin. For each SFT trajectory it asserts the rendered text keeps
the exact deployment tool schema, the ChatML markers that assistant-only
masking keys on, and the thinking channel; for each preference pair it
replays notebook 04's prompt/completion prefix-separation check. Real token
statistics are merged back into the corpus quality reports.

Run with:
    uv run --group dev --with "transformers==5.3.0" --with jinja2 \
        python scripts/validate_dataset_rendering.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_8_27b_code.schema import TOOLS, TOOL_SCHEMA_JSON, canonical_to_qwen

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "unsloth/Qwen3.8-27B"


def rendered_tool_schema(rendered_prompt: str) -> str:
    start_tag, end_tag = "<tools>", "</tools>"
    if start_tag not in rendered_prompt or end_tag not in rendered_prompt:
        raise ValueError("Rendered prompt does not contain a <tools> block.")
    payload = rendered_prompt.split(start_tag, 1)[1].split(end_tag, 1)[0]
    rendered_tools = [json.loads(line) for line in payload.splitlines() if line.strip()]
    return json.dumps(rendered_tools, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def percentile(ordered: list[int], fraction: float) -> int:
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def stats(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    return {
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_sft(tokenizer, corpus_path: Path, max_length: int) -> dict:
    rows = load_jsonl(corpus_path)
    lengths = []
    for row in rows:
        text = tokenizer.apply_chat_template(
            canonical_to_qwen(row["messages"]),
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
            reasoning_effort=row["reasoning_effort"],
            preserve_thinking=True,
        )
        identity = row["id"]
        assert text.startswith("<|im_start|>system"), identity
        assert rendered_tool_schema(text) == TOOL_SCHEMA_JSON, f"{identity}: tool schema drift"
        assert "<|im_start|>user\n" in text and "<|im_start|>assistant\n" in text, (
            f"{identity}: response-masking markers missing"
        )
        assert "<think>" in text, f"{identity}: thinking channel missing"
        has_tool_call = any(message.get("tool_calls") for message in row["messages"])
        if has_tool_call:
            assert "<tool_call>" in text and "<tool_response>" in text, f"{identity}: tool syntax missing"
        token_count = len(tokenizer(text=text, add_special_tokens=False)["input_ids"])
        assert token_count <= max_length, f"{identity}: {token_count} tokens exceeds {max_length}"
        lengths.append(token_count)
    return {"rows": len(rows), "tokenizer": MODEL_ID, "tokens": stats(lengths), "max_length_checked": max_length}


def validate_preferences(tokenizer, corpus_path: Path) -> dict:
    pairs = load_jsonl(corpus_path)
    completion_lengths = []
    for pair in pairs:
        prompt = canonical_to_qwen(pair["prompt_messages"])
        prompt_text = tokenizer.apply_chat_template(
            prompt,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
        )
        for key in ("chosen_message", "rejected_message"):
            full = tokenizer.apply_chat_template(
                prompt + [pair[key]],
                tools=TOOLS,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
                reasoning_effort="medium",
                preserve_thinking=True,
            )
            assert full.startswith(prompt_text), f"{pair['id']}: template prefix drift on {key}"
            completion = full[len(prompt_text):]
            completion_lengths.append(len(tokenizer(text=completion, add_special_tokens=False)["input_ids"]))
    return {"rows": len(pairs), "tokenizer": MODEL_ID, "completion_tokens": stats(completion_lengths)}


def merge_report(report_path: Path, rendered: dict) -> None:
    report = json.loads(report_path.read_text())
    report["rendered"] = rendered
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=4096, help="Notebook 03 smoke sequence length.")
    arguments = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    sft = validate_sft(tokenizer, ROOT / "data" / "native_sft" / "trajectories.jsonl", arguments.max_length)
    merge_report(ROOT / "data" / "native_sft" / "quality_report.json", sft)
    print(json.dumps({"native_sft": sft}, indent=2))

    preferences = validate_preferences(tokenizer, ROOT / "data" / "preferences" / "pairs.jsonl")
    merge_report(ROOT / "data" / "preferences" / "quality_report.json", preferences)
    print(json.dumps({"preferences": preferences}, indent=2))
    print("All rows rendered through the real chat template with the deployment tool schema intact.")


if __name__ == "__main__":
    main()
