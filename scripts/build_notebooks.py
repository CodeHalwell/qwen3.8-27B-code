#!/usr/bin/env python3
"""Build the Colab notebook suite with nbformat.

Run with:
    uv run --with nbformat scripts/build_notebooks.py
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def notebook(title: str, cells: list):
    for index, cell in enumerate(cells):
        identity = f"{title}\0{index}\0{cell.cell_type}\0{cell.source}"
        cell["id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "accelerator": "GPU",
            "colab": {
                "name": title,
                "provenance": [],
                "toc_visible": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )


INSTALL_CORE = r"""
import subprocess
import sys
from pathlib import Path

# The Git pins supply current Unsloth/Qwen3.8 support. Transformers, TRL and
# Datasets deliberately use the mutually compatible versions from the adjacent
# official Unsloth Qwen3.5 27B notebook. Do not replace these with branch-head
# SHAs without resolving package metadata together first.
GIT_REVISIONS = {
    "unsloth": "c87fe20e32aca9ceb2dc5059c2987738f32446e8",
    "unsloth_zoo": "5b239e574f03ab3077c17e49aeef3cacfe7cdd4e",
}

import torch

torch_version = torch.__version__.split("+", 1)[0]
torch_minor = ".".join(torch_version.split(".")[:2])
torchao_by_torch = {"2.8": "0.16.0", "2.9": "0.16.0", "2.10": "0.16.0", "2.11": "0.18.0"}
xformers_by_torch = {"2.8": "0.0.32.post2", "2.9": "0.0.33.post1", "2.10": "0.0.34", "2.11": "0.0.34"}
if torch_minor not in torchao_by_torch:
    raise RuntimeError(
        f"No reviewed Colab dependency set for torch {torch.__version__}. "
        f"Expected one of {sorted(torchao_by_torch)}; update the compatibility matrix first."
    )

COMPATIBILITY_PINS = {
    "transformers": "5.3.0",
    "trl": "0.22.2",
    "datasets": "4.3.0",
    "peft": "0.19.0",
    "torchao": torchao_by_torch[torch_minor],
    "xformers": xformers_by_torch[torch_minor],
}
INSTALLER_REVISION = "colab-v2"
pin_key = "-".join(value.replace(".", "") for value in COMPATIBILITY_PINS.values())
git_key = "-".join(value[:8] for value in GIT_REVISIONS.values())
INSTALL_KEY = f"{INSTALLER_REVISION}-torch{torch_minor}-{git_key}-{pin_key}"
INSTALL_MARKER = Path(f"/content/.qwen38_env_{INSTALL_KEY}")
PIP_LOG = Path("/content/qwen38_pip_install.log")
FORCE_INSTALL = False

def install_phase(name: str, packages: list[str], *, no_deps: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--no-cache-dir",
        "--log",
        str(PIP_LOG),
    ]
    if no_deps:
        command.append("--no-deps")
    command.extend(packages)
    print(f"\n=== install phase: {name} ===")
    print("\n".join(f"  {package}" for package in packages))
    result = subprocess.run(command, check=False)
    if result.returncode:
        log_tail = (
            "\n".join(PIP_LOG.read_text(errors="replace").splitlines()[-120:])
            if PIP_LOG.exists()
            else "[pip did not create its log file]"
        )
        print(f"\n--- tail of {PIP_LOG} ---\n{log_tail}")
        raise RuntimeError(
            f"Package installation failed during {name!r} with exit code {result.returncode}. "
            f"The detailed log is at {PIP_LOG}."
        )

if FORCE_INSTALL or not INSTALL_MARKER.exists():
    if PIP_LOG.exists():
        PIP_LOG.unlink()
    install_phase("packaging tools", ["pip", "setuptools==80.9.0", "wheel>=0.42.0"])
    install_phase("Qwen3.8 training stack", [
        f"unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git@{GIT_REVISIONS['unsloth_zoo']}",
        f"unsloth @ git+https://github.com/unslothai/unsloth.git@{GIT_REVISIONS['unsloth']}",
        f"torch=={torch_version}",
        f"torchao=={COMPATIBILITY_PINS['torchao']}",
        f"transformers=={COMPATIBILITY_PINS['transformers']}",
        f"trl=={COMPATIBILITY_PINS['trl']}",
        f"datasets=={COMPATIBILITY_PINS['datasets']}",
        f"peft=={COMPATIBILITY_PINS['peft']}",
        "accelerate",
        "bitsandbytes",
        "trackio",
        "huggingface_hub>=0.34.0,<2.0",
        "hf_transfer",
        "sentencepiece>=0.2.0",
        "protobuf",
        "pytest",
        "jmespath",
    ])
    install_phase(
        "PyTorch-matched xFormers wheel",
        [f"xformers=={COMPATIBILITY_PINS['xformers']}"],
        no_deps=True,
    )
    INSTALL_MARKER.write_text(INSTALL_KEY)
    print("Packages installed. Restart the Colab runtime, then rerun this notebook from the top.")
else:
    print(f"Pinned environment already installed: {INSTALL_KEY}")
"""


AUTH_AND_RUNTIME = r"""
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from huggingface_hub import login, whoami

if "GIT_REVISIONS" not in globals():
    raise RuntimeError(
        "This runtime was restarted. Rerun the notebook from the first cell; "
        "the install marker will skip the expensive package installation."
    )
if "COMPATIBILITY_PINS" not in globals():
    raise RuntimeError("Missing compatibility pins; rerun the notebook from the first cell.")

try:
    from google.colab import userdata
except ImportError:
    userdata = None

if not torch.cuda.is_available():
    raise RuntimeError("Select a Colab G4 GPU runtime before continuing.")

gpu = torch.cuda.get_device_properties(0)
gpu_total_gib = gpu.total_memory / 1024**3
# A vendor-labelled 96 GB card can be reported as about 89.4 GiB because
# PyTorch converts the byte count with a binary divisor. Keep the floor well
# above the roughly 44.7 GiB reported for a 48 GB card without rejecting G4.
MIN_G4_TOTAL_GIB = 85.0
print(
    f"GPU: {gpu.name} ({gpu_total_gib:.1f} GiB total), "
    f"capability={torch.cuda.get_device_capability(0)}"
)
if gpu_total_gib < MIN_G4_TOTAL_GIB:
    raise RuntimeError(
        "This suite expects the nominal 96 GB Colab G4 runtime. "
        f"PyTorch reports {gpu_total_gib:.1f} GiB total; expected at least "
        f"{MIN_G4_TOTAL_GIB:.0f} GiB. A value near 45 GiB usually indicates "
        "the 48 GB GPU variant."
    )

hf_token = userdata.get("HF_TOKEN") if userdata is not None else os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Add a write-capable HF_TOKEN to Colab Secrets before continuing.")
login(token=hf_token, add_to_git_credential=False)
HF_USERNAME = whoami()["name"]

def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"

observed_pins = {name: package_version(name) for name in COMPATIBILITY_PINS}
pin_mismatches = {
    name: {"expected": expected, "observed": observed_pins[name]}
    for name, expected in COMPATIBILITY_PINS.items()
    if observed_pins[name] != expected
}
if pin_mismatches:
    raise RuntimeError(
        "The runtime does not match the reviewed compatibility set. "
        f"Rerun the install cell with FORCE_INSTALL=True: {pin_mismatches}"
    )

RUN_ROOT = Path("/content/qwen38_runs")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
runtime_manifest = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu.name,
    "gpu_total_gib": round(gpu_total_gib, 2),
    "packages": {
        name: package_version(name)
        for name in ["unsloth", "unsloth_zoo", "transformers", "trl", "peft", "datasets"]
    },
    "git_revisions": GIT_REVISIONS,
    "compatibility_pins": COMPATIBILITY_PINS,
}
(RUN_ROOT / "runtime_manifest.json").write_text(json.dumps(runtime_manifest, indent=2))
print(json.dumps(runtime_manifest, indent=2))
print(f"Authenticated as {HF_USERNAME}")
"""


TOOLS_CELL = r'''
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
            "description": "Run a restricted allow-listed command. It is disabled in the pilot.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]

def _without_arrow_nulls(value):
    """Remove null struct fields inserted by a Datasets/Arrow round trip."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = _without_arrow_nulls(item)
            if normalized is not None:
                cleaned[key] = normalized
        return cleaned
    if isinstance(value, list):
        return [_without_arrow_nulls(item) for item in value]
    return value

def canonical_tool_schema(tools: list[dict]) -> str:
    """Return a stable semantic fingerprint while retaining tool order."""
    return json.dumps(
        _without_arrow_nulls(tools),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

TOOL_SCHEMA_JSON = canonical_tool_schema(TOOLS)

def rendered_tool_schema(rendered_prompt: str) -> str:
    """Extract and canonicalise JSON tool declarations from a Qwen prompt."""
    start_tag = "<tools>"
    end_tag = "</tools>"
    if start_tag not in rendered_prompt or end_tag not in rendered_prompt:
        raise ValueError("Rendered prompt does not contain a <tools> block.")
    payload = rendered_prompt.split(start_tag, 1)[1].split(end_tag, 1)[0]
    try:
        rendered_tools = [
            json.loads(line)
            for line in payload.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise ValueError("Rendered <tools> block is not newline-delimited JSON.") from exc
    return canonical_tool_schema(rendered_tools)

def canonical_to_qwen(messages: list[dict]) -> list[dict]:
    """Fold an initial developer message into system for the HF tokenizer.

    The adapter performs the same mapping in training and deployment. The
    official safetensor tokenizer currently accepts system/user/assistant/tool.
    """
    converted = []
    pending_system = []
    for stored_message in messages:
        message = _without_arrow_nulls(stored_message)
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

def render_chat(messages: list[dict], *, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        canonical_to_qwen(messages),
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
        reasoning_effort="medium",
        preserve_thinking=True,
    )
'''


def build_00_preflight():
    return notebook(
        "00 - Colab G4 preflight",
        [
            markdown(
                """
                # 00 — Colab G4 preflight

                ## Goal

                Prove the G4 runtime, pinned day-zero package set, Hub
                authentication, BF16 model load and native Qwen3.8 tool template
                before any training. This notebook performs no optimisation.

                **Gate:** do not continue if the GPU, model load, template or
                durable manifest check fails.
                """
            ),
            markdown(
                """
                ## Setup

                Select the **G4** runtime and add a write-capable `HF_TOKEN` in
                Colab Secrets. Run the install cell once, restart the runtime,
                then rerun the notebook from the top. The marker makes the
                install cell cheap and idempotent on the second pass.
                """
            ),
            code(INSTALL_CORE),
            code(AUTH_AND_RUNTIME),
            markdown("## Load the trainable BF16 checkpoint"),
            code(
                r"""
                from unsloth import FastLanguageModel

                MODEL_ID = "unsloth/Qwen3.8-27B"
                MODEL_REVISION = None  # Set to an immutable Hub commit after the first successful load.
                MAX_SEQUENCE_LENGTH = 4096

                load_kwargs = {
                    "model_name": MODEL_ID,
                    "max_seq_length": MAX_SEQUENCE_LENGTH,
                    "load_in_4bit": False,
                    "load_in_8bit": False,
                    "full_finetuning": False,
                }
                if MODEL_REVISION:
                    load_kwargs["revision"] = MODEL_REVISION

                torch.cuda.reset_peak_memory_stats()
                model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
                peak_gib = torch.cuda.max_memory_reserved() / 1024**3
                print(f"Loaded {MODEL_ID}; peak reserved VRAM={peak_gib:.2f} GiB")

                model_type = getattr(model.config, "model_type", None)
                text_config = getattr(model.config, "text_config", model.config)
                assert model_type == "qwen3_5", model_type
                assert getattr(text_config, "max_position_embeddings", None) == 262144
                """
            ),
            markdown("## Validate the native tool template"),
            code(TOOLS_CELL),
            code(
                r"""
                template_probe = [
                    {"role": "developer", "content": "Work only in the provided repository and verify changes."},
                    {"role": "user", "content": "Read src/cache.py before proposing a fix."},
                ]
                rendered_probe = render_chat(template_probe, add_generation_prompt=True)
                assert rendered_tool_schema(rendered_probe) == TOOL_SCHEMA_JSON, (
                    "The tokenizer changed the deployment tool declarations."
                )
                assert "Work only in the provided repository" in rendered_probe, (
                    "The developer-to-system adapter lost the repository policy."
                )
                assert rendered_probe.endswith("<think>\n"), (
                    "The generation prompt no longer opens Qwen's thinking channel."
                )
                print(rendered_probe[:4000])
                """
            ),
            markdown("## Run one bounded inference probe"),
            code(
                r"""
                FastLanguageModel.for_inference(model)
                inputs = tokenizer(
                    text=rendered_probe,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to("cuda")
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=20,
                        do_sample=True,
                        use_cache=True,
                    )
                new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=False)
                print(generated)
                print(f"Peak reserved VRAM: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")
                """
            ),
            markdown("## Inspect language and vision parameters"),
            code(
                r"""
                vision_markers = ("vision", "visual", "image")
                vision_names = [
                    name for name, _ in model.named_parameters()
                    if any(marker in name.lower() for marker in vision_markers)
                ]
                linear_suffixes = sorted({
                    name.rsplit(".", 1)[-1]
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and not any(marker in name.lower() for marker in vision_markers)
                })
                print(f"Vision-associated parameters: {len(vision_names)}")
                print("Language linear suffixes:", linear_suffixes)
                assert vision_names, "Expected the multimodal checkpoint to expose vision parameters."
                """
            ),
            markdown(
                """
                ## Checks and next step

                The preflight passes when the model loads in BF16, the rendered
                prompt contains the exact XML tool syntax, a bounded generation
                completes, and `runtime_manifest.json` exists. Record the Hub
                commit used, then continue to `01_tool_calling_baseline.ipynb`.
                """
            ),
        ],
    )


def build_01_baseline():
    return notebook(
        "01 - Native tool-calling baseline",
        [
            markdown(
                """
                # 01 — Native tool-calling baseline

                ## Goal

                Run the upstream BF16 model through the exact six-tool schema,
                preserve typed failures and measure episode cost before SFT.

                This notebook implements the documented `trusted-dev` fallback
                for reviewed pilot repositories. It is **not** a security
                boundary for arbitrary public code. Replace the executor with a
                Harbor isolated backend before scaling collection or RL.
                """
            ),
            code(INSTALL_CORE),
            code(AUTH_AND_RUNTIME),
            markdown("## Parameters"),
            code(
                r"""
                from dataclasses import dataclass, replace
                from datetime import datetime, timezone
                import hashlib
                import re
                import shutil
                import subprocess
                import tempfile
                import time
                import uuid

                from unsloth import FastLanguageModel

                MODEL_ID = "unsloth/Qwen3.8-27B"
                MAX_SEQUENCE_LENGTH = 16384
                MAX_NEW_TOKENS_PER_TURN = 1024
                MAX_TOOL_CALLS = 10
                EPISODE_TIMEOUT_SECONDS = 480
                BASELINE_SEEDS = (3407, 9176, 20261)
                DEMO_MODE = True
                PILOT_MANIFEST = Path("/content/pilot_tasks.jsonl")
                RESULTS_DIR = RUN_ROOT / "baseline"
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                secret_markers = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
                TASK_ENV = {
                    key: value for key, value in os.environ.items()
                    if not any(marker in key.upper() for marker in secret_markers)
                }

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_ID,
                    max_seq_length=MAX_SEQUENCE_LENGTH,
                    load_in_4bit=False,
                    full_finetuning=False,
                )
                FastLanguageModel.for_inference(model)
                """
            ),
            code(TOOLS_CELL),
            markdown("## Parse Qwen3.8's native XML tool calls"),
            code(
                r"""
                TOOL_CALL_RE = re.compile(
                    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
                    re.DOTALL,
                )
                PARAM_RE = re.compile(
                    r"<parameter=([^>\n]+)>(?:\r?\n)?(.*?)\s*</parameter>",
                    re.DOTALL,
                )

                def split_reasoning(text: str) -> tuple[str, str]:
                    if "</think>" in text:
                        reasoning, content = text.split("</think>", 1)
                        return reasoning.removeprefix("<think>").strip(), content.strip()
                    return "", text.strip()

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

                parser_probe = ("</think>\n\n<tool_call>\n<function=read_file>\n"
                                "<parameter=path>\nsrc/cache.py\n</parameter>\n"
                                "</function>\n</tool_call>")
                assert parse_tool_calls(parser_probe)[1][0]["function"]["arguments"]["path"] == "src/cache.py"
                """
            ),
            markdown("## Trusted pilot task and executor"),
            code(
                r"""
                @dataclass(frozen=True)
                class PilotTask:
                    task_id: str
                    repo_path: str
                    request: str
                    visible_test_command: list[str]
                    hidden_test_command: list[str]

                def make_demo_task() -> PilotTask:
                    repo = Path("/content/qwen38_demo_repo")
                    if repo.exists():
                        shutil.rmtree(repo)
                    (repo / "src").mkdir(parents=True)
                    (repo / "tests").mkdir()
                    (repo / "src" / "clamp.py").write_text(
                        "def clamp(value, lower, upper):\n"
                        "    return min(lower, max(upper, value))\n"
                    )
                    (repo / "tests" / "test_clamp.py").write_text(
                        "from src.clamp import clamp\n\n"
                        "def test_value_in_range():\n"
                        "    assert clamp(5, 0, 10) == 5\n"
                    )
                    hidden = Path("/content/qwen38_hidden_test.py")
                    hidden.write_text(
                        "from pathlib import Path\n"
                        "ns = {}\n"
                        "exec((Path.cwd() / 'src' / 'clamp.py').read_text(), ns)\n"
                        "clamp = ns['clamp']\n"
                        "assert clamp(-2, 0, 10) == 0\n"
                        "assert clamp(20, 0, 10) == 10\n"
                        "assert clamp(5, 0, 10) == 5\n"
                    )
                    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
                    subprocess.run(["git", "add", "."], cwd=repo, check=True)
                    subprocess.run(
                        ["git", "-c", "user.name=pilot", "-c", "user.email=pilot@example.invalid", "commit", "-qm", "fixture"],
                        cwd=repo,
                        check=True,
                    )
                    return PilotTask(
                        task_id="demo/clamp-001",
                        repo_path=str(repo),
                        request="Fix clamp so values inside the range are unchanged and out-of-range values use the nearest bound. Run the unit tests.",
                        visible_test_command=[sys.executable, "-m", "pytest", "-q"],
                        hidden_test_command=[sys.executable, str(hidden)],
                    )

                def load_tasks() -> list[PilotTask]:
                    if DEMO_MODE:
                        return [make_demo_task()]
                    if not PILOT_MANIFEST.exists():
                        raise FileNotFoundError(PILOT_MANIFEST)
                    return [PilotTask(**json.loads(line)) for line in PILOT_MANIFEST.read_text().splitlines() if line.strip()]

                def rooted(root: Path, relative: str) -> Path:
                    candidate = (root / relative).resolve()
                    if candidate != root and root not in candidate.parents:
                        raise ValueError("path escapes repository root")
                    return candidate

                def execute_tool(task: PilotTask, name: str, arguments: dict) -> str:
                    root = Path(task.repo_path).resolve()
                    if name == "list_files":
                        base = rooted(root, arguments["path"])
                        files = [str(path.relative_to(root)) for path in base.rglob("*") if path.is_file() and ".git" not in path.parts]
                        return "\n".join(files[:200]) or "[no files]"
                    if name == "read_file":
                        return rooted(root, arguments["path"]).read_text(errors="replace")[:20000]
                    if name == "search":
                        try:
                            regex = re.compile(arguments["query"])
                        except re.error as exc:
                            return f"invalid regular expression: {exc}"
                        hits = []
                        scanned_bytes = 0
                        for path in sorted(root.rglob("*")):
                            if not path.is_file() or ".git" in path.parts:
                                continue
                            try:
                                payload = path.read_bytes()
                            except OSError as exc:
                                hits.append(f"{path.relative_to(root)}:read_error:{exc}")
                                continue
                            if b"\x00" in payload:
                                continue
                            scanned_bytes += len(payload)
                            if scanned_bytes > 5_000_000:
                                hits.append("[search truncated after 5 MB]")
                                break
                            for line_no, line in enumerate(payload.decode("utf-8", errors="replace").splitlines(), 1):
                                if regex.search(line):
                                    hits.append(f"{path.relative_to(root)}:{line_no}:{line}")
                                    if len(hits) >= 200:
                                        hits.append("[search truncated after 200 matches]")
                                        return "\n".join(hits)[:20000]
                        return "\n".join(hits)[:20000] or "[no matches]"
                    if name == "apply_patch":
                        result = subprocess.run(
                            ["git", "apply", "--whitespace=nowarn", "-"],
                            cwd=root,
                            input=arguments["patch"],
                            text=True,
                            capture_output=True,
                            timeout=30,
                        )
                        return "patch applied" if result.returncode == 0 else f"patch rejected: {result.stderr[:4000]}"
                    if name == "run_tests":
                        if arguments["profile"] != "unit":
                            return "unknown test profile"
                        result = subprocess.run(
                            task.visible_test_command,
                            cwd=root,
                            env=TASK_ENV,
                            text=True,
                            capture_output=True,
                            timeout=120,
                        )
                        return f"exit={result.returncode}\n{(result.stdout + result.stderr)[-12000:]}"
                    if name == "shell":
                        return "shell is disabled in the pilot; use the semantic tools"
                    return f"unknown tool: {name}"
                """
            ),
            markdown("## Run a bounded episode"),
            code(
                r"""
                def generate_turn(messages: list[dict]) -> tuple[str, int, int]:
                    rendered = render_chat(messages, add_generation_prompt=True)
                    inputs = tokenizer(
                        text=rendered,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to("cuda")
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=MAX_NEW_TOKENS_PER_TURN,
                            temperature=1.0,
                            top_p=0.95,
                            top_k=20,
                            do_sample=True,
                            use_cache=True,
                        )
                    new_ids = outputs[0, inputs["input_ids"].shape[1]:]
                    return tokenizer.decode(new_ids, skip_special_tokens=False), inputs["input_ids"].numel(), new_ids.numel()

                def run_episode(task: PilotTask, seed: int) -> dict:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    start = time.monotonic()
                    messages = [
                        {"role": "developer", "content": "Work only in the provided repository. Inspect before editing and run tests before completing."},
                        {"role": "user", "content": task.request},
                    ]
                    prompt_tokens = completion_tokens = tool_count = 0
                    termination = "assistant_complete"
                    final_text = ""

                    for _ in range(MAX_TOOL_CALLS + 1):
                        if time.monotonic() - start > EPISODE_TIMEOUT_SECONDS:
                            termination = "timeout"
                            break
                        raw, prompt_count, completion_count = generate_turn(messages)
                        prompt_tokens += prompt_count
                        completion_tokens += completion_count
                        reasoning, calls = parse_tool_calls(raw)
                        if not calls:
                            _, final_text = split_reasoning(raw)
                            messages.append({"role": "assistant", "reasoning_content": reasoning, "content": final_text})
                            break
                        if tool_count + len(calls) > MAX_TOOL_CALLS:
                            termination = "tool_budget"
                            break
                        messages.append({"role": "assistant", "reasoning_content": reasoning, "content": "", "tool_calls": calls})
                        for call in calls:
                            function = call["function"]
                            try:
                                observation = execute_tool(task, function["name"], function["arguments"])
                            except Exception as exc:
                                observation = f"tool_error: {type(exc).__name__}: {exc}"
                            messages.append({"role": "tool", "name": function["name"], "content": observation})
                            tool_count += 1
                    else:
                        termination = "tool_budget"

                    hidden = subprocess.run(
                        task.hidden_test_command,
                        cwd=task.repo_path,
                        env=TASK_ENV,
                        text=True,
                        capture_output=True,
                        timeout=120,
                    )
                    elapsed = time.monotonic() - start
                    return {
                        "trajectory_id": str(uuid.uuid4()),
                        "task_id": task.task_id,
                        "seed": seed,
                        "model_id": MODEL_ID,
                        "adapter_version": "qwen-native-tools-v0.1",
                        "messages": messages,
                        "termination": termination,
                        "success": hidden.returncode == 0,
                        "hidden_output": (hidden.stdout + hidden.stderr)[-4000:],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "tool_calls": tool_count,
                            "wall_seconds": round(elapsed, 3),
                        },
                        "final_text": final_text,
                    }

                def run_isolated_episode(task: PilotTask, seed: int) -> dict:
                    attempt_root = Path(tempfile.mkdtemp(prefix="qwen38_baseline_"))
                    working_repo = attempt_root / "repo"
                    shutil.copytree(task.repo_path, working_repo)
                    attempt = replace(task, repo_path=str(working_repo))
                    try:
                        return run_episode(attempt, seed)
                    finally:
                        shutil.rmtree(attempt_root, ignore_errors=True)

                tasks = load_tasks()
                active_seeds = BASELINE_SEEDS[:1] if DEMO_MODE else BASELINE_SEEDS
                trajectories = [
                    run_isolated_episode(task, seed)
                    for task in tasks
                    for seed in active_seeds
                ]
                print(json.dumps([{k: v for k, v in row.items() if k != "messages"} for row in trajectories], indent=2))
                """
            ),
            markdown("## Persist results and estimate the next gate"),
            code(
                r"""
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                trace_path = RESULTS_DIR / f"trajectories-{timestamp}.jsonl"
                trace_path.write_text("\n".join(json.dumps(row) for row in trajectories) + "\n")

                durations = [row["usage"]["wall_seconds"] for row in trajectories]
                mean_seconds = sum(durations) / len(durations)
                projected_candidate_hours = 24 * 3 * mean_seconds / 3600
                summary = {
                    "unique_tasks": len(tasks),
                    "attempts": len(trajectories),
                    "seeds": list(active_seeds),
                    "successes": sum(row["success"] for row in trajectories),
                    "mean_episode_seconds": mean_seconds,
                    "candidate_gate_gpu_hours_at_observed_mean": projected_candidate_hours,
                    "trace_path": str(trace_path),
                }
                (RESULTS_DIR / f"summary-{timestamp}.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary, indent=2))

                # Upload after manual trace review. This guards against publishing raw reasoning accidentally.
                PUSH_PRIVATE_RESULTS = False
                if PUSH_PRIVATE_RESULTS:
                    from huggingface_hub import HfApi
                    HfApi().upload_folder(
                        repo_id=f"{HF_USERNAME}/qwen38-code-pilot-results",
                        repo_type="dataset",
                        folder_path=str(RESULTS_DIR),
                        private=True,
                    )
                """
            ),
            markdown(
                """
                ## Checks and next step

                Manually inspect every pilot trace. Infrastructure errors must
                be separated from model failures. Replace demo mode with the
                frozen 12-task manifest, then use the resulting failure mix to
                decide which native trajectories to collect for SFT.
                """
            ),
        ],
    )


def build_02_data():
    return notebook(
        "02 - Prepare native SFT data",
        [
            markdown(
                """
                # 02 — Prepare native-schema SFT data

                ## Goal

                Validate, render, measure and publish execution-verified Qwen
                tool trajectories without syntactically translating third-party
                harness traces. The output is a private, versioned dataset with
                `messages` and rendered `text` columns.
                """
            ),
            code(INSTALL_CORE),
            code(AUTH_AND_RUNTIME),
            markdown("## Load only the tokenizer and define the target schema"),
            code(
                r"""
                from collections import Counter
                from datasets import Dataset, concatenate_datasets, load_dataset
                import numpy as np
                from transformers import AutoTokenizer

                MODEL_ID = "unsloth/Qwen3.8-27B"
                SOURCE_DATASET_IDS = []  # Native-schema datasets only.
                OUTPUT_DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                DEMO_MODE = True
                AUDIT_PUBLIC_SCHEMAS = False
                PUBLIC_AUDIT_IDS = [
                    "nvidia/Nemotron-SFT-SWE-v3",
                    "nvidia/Open-SWE-Traces",
                    "nvidia/OpenCodeReasoning",
                    "nvidia/OpenCodeInstruct",
                ]

                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                """
            ),
            code(TOOLS_CELL),
            markdown("## Load native examples or the format-only fixture"),
            code(
                r"""
                demo_rows = [
                    {
                        "id": "fixture/native-tool-001",
                        "repo_family": "fixture-clamp",
                        "tool_schema_version": "qwen38-six-tools-v1",
                        "tool_schema_json": TOOL_SCHEMA_JSON,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "developer", "content": "Inspect, edit narrowly, and run tests."},
                            {"role": "user", "content": "Fix clamp and add regression coverage."},
                            {"role": "assistant", "reasoning_content": "I should inspect the implementation first.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "src/clamp.py"}}}]},
                            {"role": "tool", "name": "read_file", "content": "def clamp(value, lower, upper):\n    return min(lower, max(upper, value))\n"},
                            {"role": "assistant", "reasoning_content": "The min/max order is reversed.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "apply_patch", "arguments": {"patch": "--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n def clamp(value, lower, upper):\n-    return min(lower, max(upper, value))\n+    return max(lower, min(upper, value))\n"}}}]},
                            {"role": "tool", "name": "apply_patch", "content": "patch applied"},
                            {"role": "assistant", "reasoning_content": "I should verify the change.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run_tests", "arguments": {"profile": "unit"}}}]},
                            {"role": "tool", "name": "run_tests", "content": "exit=0\n3 passed"},
                            {"role": "assistant", "reasoning_content": "", "content": "Fixed the bound ordering and verified all tests pass."},
                        ],
                        "verification": {"all_required_tests_pass": True},
                    },
                    {
                        "id": "fixture/native-tool-002",
                        "repo_family": "fixture-parser",
                        "tool_schema_version": "qwen38-six-tools-v1",
                        "tool_schema_json": TOOL_SCHEMA_JSON,
                        "tools": TOOLS,
                        "messages": [
                            {"role": "developer", "content": "Inspect the failing path and verify the focused change."},
                            {"role": "user", "content": "Handle an empty CSV field as an empty list."},
                            {"role": "assistant", "reasoning_content": "I should inspect the parser branch first.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "src/parser.py"}}}]},
                            {"role": "tool", "name": "read_file", "content": "def parse_field(value):\n    return value.split(',')\n"},
                            {"role": "assistant", "reasoning_content": "The empty string needs an explicit branch.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "apply_patch", "arguments": {"patch": "--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1,2 +1,2 @@\n def parse_field(value):\n-    return value.split(',')\n+    return [] if value == '' else value.split(',')\n"}}}]},
                            {"role": "tool", "name": "apply_patch", "content": "patch applied"},
                            {"role": "assistant", "reasoning_content": "I should run the regression tests.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "run_tests", "arguments": {"profile": "unit"}}}]},
                            {"role": "tool", "name": "run_tests", "content": "exit=0\n4 passed"},
                            {"role": "assistant", "reasoning_content": "", "content": "Added the empty-field branch and verified the parser tests."},
                        ],
                        "verification": {"all_required_tests_pass": True},
                    },
                ]

                if DEMO_MODE:
                    raw_dataset = Dataset.from_list(demo_rows)
                else:
                    if not SOURCE_DATASET_IDS:
                        raise ValueError("Set SOURCE_DATASET_IDS to native-schema datasets.")
                    parts = [load_dataset(dataset_id, split="train") for dataset_id in SOURCE_DATASET_IDS]
                    raw_dataset = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

                print(raw_dataset)
                print(raw_dataset[0])
                """
            ),
            markdown("## Validate roles, tool calls and outcomes"),
            code(
                r"""
                allowed_roles = {"system", "developer", "user", "assistant", "tool"}
                allowed_tools = {item["function"]["name"] for item in TOOLS}
                tool_specs = {item["function"]["name"]: item["function"] for item in TOOLS}

                def validate_row(row: dict) -> list[str]:
                    errors = []
                    if row.get("tool_schema_version") != "qwen38-six-tools-v1":
                        errors.append("wrong or missing tool_schema_version")
                    if row.get("tool_schema_json") != TOOL_SCHEMA_JSON:
                        errors.append("wrong or missing canonical tool_schema_json")
                    if canonical_tool_schema(row.get("tools") or []) != TOOL_SCHEMA_JSON:
                        errors.append("row tools differ from the deployment tool surface")
                    messages = row.get("messages")
                    if not isinstance(messages, list) or not messages:
                        return ["messages must be a non-empty list"]
                    pending_tools = []
                    saw_tool_call = False
                    for index, stored_message in enumerate(messages):
                        message = _without_arrow_nulls(stored_message)
                        role = message.get("role")
                        if role not in allowed_roles:
                            errors.append(f"message {index}: unexpected role {role!r}")
                        if role == "assistant":
                            if pending_tools:
                                errors.append(f"message {index}: assistant turn before tool responses {pending_tools!r}")
                            for call in message.get("tool_calls") or []:
                                function = call.get("function", call)
                                name = function.get("name")
                                arguments = _without_arrow_nulls(function.get("arguments", {}))
                                if name not in allowed_tools:
                                    errors.append(f"message {index}: unknown tool {name!r}")
                                    continue
                                if not isinstance(arguments, dict):
                                    errors.append(f"message {index}: arguments must be a mapping")
                                    continue
                                parameters = tool_specs[name]["parameters"]
                                required = set(parameters.get("required", []))
                                properties = set(parameters.get("properties", {}))
                                if not required.issubset(arguments):
                                    errors.append(f"message {index}: {name} missing required arguments")
                                if parameters.get("additionalProperties") is False and not set(arguments).issubset(properties):
                                    errors.append(f"message {index}: {name} has unknown arguments")
                                pending_tools.append(name)
                                saw_tool_call = True
                        elif role == "tool":
                            if not pending_tools:
                                errors.append(f"message {index}: tool response without a pending call")
                            else:
                                expected = pending_tools.pop(0)
                                if message.get("name") != expected:
                                    errors.append(f"message {index}: response name {message.get('name')!r}, expected {expected!r}")
                        elif pending_tools:
                            errors.append(f"message {index}: unresolved tool responses {pending_tools!r}")
                    if pending_tools:
                        errors.append(f"trajectory ends with unresolved tool calls {pending_tools!r}")
                    if not saw_tool_call:
                        errors.append("native agent trajectory has no tool call")
                    if not row.get("verification", {}).get("all_required_tests_pass", False):
                        errors.append("trajectory is not execution-verified")
                    return errors

                validation = [validate_row(row) for row in raw_dataset]
                bad = [(index, errors) for index, errors in enumerate(validation) if errors]
                if bad:
                    raise ValueError(f"Invalid native trajectories (first 20): {bad[:20]}")
                print(f"Validated {len(raw_dataset)} native trajectories")
                """
            ),
            markdown("## Render and measure before truncation"),
            code(
                r"""
                def render_row(row: dict) -> dict:
                    messages = [_without_arrow_nulls(message) for message in row["messages"]]
                    text = render_chat(messages, add_generation_prompt=False)
                    token_count = len(tokenizer(text=text, add_special_tokens=False)["input_ids"])
                    return {
                        "messages": messages,
                        "text": text,
                        "token_count": token_count,
                        "tools": TOOLS,
                        "tool_schema_version": "qwen38-six-tools-v1",
                        "tool_schema_json": TOOL_SCHEMA_JSON,
                    }

                prepared = raw_dataset.map(render_row)
                lengths = np.array(prepared["token_count"])
                percentiles = {
                    percentile: float(np.percentile(lengths, percentile))
                    for percentile in [50, 90, 95, 99]
                }
                print({"rows": len(prepared), "tokens": int(lengths.sum()), "percentiles": percentiles, "max": int(lengths.max())})
                print(prepared[0]["text"][:5000])
                assert "<tool_call>" in prepared[0]["text"]
                assert "<tool_response>" in prepared[0]["text"]
                """
            ),
            markdown("## Optional: audit public schemas without importing their actions"),
            code(
                r"""
                if AUDIT_PUBLIC_SCHEMAS:
                    audit_rows = []
                    for dataset_id in PUBLIC_AUDIT_IDS:
                        try:
                            sample = load_dataset(dataset_id, split="train", streaming=True).take(100)
                            rows = list(sample)
                            columns = sorted({key for row in rows for key in row})
                            has_messages = sum(isinstance(row.get("messages"), list) for row in rows)
                            audit_rows.append({
                                "dataset": dataset_id,
                                "rows": len(rows),
                                "columns": columns,
                                "message_rows": has_messages,
                                "planning_direct_survival": 0,
                            })
                        except Exception as exc:
                            audit_rows.append({"dataset": dataset_id, "error": f"{type(exc).__name__}: {exc}"})
                    print(json.dumps(audit_rows, indent=2))
                else:
                    print("Public schema audit skipped; no third-party tool actions are imported by this notebook.")
                """
            ),
            markdown("## Freeze repository-family splits and publish privately"),
            code(
                r"""
                repo_families = sorted(set(prepared["repo_family"]))
                if len(repo_families) < 2:
                    raise ValueError(
                        "At least two repository families are required to create disjoint train and validation splits."
                    )
                validation_family_count = max(1, round(len(repo_families) * 0.10))
                validation_family_count = min(validation_family_count, len(repo_families) - 1)
                ranked_families = sorted(
                    repo_families,
                    key=lambda family: hashlib.sha256(family.encode()).hexdigest(),
                )
                validation_families = set(ranked_families[:validation_family_count])

                def split_name(repo_family: str) -> str:
                    return "validation" if repo_family in validation_families else "train"

                prepared = prepared.map(lambda row: {"split": split_name(row["repo_family"])})
                split_counts = Counter(prepared["split"])
                print(split_counts)

                from datasets import DatasetDict
                dataset_dict = DatasetDict({
                    split: prepared.filter(lambda row, expected=split: row["split"] == expected)
                    for split in ("train", "validation")
                })
                if not dataset_dict["train"] or not dataset_dict["validation"]:
                    raise RuntimeError(f"Split construction produced an empty partition: {split_counts}")

                PUSH_DATASET = False
                if PUSH_DATASET:
                    if DEMO_MODE:
                        raise RuntimeError("Refusing to publish the synthetic format fixture as training data.")
                    dataset_dict.push_to_hub(OUTPUT_DATASET_ID, private=True)
                    print(f"Pushed {OUTPUT_DATASET_ID}")
                else:
                    print("Set PUSH_DATASET=True only after reviewing rendered text and source licences.")
                """
            ),
            markdown(
                """
                ## Checks and next step

                Do not use the demo fixture for capability training. Proceed to
                SFT only after 100–300 successful native-schema trajectories are
                validated, the repository-family split is frozen, and the
                private dataset revision is recorded.
                """
            ),
        ],
    )


def build_03_sft():
    return notebook(
        "03 · Qwen3.8-27B coding-agent SFT",
        [
            markdown(
                """
                # 03 · SFT a native-schema coding agent

                Train a language-only LoRA on validated, replayable trajectories.
                The notebook defaults to a two-step plumbing smoke test; the real
                run remains gated until the baseline, schema-survival, split, and
                replay checks pass.

                **Input:** a private dataset from notebook 02.
                **Output:** a versioned LoRA adapter, not a merged base model.
                """
            ),
            markdown("## Install the pinned day-zero environment"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            markdown("## Run configuration"),
            code(
                r"""
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import train_on_responses_only
                from datasets import Dataset, load_dataset
                from trl import SFTConfig, SFTTrainer

                MODEL_ID = "unsloth/Qwen3.8-27B"
                DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                DATASET_REVISION = "main"  # Replace with an immutable commit SHA for a real run.
                OUTPUT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                MERGED_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-accepted-merged"
                MAX_SEQ_LENGTH = 4_096       # Use 8_192 only after the 4k memory smoke passes.
                MAX_STEPS = 2                # Replace only after the gates pass.
                DEMO_MODE = True
                RUN_TRAINING = False
                PUSH_ADAPTER = False
                SAVE_MERGED_BF16 = False
                PUSH_MERGED_BF16 = False

                if DEMO_MODE and (PUSH_ADAPTER or PUSH_MERGED_BF16 or MAX_STEPS > 2):
                    raise RuntimeError("Demo mode is limited to two local smoke steps and cannot be published.")

                run_manifest = {
                    "stage": "sft",
                    "model_id": MODEL_ID,
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "max_steps": MAX_STEPS,
                    "demo_mode": DEMO_MODE,
                    "tool_schema_version": "qwen38-six-tools-v1",
                    "harness_version": "pilot-local-v1",
                    "run_training": RUN_TRAINING,
                }
                print(json.dumps(run_manifest, indent=2))
                """
            ),
            markdown("## Load the model and discover supported LoRA targets"),
            code(
                r"""
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=MODEL_ID,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    full_finetuning=False,
                    token=hf_token,
                )

                candidate_suffixes = {
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj",
                }
                language_linear_names = [
                    name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and not any(part in name.lower() for part in ("vision", "visual", "image"))
                ]
                target_modules = sorted({
                    name.rsplit(".", 1)[-1]
                    for name in language_linear_names
                    if name.rsplit(".", 1)[-1] in candidate_suffixes
                })
                if not target_modules:
                    raise RuntimeError("No supported language LoRA targets were discovered; stop and inspect the architecture.")
                print("LoRA targets:", target_modules)

                model = FastLanguageModel.get_peft_model(
                    model,
                    r=16,
                    target_modules=target_modules,
                    lora_alpha=32,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )

                # Qwen3.8 is natively multimodal. This project tunes only language behavior.
                for name, parameter in model.named_parameters():
                    if any(part in name.lower() for part in ("vision", "visual", "image")):
                        parameter.requires_grad_(False)
                trainable_vision = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and any(part in name.lower() for part in ("vision", "visual", "image"))
                ]
                assert not trainable_vision, trainable_vision[:20]
                model.print_trainable_parameters()
                """
            ),
            markdown("## Load native-schema data and render the exact deployment template"),
            code(TOOLS_CELL),
            code(
                r"""
                def demo_rows():
                    return [
                        {
                            "repo_family": "fixture/clamp",
                            "tool_schema_version": "qwen38-six-tools-v1",
                            "tool_schema_json": TOOL_SCHEMA_JSON,
                            "tools": TOOLS,
                            "messages": [
                                {"role": "developer", "content": "Fix the bug, run tests, and keep the change minimal."},
                                {"role": "user", "content": "clamp() returns values outside its bounds."},
                                {"role": "assistant", "content": "", "tool_calls": [{
                                    "type": "function",
                                    "function": {"name": "read_file", "arguments": {"path": "src/clamp.py"}},
                                }]},
                                {"role": "tool", "name": "read_file", "content": "def clamp(x, low, high):\n    return x\n"},
                                {"role": "assistant", "content": "", "tool_calls": [{
                                    "type": "function",
                                    "function": {"name": "apply_patch", "arguments": {
                                        "patch": "--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n def clamp(x, low, high):\n-    return x\n+    return max(low, min(high, x))\n"
                                    }},
                                }]},
                                {"role": "tool", "name": "apply_patch", "content": "Done!"},
                                {"role": "assistant", "content": "Implemented the bounded clamp and kept the patch focused."},
                            ],
                        },
                        {
                            "repo_family": "fixture/parser",
                            "tool_schema_version": "qwen38-six-tools-v1",
                            "tool_schema_json": TOOL_SCHEMA_JSON,
                            "tools": TOOLS,
                            "messages": [
                                {"role": "developer", "content": "Investigate first, then make the smallest correct edit."},
                                {"role": "user", "content": "Return an empty list for an empty CSV field."},
                                {"role": "assistant", "content": "I will inspect the parser and its tests before editing."},
                            ],
                        },
                    ]

                USE_DEMO_DATA = DEMO_MODE
                if USE_DEMO_DATA:
                    raw = Dataset.from_list(demo_rows())
                    split = raw.train_test_split(test_size=0.5, seed=3407)
                    train_raw, eval_raw = split["train"], split["test"]
                    print("Using synthetic plumbing data; this is not a capability run.")
                else:
                    loaded = load_dataset(DATASET_ID, revision=DATASET_REVISION, token=hf_token)
                    missing_splits = {"train", "validation"} - set(loaded)
                    if missing_splits:
                        raise ValueError(
                            f"Dataset is missing required repository-family splits: {sorted(missing_splits)}. "
                            "Rebuild it with notebook 02."
                        )
                    if len(loaded["train"]) == 0 or len(loaded["validation"]) == 0:
                        raise ValueError("Both train and validation splits must contain at least one repository family.")
                    train_raw = loaded["train"]
                    eval_raw = loaded["validation"]

                def render_row(row):
                    if row.get("tool_schema_version") != "qwen38-six-tools-v1":
                        raise ValueError("Dataset schema version differs from this notebook.")
                    if row.get("tool_schema_json") != TOOL_SCHEMA_JSON:
                        raise ValueError("Dataset canonical tool fingerprint differs from this notebook.")
                    if canonical_tool_schema(row.get("tools") or []) != TOOL_SCHEMA_JSON:
                        raise ValueError("Dataset tools differ from the deployment tool surface.")
                    return {"text": render_chat(row["messages"], add_generation_prompt=False)}

                train_dataset = train_raw.map(render_row)
                eval_dataset = eval_raw.map(render_row)
                print(train_dataset[0]["text"][:4000])
                """
            ),
            markdown("## Build the assistant-only trainer and inspect its labels"),
            code(
                r"""
                training_args = SFTConfig(
                    output_dir=str(RUN_ROOT / "sft"),
                    dataset_text_field="text",
                    max_length=MAX_SEQ_LENGTH,
                    packing=False,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=2e-5,
                    warmup_ratio=0.05,
                    lr_scheduler_type="cosine",
                    max_steps=MAX_STEPS,
                    bf16=True,
                    fp16=False,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    logging_steps=1,
                    eval_strategy="steps",
                    eval_steps=1,
                    save_strategy="steps",
                    save_steps=1,
                    save_total_limit=2,
                    seed=3407,
                    report_to="trackio",
                    run_name="qwen38-code-sft-smoke" if USE_DEMO_DATA else "qwen38-code-sft",
                    push_to_hub=PUSH_ADAPTER,
                    hub_model_id=OUTPUT_ADAPTER_ID,
                    hub_strategy="every_save",
                )
                trainer = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    args=training_args,
                )
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|im_start|>user\n",
                    response_part="<|im_start|>assistant\n",
                )

                batch = next(iter(trainer.get_train_dataloader()))
                labels = batch["labels"]
                assert (labels != -100).any(), "No assistant tokens remain after response masking."
                first_trainable = int((labels[0] != -100).nonzero()[0])
                assert (labels[0, :first_trainable] == -100).all(), "Prompt/tool context leaked into the first response loss."

                supervised_texts = []
                for split_dataset in (trainer.train_dataset, trainer.eval_dataset):
                    for row in split_dataset:
                        supervised_ids = [
                            token_id for token_id, label in zip(row["input_ids"], row["labels"])
                            if label != -100
                        ]
                        supervised_texts.append(tokenizer.decode(supervised_ids, skip_special_tokens=False))
                joined_supervision = "\n".join(supervised_texts)
                assert "Implemented the bounded clamp" in joined_supervision, "Expected final assistant answer is masked."
                assert "def clamp(x, low, high)" not in joined_supervision, "Tool observation leaked into the loss."
                print({
                    "batch_shape": tuple(labels.shape),
                    "first_trained_token": first_trainable,
                    "supervision_preview": joined_supervision[:2000],
                })
                """
            ),
            markdown("## Train, resume, and publish the adapter"),
            code(
                r"""
                if RUN_TRAINING:
                    checkpoints = sorted((RUN_ROOT / "sft").glob("checkpoint-*"))
                    torch.cuda.reset_peak_memory_stats()
                    start_reserved_gib = torch.cuda.memory_reserved() / 1024**3
                    result = trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
                    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
                    run_manifest["train_runtime_seconds"] = result.metrics.get("train_runtime")
                    run_manifest["peak_reserved_gib"] = round(peak_reserved_gib, 3)
                    run_manifest["training_memory_delta_gib"] = round(peak_reserved_gib - start_reserved_gib, 3)
                    trainer.save_model(str(RUN_ROOT / "sft" / "final_adapter"))
                    tokenizer.save_pretrained(str(RUN_ROOT / "sft" / "final_adapter"))
                    (RUN_ROOT / "sft" / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
                    if PUSH_ADAPTER:
                        trainer.push_to_hub(commit_message="SFT adapter with native six-tool schema")
                    print(result.metrics)
                    print({
                        "peak_reserved_gib": round(peak_reserved_gib, 3),
                        "training_memory_delta_gib": round(peak_reserved_gib - start_reserved_gib, 3),
                    })
                else:
                    print("Dry run complete. Set RUN_TRAINING=True only after inspecting labels and memory.")
                """
            ),
            markdown("## Optional merged checkpoint (large and deliberately separate)"),
            code(
                r"""
                if SAVE_MERGED_BF16:
                    if PUSH_MERGED_BF16:
                        from huggingface_hub import HfApi

                        HfApi(token=hf_token).create_repo(
                            repo_id=MERGED_MODEL_ID,
                            repo_type="model",
                            private=True,
                            exist_ok=True,
                        )
                        model.push_to_hub_merged(
                            MERGED_MODEL_ID,
                            tokenizer,
                            save_method="merged_16bit",
                            token=hf_token,
                        )
                        print(f"Published private merged checkpoint to {MERGED_MODEL_ID}.")
                    else:
                        merged_dir = RUN_ROOT / "sft" / "merged_16bit"
                        model.save_pretrained_merged(
                            str(merged_dir), tokenizer, save_method="merged_16bit"
                        )
                        print(f"Saved merged checkpoint to {merged_dir}.")
                else:
                    print("Adapter-only is the default durable artifact. Merged BF16 export is disabled.")
                """
            ),
            markdown(
                """
                ## Gate to notebook 04

                Training loss is diagnostic, not success. Accept this adapter only
                if native tool syntax, held-out patch correctness, non-regression,
                and sentinel long-horizon outcomes beat the frozen baseline.
                Record the exact adapter commit SHA before preference tuning.
                """
            ),
        ],
    )


def build_04_dpo():
    return notebook(
        "04 · Qwen3.8-27B coding preference tuning",
        [
            markdown(
                """
                # 04 · Preference tuning after SFT

                Use DPO only on preferences whose chosen answer is demonstrably
                better under the same harness and hidden verifier. This stage is
                intentionally smaller than SFT and cannot repair a broken tool schema.
                """
            ),
            markdown("## Install and authenticate"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            code(
                r"""
                from unsloth import FastLanguageModel
                from datasets import Dataset, load_dataset
                from trl import DPOConfig, DPOTrainer

                SFT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                SFT_ADAPTER_REVISION = "REPLACE_WITH_ACCEPTED_COMMIT"
                PREFERENCE_DATASET_ID = f"{HF_USERNAME}/qwen38-code-preferences"
                PREFERENCE_DATASET_REVISION = "main"
                OUTPUT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-dpo-lora"
                MAX_SEQ_LENGTH = 4_096
                MAX_STEPS = 2
                DEMO_MODE = True
                RUN_TRAINING = False
                PUSH_ADAPTER = False

                if RUN_TRAINING and SFT_ADAPTER_REVISION.startswith("REPLACE_"):
                    raise RuntimeError("Pin the accepted SFT adapter commit before DPO.")
                if DEMO_MODE and (PUSH_ADAPTER or MAX_STEPS > 2):
                    raise RuntimeError("Demo preferences are limited to two local smoke steps and cannot be published.")
                """
            ),
            markdown("## Load the accepted SFT adapter"),
            code(
                r"""
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=SFT_ADAPTER_ID,
                    revision=None if SFT_ADAPTER_REVISION.startswith("REPLACE_") else SFT_ADAPTER_REVISION,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    token=hf_token,
                )
                tokenizer.padding_side = "left"
                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                total = sum(parameter.numel() for parameter in model.parameters())
                if not trainable:
                    raise RuntimeError("The SFT adapter loaded without trainable parameters; inspect PEFT loading before DPO.")
                print({"trainable": trainable, "total": total, "fraction": trainable / total})
                """
            ),
            markdown("## Render prompt/chosen/rejected with the native Qwen3.8 template"),
            code(TOOLS_CELL),
            code(
                r"""
                demo_preferences = Dataset.from_list([
                    {
                        "prompt_messages": [
                            {"role": "developer", "content": "Make the smallest correct change and report verification."},
                            {"role": "user", "content": "The bounds check is inverted; what did you change?"},
                        ],
                        "chosen_message": {"role": "assistant", "content": "Corrected only the inverted comparison and verified the focused unit tests pass."},
                        "rejected_message": {"role": "assistant", "content": "Rewrote the entire module and skipped tests."},
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "infra_status": "ok",
                    },
                    {
                        "prompt_messages": [
                            {"role": "developer", "content": "Inspect evidence before proposing a patch."},
                            {"role": "user", "content": "A parser test fails only for empty input."},
                        ],
                        "chosen_message": {"role": "assistant", "content": "I would first inspect the failing test and empty-input branch before editing."},
                        "rejected_message": {"role": "assistant", "content": "Delete the failing test."},
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "infra_status": "ok",
                    },
                ])

                USE_DEMO_DATA = DEMO_MODE
                if USE_DEMO_DATA:
                    raw = demo_preferences
                    print("Using synthetic plumbing preferences; this is not a capability run.")
                else:
                    raw = load_dataset(
                        PREFERENCE_DATASET_ID,
                        split="train",
                        revision=PREFERENCE_DATASET_REVISION,
                        token=hf_token,
                    )

                def render_preference(row):
                    if row.get("infra_status") != "ok":
                        raise ValueError("Infrastructure failures must not become preferences.")
                    if not row["chosen_reward"] > row["rejected_reward"]:
                        raise ValueError("Chosen reward must be strictly greater than rejected reward.")
                    prompt = canonical_to_qwen(row["prompt_messages"])
                    prompt_text = tokenizer.apply_chat_template(
                        prompt,
                        tools=TOOLS,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                        reasoning_effort="medium",
                    )

                    def completion(message):
                        full = tokenizer.apply_chat_template(
                            prompt + [message],
                            tools=TOOLS,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=True,
                            reasoning_effort="medium",
                            preserve_thinking=True,
                        )
                        if not full.startswith(prompt_text):
                            raise ValueError("Template prefix drift: chosen/rejected cannot be separated safely.")
                        return full[len(prompt_text):]

                    return {
                        "prompt": prompt_text,
                        "chosen": completion(row["chosen_message"]),
                        "rejected": completion(row["rejected_message"]),
                    }

                preferences = raw.map(render_preference, remove_columns=raw.column_names)
                split = preferences.train_test_split(test_size=0.5 if len(preferences) < 20 else 0.1, seed=3407)
                print({key: split["train"][0][key][:1000] for key in ["prompt", "chosen", "rejected"]})
                """
            ),
            markdown("## Configure and optionally run DPO"),
            code(
                r"""
                dpo_args = DPOConfig(
                    output_dir=str(RUN_ROOT / "dpo"),
                    max_length=MAX_SEQ_LENGTH,
                    beta=0.1,
                    loss_type="sigmoid",
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=5e-7,
                    warmup_ratio=0.05,
                    lr_scheduler_type="cosine",
                    max_steps=MAX_STEPS,
                    bf16=True,
                    optim="adamw_8bit",
                    logging_steps=1,
                    eval_strategy="steps",
                    eval_steps=1,
                    save_strategy="steps",
                    save_steps=1,
                    save_total_limit=2,
                    precompute_ref_log_probs=True,
                    report_to="trackio",
                    run_name="qwen38-code-dpo-smoke" if USE_DEMO_DATA else "qwen38-code-dpo",
                    push_to_hub=PUSH_ADAPTER,
                    hub_model_id=OUTPUT_ADAPTER_ID,
                    hub_strategy="every_save",
                    seed=3407,
                )
                trainer = DPOTrainer(
                    model=model,
                    ref_model=None,
                    args=dpo_args,
                    processing_class=tokenizer,
                    train_dataset=split["train"],
                    eval_dataset=split["test"],
                )

                if RUN_TRAINING:
                    result = trainer.train()
                    trainer.save_model(str(RUN_ROOT / "dpo" / "final_adapter"))
                    if PUSH_ADAPTER:
                        trainer.push_to_hub(commit_message="DPO adapter from verifier-backed preferences")
                    print(result.metrics)
                else:
                    print("DPO dry run configured. Inspect rendered pairs before setting RUN_TRAINING=True.")
                """
            ),
            markdown(
                """
                ## Acceptance gate

                Compare the DPO adapter against the accepted SFT adapter on the
                same frozen tasks. Reject it if patch correctness, native tool
                validity, or reasoning-retention policy regresses—even if the
                preference objective improves.
                """
            ),
        ],
    )


def build_05_grpo():
    return notebook(
        "05 · Qwen3.8-27B agentic GRPO pilot",
        [
            markdown(
                """
                # 05 · Agentic GRPO pilot

                This is a deliberately tiny, stateful coding environment. The
                core Colab stack can validate its tools, hidden reward and
                reward-hacking fixtures, but intentionally does not install
                TRL's newer `environment_factory` API: that TRL release conflicts
                with the current Unsloth dependency bounds. Trainer construction
                remains gated until a compatible Unsloth/TRL pair or a separate
                NeMo Gym/Harbor rollout backend is validated.
                """
            ),
            markdown("## Install and authenticate"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            code(
                r"""
                import hashlib
                import inspect
                import re
                import shutil
                import tempfile
                from pathlib import Path

                from unsloth import FastLanguageModel
                from datasets import Dataset
                from packaging.version import Version
                from transformers import __version__ as transformers_version
                from trl import GRPOConfig, GRPOTrainer

                ACCEPTED_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                ACCEPTED_REVISION = "REPLACE_WITH_ACCEPTED_COMMIT"
                OUTPUT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-grpo-lora"
                MAX_SEQ_LENGTH = 4_096
                MAX_STEPS = 2
                ROLLOUT_POLICY_PRECISION = "bf16"  # Set to "bnb4" only as an explicit approximation experiment.
                ALLOW_QUANTIZED_ROLLOUT_POLICY = False
                RUN_TRAINING = False
                PUSH_ADAPTER = False

                AGENTIC_TRL_AVAILABLE = (
                    Version(transformers_version) >= Version("5.2.0")
                    and "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters
                )
                AGENTIC_RL_BLOCKER = (
                    "The reviewed core stack pins TRL 0.22.2 for Unsloth compatibility; "
                    "TRL environment_factory starts at 0.29.0, outside the pinned "
                    "Unsloth Zoo trl<=0.24.0 constraint."
                )
                if RUN_TRAINING and not AGENTIC_TRL_AVAILABLE:
                    raise RuntimeError(AGENTIC_RL_BLOCKER)
                if RUN_TRAINING and ACCEPTED_REVISION.startswith("REPLACE_"):
                    raise RuntimeError("Pin an accepted adapter revision before RL.")
                if ROLLOUT_POLICY_PRECISION not in {"bf16", "bnb4"}:
                    raise ValueError("ROLLOUT_POLICY_PRECISION must be 'bf16' or 'bnb4'.")
                if ROLLOUT_POLICY_PRECISION == "bnb4" and not ALLOW_QUANTIZED_ROLLOUT_POLICY:
                    raise RuntimeError(
                        "A quantized rollout/update policy changes the accepted BF16 policy distribution. "
                        "Set ALLOW_QUANTIZED_ROLLOUT_POLICY=True only for a labelled approximation experiment."
                    )
                if PUSH_ADAPTER:
                    raise RuntimeError("This notebook contains only a toy RL environment; do not publish its adapter.")
                """
            ),
            markdown("## Define a rooted toy coding environment and hidden verifier"),
            code(
                r'''
                class ToyCodingEnv:
                    """Fresh, isolated clamp task for one rollout."""

                    def __init__(self):
                        self._tmp = None
                        self.root = None
                        self.tool_errors = 0
                        self.test_ran = False
                        self.visible_passed = False
                        self.expected_test_digest = None

                    def reset(self, **kwargs) -> str:
                        """Create a clean task and return its user prompt."""
                        if self._tmp is not None:
                            shutil.rmtree(self._tmp, ignore_errors=True)
                        self._tmp = Path(tempfile.mkdtemp(prefix="qwen38_grpo_"))
                        self.root = self._tmp / "repo"
                        (self.root / "src").mkdir(parents=True)
                        (self.root / "tests").mkdir()
                        (self.root / "src" / "clamp.py").write_text(
                            "def clamp(value, low, high):\n    return value\n"
                        )
                        test_source = (
                            "from src.clamp import clamp\n\n"
                            "def test_inside(): assert clamp(2, 0, 4) == 2\n"
                            "def test_low(): assert clamp(-1, 0, 4) == 0\n"
                        )
                        (self.root / "tests" / "test_clamp.py").write_text(test_source)
                        self.expected_test_digest = hashlib.sha256(test_source.encode()).hexdigest()
                        self.tool_errors = 0
                        self.test_ran = False
                        self.visible_passed = False
                        return "Repository ready. Fix clamp() so it respects both bounds, then run unit tests."

                    def _path(self, relative: str) -> Path:
                        candidate = (self.root / relative).resolve()
                        if candidate != self.root and self.root not in candidate.parents:
                            raise ValueError("Path escapes repository root")
                        return candidate

                    def list_files(self, path: str) -> str:
                        """List repository files under path.

                        Args:
                            path: Repository-relative directory.

                        Returns:
                            A newline-delimited file list.
                        """
                        base = self._path(path)
                        return "\n".join(str(item.relative_to(self.root)) for item in sorted(base.rglob("*")) if item.is_file())

                    def read_file(self, path: str) -> str:
                        """Read a bounded UTF-8 repository file.

                        Args:
                            path: Repository-relative file.

                        Returns:
                            Bounded file contents.
                        """
                        return self._path(path).read_text()[:20_000]

                    def _tests_intact(self) -> bool:
                        test_path = self.root / "tests" / "test_clamp.py"
                        if not test_path.is_file():
                            return False
                        digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
                        return digest == self.expected_test_digest

                    def search(self, query: str) -> str:
                        """Search repository text with a regular expression.

                        Args:
                            query: Python regular expression.

                        Returns:
                            Matching path, line, and text records.
                        """
                        regex = re.compile(query)
                        hits = []
                        for file in self.root.rglob("*.py"):
                            for line_no, line in enumerate(file.read_text().splitlines(), 1):
                                if regex.search(line):
                                    hits.append(f"{file.relative_to(self.root)}:{line_no}:{line}")
                        return "\n".join(hits[:200])

                    def apply_patch(self, patch: str) -> str:
                        """Apply the one-file unified replacement used by the pilot.

                        Args:
                            patch: Patch text targeting src/clamp.py.

                        Returns:
                            Patch status text.
                        """
                        if "src/clamp.py" not in patch or "return max(low, min(high, value))" not in patch:
                            self.tool_errors += 1
                            return "Rejected: pilot accepts only the expected minimal clamp edit."
                        target = self.root / "src" / "clamp.py"
                        before = target.read_text()
                        if "return value" not in before:
                            return "Already applied."
                        target.write_text(before.replace("return value", "return max(low, min(high, value))"))
                        return "Done!"

                    def run_tests(self, profile: str) -> str:
                        """Run the allow-listed visible unit profile.

                        Args:
                            profile: Must be `unit`.

                        Returns:
                            Visible-test status text.
                        """
                        if profile != "unit":
                            self.tool_errors += 1
                            return "Rejected profile."
                        self.test_ran = True
                        if not self._tests_intact():
                            self.visible_passed = False
                            return "Test integrity failure: required visible tests were changed or deleted."
                        namespace = {}
                        exec((self.root / "src" / "clamp.py").read_text(), namespace)
                        fn = namespace["clamp"]
                        self.visible_passed = fn(2, 0, 4) == 2 and fn(-1, 0, 4) == 0
                        return "2 passed" if self.visible_passed else "1 failed"

                    def shell(self, command: str) -> str:
                        """Reject arbitrary shell execution in the pilot.

                        Args:
                            command: Requested command.

                        Returns:
                            A rejection message.
                        """
                        self.tool_errors += 1
                        return "shell is disabled; use the semantic tools"

                    def get_reward(self) -> float:
                        """Score hidden behavior; infrastructure failures would return no sample upstream."""
                        try:
                            if not self._tests_intact():
                                return 0.0
                            namespace = {}
                            exec((self.root / "src" / "clamp.py").read_text(), namespace)
                            fn = namespace["clamp"]
                            hidden = fn(9, 0, 4) == 4 and fn(-9, 0, 4) == 0 and fn(2, 2, 2) == 2
                            reward = 0.2 * float(self.visible_passed and self.test_ran) + 0.8 * float(hidden)
                            return max(0.0, reward - 0.05 * self.tool_errors)
                        except (KeyError, SyntaxError, TypeError, ValueError):
                            return 0.0
                        finally:
                            shutil.rmtree(self._tmp, ignore_errors=True)
                '''
            ),
            markdown("## Unit-test the reward, including hacking fixtures"),
            code(
                r"""
                untouched = ToyCodingEnv()
                untouched.reset()
                assert untouched.get_reward() == 0.0

                gold = ToyCodingEnv()
                gold.reset()
                gold.apply_patch("--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n def clamp(value, low, high):\n-    return value\n+    return max(low, min(high, value))\n")
                assert gold.run_tests("unit") == "2 passed"
                assert gold.get_reward() == 1.0

                escaped = ToyCodingEnv()
                escaped.reset()
                try:
                    escaped.read_file("../secret")
                    raise AssertionError("Path escape was not blocked")
                except ValueError:
                    pass
                finally:
                    shutil.rmtree(escaped._tmp, ignore_errors=True)

                deletion = ToyCodingEnv()
                deletion.reset()
                deletion.apply_patch("--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n def clamp(value, low, high):\n-    return value\n+    return max(low, min(high, value))\n")
                (deletion.root / "tests" / "test_clamp.py").unlink()
                assert deletion.run_tests("unit").startswith("Test integrity failure")
                assert deletion.get_reward() == 0.0
                print("Reward and reward-hacking fixtures passed.")
                """
            ),
            markdown("## Load the accepted adapter and configure multi-turn GRPO"),
            code(
                r"""
                use_quantized_policy = ROLLOUT_POLICY_PRECISION == "bnb4"
                policy_manifest = {
                    "accepted_adapter_id": ACCEPTED_ADAPTER_ID,
                    "accepted_revision": ACCEPTED_REVISION,
                    "rollout_update_precision": ROLLOUT_POLICY_PRECISION,
                    "quantized_policy_approximation": use_quantized_policy,
                    "allow_quantized_rollout_policy": ALLOW_QUANTIZED_ROLLOUT_POLICY,
                    "agentic_trl_available": AGENTIC_TRL_AVAILABLE,
                    "agentic_rl_blocker": None if AGENTIC_TRL_AVAILABLE else AGENTIC_RL_BLOCKER,
                }
                grpo_root = RUN_ROOT / "grpo"
                grpo_root.mkdir(parents=True, exist_ok=True)
                (grpo_root / "policy_manifest.json").write_text(json.dumps(policy_manifest, indent=2))
                print(json.dumps(policy_manifest, indent=2))

                trainer = None
                if AGENTIC_TRL_AVAILABLE:
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=ACCEPTED_ADAPTER_ID,
                        revision=None if ACCEPTED_REVISION.startswith("REPLACE_") else ACCEPTED_REVISION,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=None if use_quantized_policy else torch.bfloat16,
                        load_in_4bit=use_quantized_policy,
                        token=hf_token,
                    )
                    if not any(parameter.requires_grad for parameter in model.parameters()):
                        raise RuntimeError("Accepted adapter has no trainable parameters; inspect PEFT loading before RL.")

                    grpo_args = GRPOConfig(
                        output_dir=str(grpo_root),
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=2,
                        num_generations=2,
                        max_completion_length=1_024,
                        learning_rate=5e-6,
                        max_steps=MAX_STEPS,
                        bf16=True,
                        optim="adamw_8bit",
                        logging_steps=1,
                        save_strategy="steps",
                        save_steps=1,
                        save_total_limit=2,
                        mask_truncated_completions=True,
                        scale_rewards="batch",
                        loss_type="dr_grpo",
                        report_to="trackio",
                        run_name="qwen38-code-agent-grpo-smoke",
                        push_to_hub=PUSH_ADAPTER,
                        hub_model_id=OUTPUT_ADAPTER_ID,
                        seed=3407,
                    )
                    trainer = GRPOTrainer(
                        model=model,
                        args=grpo_args,
                        processing_class=tokenizer,
                        environment_factory=ToyCodingEnv,
                    )
                    print("GRPO environment and trainer constructed.")
                else:
                    print(f"Trainer construction skipped: {AGENTIC_RL_BLOCKER}")
                """
            ),
            markdown("## Run only after the toy rollouts work manually"),
            code(
                r"""
                if RUN_TRAINING:
                    if trainer is None:
                        raise RuntimeError(AGENTIC_RL_BLOCKER)
                    result = trainer.train()
                    trainer.save_model(str(RUN_ROOT / "grpo" / "final_adapter"))
                    if PUSH_ADAPTER:
                        trainer.push_to_hub(commit_message="Agentic GRPO pilot adapter")
                    print(result.metrics)
                    print("Inspect reward zero-variance metrics; a collapsed group supplies no learning signal.")
                else:
                    print("RL is off. First inspect generated episodes and confirm hidden rewards manually.")
                """
            ),
            markdown(
                """
                ## Scale-up boundary

                Do not turn this fixture into the production harness. After the
                reward fixtures pass, resolve the recorded Unsloth/TRL blocker
                or use a separate adapter around Harbor/Terminal-Bench or NeMo
                Gym. Retain the same six tools, hidden-test boundary, failure
                taxonomy and reward tests. Run two independent seeds before
                accepting RL.
                """
            ),
        ],
    )


def build_06_qat_export():
    return notebook(
        "06 · Qwen3.8-27B QAT and quantization exports",
        [
            markdown(
                """
                # 06 · QAT and deployment artifacts

                QAT/TorchAO, ordinary GGUF post-training quantization, and
                Unsloth Dynamic GGUF are different experiments. This notebook
                keeps their artifacts and acceptance decisions separate.

                A custom 1-bit 27B path is research work, not a supported export.
                Start with Q4/Q3/Q2 candidates and let long-horizon evaluation decide.
                """
            ),
            markdown("## Install the pinned core environment"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            markdown("## Install TorchAO/Fbgemm versions matched to Colab PyTorch"),
            code(
                r"""
                import re

                torch_minor = re.match(r"\d+\.\d+", torch.__version__).group(0)
                torchao_by_torch = {"2.8": "0.16.0", "2.9": "0.16.0", "2.10": "0.16.0", "2.11": "0.18.0"}
                fbgemm_by_torch = {"2.8": "1.3.0", "2.9": "1.4.2", "2.10": "1.5.0", "2.11": "1.5.0"}
                if torch_minor not in torchao_by_torch:
                    raise RuntimeError(
                        f"No reviewed TorchAO pin for torch {torch_minor}; update the mapping from the Unsloth notebook catalog."
                    )
                INSTALL_QAT_DEPS = False
                if INSTALL_QAT_DEPS:
                    import numpy as np

                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                        f"torchao=={torchao_by_torch[torch_minor]}",
                        f"fbgemm-gpu-genai=={fbgemm_by_torch[torch_minor]}",
                        f"numpy=={np.__version__}",
                    ])
                    print("Restart the runtime, rerun setup, then leave INSTALL_QAT_DEPS=False.")
                else:
                    print({"torch": torch.__version__, "planned_torchao": torchao_by_torch[torch_minor], "planned_fbgemm": fbgemm_by_torch[torch_minor]})
                """
            ),
            markdown("## Artifact configuration"),
            code(
                r"""
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import train_on_responses_only
                from datasets import Dataset, load_dataset
                from trl import SFTConfig, SFTTrainer

                ACCEPTED_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                ACCEPTED_REVISION = "REPLACE_WITH_ACCEPTED_COMMIT"
                MERGED_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-accepted-merged"
                QAT_OUTPUT_ID = f"{HF_USERNAME}/qwen38-27b-code-qat-int4"
                GGUF_OUTPUT_ID = f"{HF_USERNAME}/qwen38-27b-code-gguf"
                DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                DATASET_REVISION = "REPLACE_WITH_DATASET_COMMIT"
                MAX_SEQ_LENGTH = 4_096

                RUN_QAT = False
                PUSH_QAT = False
                RUN_STANDARD_GGUF_EXPORT = False
                BUILD_CALIBRATION_CORPUS = False

                if any([RUN_QAT, RUN_STANDARD_GGUF_EXPORT, BUILD_CALIBRATION_CORPUS]):
                    if ACCEPTED_REVISION.startswith("REPLACE_"):
                        raise RuntimeError("Pin the accepted adapter revision before export.")
                """
            ),
            markdown("## QAT-LoRA branch (fresh adapter from an accepted merged checkpoint)"),
            code(
                r"""
                if RUN_QAT:
                    try:
                        import torchao
                        from torchao.quantization import quantize_
                        from torchao.quantization.qat import QATConfig
                    except ImportError as exc:
                        raise RuntimeError("Install the matched TorchAO/Fbgemm pair and restart first.") from exc

                    qat_model, qat_tokenizer = FastLanguageModel.from_pretrained(
                        model_name=MERGED_MODEL_ID,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=torch.bfloat16,
                        load_in_4bit=False,
                        token=hf_token,
                    )
                    qat_model = FastLanguageModel.get_peft_model(
                        qat_model,
                        r=16,
                        target_modules=[
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj",
                        ],
                        lora_alpha=32,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=3407,
                        qat_scheme="int4",
                    )
                    for name, parameter in qat_model.named_parameters():
                        if any(part in name.lower() for part in ("vision", "visual", "image")):
                            parameter.requires_grad_(False)
                    fake_quant_modules = [
                        module.__class__.__name__ for module in qat_model.modules()
                        if "FakeQuantized" in module.__class__.__name__
                    ]
                    if not fake_quant_modules:
                        raise RuntimeError("qat_scheme did not install fake-quantized modules; stop before training.")
                    print({"fake_quantized_modules": len(fake_quant_modules)})

                    qat_data = load_dataset(
                        DATASET_ID,
                        split="train",
                        revision=DATASET_REVISION,
                        token=hf_token,
                    )
                    if "text" not in qat_data.column_names:
                        raise RuntimeError("Publish the rendered `text` field from notebook 02 before QAT.")
                    qat_args = SFTConfig(
                        output_dir=str(RUN_ROOT / "qat"),
                        dataset_text_field="text",
                        max_length=MAX_SEQ_LENGTH,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=8,
                        learning_rate=5e-6,
                        max_steps=100,
                        bf16=True,
                        optim="adamw_8bit",
                        logging_steps=1,
                        save_steps=25,
                        report_to="trackio",
                        run_name="qwen38-code-qat-int4",
                    )
                    qat_trainer = SFTTrainer(
                        model=qat_model,
                        processing_class=qat_tokenizer,
                        train_dataset=qat_data,
                        args=qat_args,
                    )
                    qat_trainer = train_on_responses_only(
                        qat_trainer,
                        instruction_part="<|im_start|>user\n",
                        response_part="<|im_start|>assistant\n",
                    )
                    qat_labels = next(iter(qat_trainer.get_train_dataloader()))["labels"]
                    assert (qat_labels != -100).any(), "QAT response masking removed every target token."
                    qat_trainer.train()

                    # Convert the fake-quantized representation only after training.
                    quantize_(qat_model, QATConfig(step="convert"))
                    qat_dir = RUN_ROOT / "qat" / "torchao_int4"
                    qat_model.save_pretrained_torchao(
                        str(qat_dir),
                        qat_tokenizer,
                    )
                    qat_tokenizer.save_pretrained(str(qat_dir))
                    if PUSH_QAT:
                        from huggingface_hub import HfApi
                        HfApi(token=hf_token).upload_folder(
                            repo_id=QAT_OUTPUT_ID,
                            folder_path=str(qat_dir),
                            repo_type="model",
                        )
                else:
                    print("QAT is disabled. It requires an accepted merged source and matched TorchAO installation.")
                """
            ),
            markdown("## Standard GGUF control artifacts"),
            code(
                r"""
                if RUN_STANDARD_GGUF_EXPORT:
                    export_model, export_tokenizer = FastLanguageModel.from_pretrained(
                        model_name=ACCEPTED_ADAPTER_ID,
                        revision=ACCEPTED_REVISION,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=torch.bfloat16,
                        load_in_4bit=False,
                        token=hf_token,
                    )
                    export_model.push_to_hub_gguf(
                        GGUF_OUTPUT_ID,
                        export_tokenizer,
                        quantization_method=["q8_0", "q5_k_m", "q4_k_m"],
                        token=hf_token,
                    )
                    print("Published standard llama.cpp GGUF controls. These are not Unsloth Dynamic quants.")
                else:
                    print("Standard GGUF export is disabled; it can take substantial disk, RAM, and upload time.")
                """
            ),
            markdown("## Build a native-template calibration corpus for later low-bit conversion"),
            code(
                r"""
                if BUILD_CALIBRATION_CORPUS:
                    from transformers import AutoTokenizer

                    calibration_tokenizer = AutoTokenizer.from_pretrained(
                        ACCEPTED_ADAPTER_ID,
                        revision=ACCEPTED_REVISION,
                        token=hf_token,
                    )
                    calibration = load_dataset(
                        DATASET_ID,
                        split="train",
                        revision=DATASET_REVISION,
                        token=hf_token,
                    )
                    texts = calibration["text"][:512]
                    token_lengths = [
                        len(calibration_tokenizer(text=text, add_special_tokens=False)["input_ids"])
                        for text in texts
                    ]
                    calibration_path = RUN_ROOT / "calibration_native_tools.txt"
                    calibration_path.write_text("\n<|calibration_document|>\n".join(texts))
                    print({
                        "path": str(calibration_path),
                        "documents": len(texts),
                        "tokens": sum(token_lengths),
                        "max_tokens": max(token_lengths),
                    })
                else:
                    print("Calibration build is disabled.")
                """
            ),
            markdown(
                """
                ## Dynamic Q3/Q2 and the 1-bit boundary

                Do not relabel the standard exports above as Dynamic GGUF. First
                evaluate the published Unsloth Dynamic Q4/Q3/Q2 artifacts, if
                available for the accepted checkpoint. A custom importance-matrix
                conversion must pin the exact `llama.cpp` and Unsloth converter
                revisions and use the native-template calibration corpus.

                There is no supported one-bit Qwen3.8-27B path in this suite.
                Treat it as a separate research branch only after Q2 fails the
                frozen long-horizon gate. Quantization acceptance is based on
                complete episode success, not perplexity alone.
                """
            ),
        ],
    )


def build_legacy_pointer():
    return notebook(
        "Training notebook moved",
        [
            markdown(
                """
                # Training notebook moved

                The executable Colab suite now lives in [`notebooks/`](../../notebooks/README.md).
                Start with `00_colab_preflight.ipynb`; the SFT notebook replacing
                this placeholder is [`03_sft_lora.ipynb`](../../notebooks/03_sft_lora.ipynb).
                """
            )
        ],
    )


def validate_notebook(nb, path: Path) -> None:
    nbf.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            raise SyntaxError(f"{path}: code cell {index}: {exc}") from exc


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    outputs = {
        NOTEBOOKS / "00_colab_preflight.ipynb": build_00_preflight(),
        NOTEBOOKS / "01_tool_calling_baseline.ipynb": build_01_baseline(),
        NOTEBOOKS / "02_prepare_sft_data.ipynb": build_02_data(),
        NOTEBOOKS / "03_sft_lora.ipynb": build_03_sft(),
        NOTEBOOKS / "04_dpo_preferences.ipynb": build_04_dpo(),
        NOTEBOOKS / "05_agentic_grpo.ipynb": build_05_grpo(),
        NOTEBOOKS / "06_qat_and_export.ipynb": build_06_qat_export(),
        ROOT / "src" / "qwen3_8_27b_code" / "train.ipynb": build_legacy_pointer(),
    }
    for path, nb in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        validate_notebook(nb, path)
        nbf.write(nb, path)
        print(f"wrote {path.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
