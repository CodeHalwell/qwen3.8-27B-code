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
import gc
import json
import os
import platform
import sys
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

# IPython stores the last exception on sys.last_traceback, whose frames keep
# every local alive, including a ~52 GiB model from a failed cell. gc.collect()
# cannot free what those frames still reference.
def release_stale_gpu_state() -> float:
    for _stale_name in ("model", "tokenizer", "processor", "trainer"):
        globals().pop(_stale_name, None)
    for _exc_attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, _exc_attr):
            delattr(sys, _exc_attr)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except AttributeError:
        pass
    return torch.cuda.mem_get_info()[0] / 1024**3

# Fail before a model load that accelerate would silently offload.
def require_free_vram(minimum_gib: float) -> float:
    free_gib = release_stale_gpu_state()
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB VRAM is free but this load needs about "
            f"{minimum_gib:.0f} GiB. A previous model in this kernel is still "
            "holding memory. Restart the runtime and rerun from the top."
        )
    return free_gib

# Reject a load that accelerate quietly spilled to CPU or disk. A partially
# offloaded model copies weights back per forward pass (the 2.4 GiB embedding
# alone) and is guaranteed to OOM or crawl mid-episode.
def assert_model_fully_resident(model, minimum_free_gib: float = 4.0) -> None:
    non_cuda = sorted({
        parameter.device.type
        for parameter in model.parameters()
        if parameter.device.type != "cuda"
    })
    offload_hooks = [
        name for name, module in model.named_modules()
        if getattr(getattr(module, "_hf_hook", None), "offload", False)
    ]
    if non_cuda or offload_hooks:
        raise RuntimeError(
            "The checkpoint did not fit on the GPU and accelerate offloaded "
            f"part of it (devices={non_cuda}, offload_hooks={len(offload_hooks)}). "
            "Restart the runtime to release stale VRAM, then rerun from the top."
        )
    free_gib = torch.cuda.mem_get_info()[0] / 1024**3
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB VRAM is free after the load; the KV "
            "cache and generation workspaces need headroom. Restart the "
            "runtime and rerun from the top."
        )
    print(f"Model fully resident on GPU; {free_gib:.1f} GiB VRAM free.")

release_stale_gpu_state()

hf_token = userdata.get("HF_TOKEN") if userdata is not None else os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Add a write-capable HF_TOKEN to Colab Secrets before continuing.")
login(token=hf_token, add_to_git_credential=False)
HF_USERNAME = whoami()["name"]


def require_private_repo(repo_id: str, repo_type: str = "model") -> None:
    # Refuse to publish into a Hub repo that already exists and is public.
    # private=True on create_repo, push_to_hub and hub_private_repo applies
    # only when the repo is created; an existing public repo stays public
    # and every later push lands in the open.
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    if not api.repo_exists(repo_id, repo_type=repo_type):
        return
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise RuntimeError(
            f"{repo_type} repo {repo_id} exists and is public. Make it private first with "
            f"HfApi(token=hf_token).update_repo_settings(repo_id={repo_id!r}, repo_type={repo_type!r}, "
            "private=True), or publish under a new id."
        )

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
# Bumped from v1 when the `shell` description stopped carrying pilot status
# text. Tool descriptions are model inputs and part of the fingerprint, so a
# wording change is a schema change.
TOOL_SCHEMA_VERSION = "qwen38-six-tools-v3"

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
            "description": "Run one bash command in the repository; output is bounded and a non-zero exit code is reported.",
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

def unify_columns(rows: list[dict]) -> list[dict]:
    """Give every row every key that any row in the list carries.

    ``Dataset.from_list`` names its columns from the first row alone, so a
    key that row happens not to carry is dropped from the whole table: a
    corpus whose bootstrap rows predate ``lane`` silently loses the lane of
    every public row after them, and non-agentic rows are then read as
    agentic. Filling the gaps with None keeps each row's own value and
    leaves the absent ones null, which is what the readers already expect.
    """
    columns = sorted({key for row in rows for key in row})
    return [{column: row.get(column) for column in columns} for row in rows]

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
    """Merge the leading policy messages into one system message.

    The Qwen3.8 template accepts `developer` natively and merges a run of
    leading system/developer messages itself. This fold is therefore not a
    compatibility shim: it exists so training and deployment both hand the
    template one deterministically joined policy message.
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

def text_tokenizer_of(tokenizer):
    """The text tokenizer behind a multimodal processor, or the tokenizer itself.

    FastModel hands back a processor for this vision-capable checkpoint. It
    renders chat templates and decodes, but a bare positional string is read
    as an image and token-level attributes (eos, added tokens) live one
    level down. Reach through for those, and pass text= otherwise.
    """
    return getattr(tokenizer, "tokenizer", tokenizer)


def render_chat(messages: list[dict], *, add_generation_prompt: bool, reasoning_effort: str = "medium") -> str:
    return text_tokenizer_of(tokenizer).apply_chat_template(
        canonical_to_qwen(messages),
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
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
                from unsloth import FastModel

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

                require_free_vram(60.0)
                torch.cuda.reset_peak_memory_stats()
                model, tokenizer = FastModel.from_pretrained(**load_kwargs)
                assert_model_fully_resident(model)
                peak_gib = torch.cuda.max_memory_reserved() / 1024**3
                print(f"Loaded {MODEL_ID}; peak reserved VRAM={peak_gib:.2f} GiB")
                # Preflight item 3 of docs/model-and-hardware.md: FastModel returns
                # a processor for this vision-capable checkpoint, with the text
                # tokenizer one level down. Record both so later notebooks can
                # rely on it.
                print({
                    "loader": "FastModel",
                    "returned": type(tokenizer).__name__,
                    "text_tokenizer": type(getattr(tokenizer, "tokenizer", tokenizer)).__name__,
                })

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
                FastModel.for_inference(model)
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

                from unsloth import FastModel

                MODEL_ID = "unsloth/Qwen3.8-27B"
                # Every think block after the request stays in context for the rest
                # of the episode, so a long-horizon run needs the window sized for
                # thirty turns of reasoning, not one.
                MAX_SEQUENCE_LENGTH = 32_768
                # Thinking mode spends tokens before the tool call appears, so a
                # 1k cap truncated ordinary turns and scored them as answers. This
                # is a medium-effort cap; notebook 07 sizes one per effort.
                MAX_NEW_TOKENS_PER_TURN = 2048
                # Ceilings, not targets: the long band of docs/evaluation.md runs to
                # 30 tool calls, and a smaller budget excludes it by construction.
                MAX_TOOL_CALLS = 30
                EPISODE_TIMEOUT_SECONDS = 900
                BASELINE_SEEDS = (3407, 9176, 20261)
                DEMO_MODE = True
                PILOT_MANIFEST = Path("/content/pilot_tasks.jsonl")
                RESULTS_DIR = RUN_ROOT / "baseline"
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                secret_markers = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
                TASK_ENV = {
                    key: value for key, value in os.environ.items()
                    if not any(marker in key.upper() for marker in secret_markers)
                }
                # A same-length patch applied within the mtime second of the last
                # test run revalidates a stale .pyc (bytecode headers store whole
                # seconds), silently testing pre-patch code. Never cache bytecode
                # inside task repositories.
                TASK_ENV["PYTHONDONTWRITEBYTECODE"] = "1"

                # Dropping secret-looking variable names is not enough on its own:
                # the auth cell wrote the Hub token to a file under HOME, so any
                # test the model runs could read the credential straight off disk.
                # Give task subprocesses an empty home and no Hub access.
                TASK_HOME = Path("/content/qwen38_task_home")
                TASK_HOME.mkdir(parents=True, exist_ok=True)
                for cache_key in ("HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
                    TASK_ENV[cache_key] = str(TASK_HOME)
                TASK_ENV["HF_HUB_OFFLINE"] = "1"
                TASK_ENV["TRANSFORMERS_OFFLINE"] = "1"

                token_probe = subprocess.run(
                    [sys.executable, "-c", "import os, pathlib; print(any(pathlib.Path(os.path.expanduser('~')).rglob('token')))"],
                    env=TASK_ENV,
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                if token_probe.stdout.strip() != "False":
                    raise RuntimeError(
                        "Task subprocesses can still reach a credential cache: "
                        f"{token_probe.stdout.strip()} {token_probe.stderr[:500]}"
                    )
                print(f"Task subprocesses isolated to {TASK_HOME} with Hub access disabled.")

                require_free_vram(60.0)
                model, tokenizer = FastModel.from_pretrained(
                    model_name=MODEL_ID,
                    max_seq_length=MAX_SEQUENCE_LENGTH,
                    load_in_4bit=False,
                    full_finetuning=False,
                )
                assert_model_fully_resident(model)
                FastModel.for_inference(model)
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
                # Capture parameter values verbatim up to the closing tag; the
                # per-parameter trimming below keeps a patch's trailing-whitespace
                # diff lines intact instead of letting the regex eat them.
                PARAM_RE = re.compile(
                    r"<parameter=([^>\n]+)>(?:\r?\n)?(.*?)</parameter>",
                    re.DOTALL,
                )

                # Generation is decoded with special tokens visible so the
                # tool-call markup survives; the turn terminators must not leak
                # into a stored final answer.
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
                    # Written to the command's path only when it runs, after
                    # the episode: a shell command cannot read it meanwhile.
                    hidden_test_source: str | None = None

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
                    hidden_source = (
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
                        hidden_test_source=hidden_source,
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
                        # The XML parameter parser cannot distinguish a patch's
                        # final newline from tag whitespace, and git apply calls
                        # a patch whose last line lacks one "corrupt". Normalise
                        # to exactly one trailing newline.
                        patch = arguments["patch"].rstrip("\r\n") + "\n"
                        result = subprocess.run(
                            ["git", "apply", "--whitespace=nowarn", "-"],
                            cwd=root,
                            input=patch,
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
                        # The same scrubbed environment, working directory,
                        # time limit and bounded observation as run_tests;
                        # the package twin is RepoHarness._shell. Its own
                        # process group, so a timeout kills whatever the
                        # command started, not only bash; and only the head
                        # and tail of the output are held, so a command that
                        # prints without end costs bounded memory.
                        import os
                        import signal
                        import threading

                        process = subprocess.Popen(
                            ["bash", "-c", arguments["command"]],
                            cwd=root,
                            env=TASK_ENV,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        timed_out = threading.Event()

                        def kill_group():
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                return
                            timed_out.set()

                        timer = threading.Timer(120, kill_group)
                        timer.start()
                        limit = 12_000
                        head, tail, total = bytearray(), bytearray(), 0
                        try:
                            while chunk := process.stdout.read1(65_536):
                                total += len(chunk)
                                head += chunk[: max(0, limit - len(head))]
                                tail += chunk
                                del tail[:-limit]
                        finally:
                            timer.cancel()
                            process.wait()
                        if timed_out.is_set():
                            return "[timed out after 120s]"
                        output = bytes(head if total <= limit else head + tail).decode("utf-8", errors="replace")
                        observation = ("" if process.returncode == 0 else f"[exit code {process.returncode}]\n") + output
                        marker = "\n... [output trimmed] ...\n"
                        if len(observation) > limit:
                            keep = (limit - len(marker)) // 2
                            observation = observation[:keep] + marker + observation[-keep:]
                        return observation
                    return f"unknown tool: {name}"
                """
            ),
            markdown("## Run a bounded episode"),
            code(
                r"""
                generation_eos = model.generation_config.eos_token_id
                EOS_TOKEN_IDS = {
                    token_id
                    for token_id in (
                        *(generation_eos if isinstance(generation_eos, (list, tuple)) else [generation_eos]),
                        text_tokenizer_of(tokenizer).eos_token_id,
                    )
                    if token_id is not None
                }
                if not EOS_TOKEN_IDS:
                    raise RuntimeError("No end-of-turn token id is available; truncation cannot be detected.")

                # Returns the generated text plus how the turn ended, so the
                # episode loop can tell a finished answer from a cut-off one.
                def generate_turn(messages: list[dict]) -> dict:
                    rendered = render_chat(messages, add_generation_prompt=True)
                    inputs = tokenizer(
                        text=rendered,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to("cuda")
                    prompt_tokens = int(inputs["input_ids"].numel())
                    # Ten bounded observations still add up. Stop the episode
                    # here rather than let the window overflow mid-generation.
                    if prompt_tokens + MAX_NEW_TOKENS_PER_TURN > MAX_SEQUENCE_LENGTH:
                        return {
                            "text": "",
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "fault": "context_budget",
                        }
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
                    completion_tokens = int(new_ids.numel())
                    # A turn cut off at the cap carries no closing </think> or
                    # </tool_call>, so it parses as "no tool calls". Recording
                    # that prefix as the final answer would score a truncation
                    # as a completed episode.
                    stopped_on_eos = completion_tokens > 0 and int(new_ids[-1]) in EOS_TOKEN_IDS
                    return {
                        "text": tokenizer.decode(new_ids, skip_special_tokens=False),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "fault": None if stopped_on_eos else "output_truncated",
                    }

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
                        turn = generate_turn(messages)
                        prompt_tokens += turn["prompt_tokens"]
                        completion_tokens += turn["completion_tokens"]
                        if turn["fault"] is not None:
                            termination = turn["fault"]
                            break
                        reasoning, calls = parse_tool_calls(turn["text"])
                        if not calls:
                            _, final_text = split_reasoning(turn["text"])
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

                    # The verifier reaches disk only now, once the episode is
                    # over: while the model held the shell, it was not there.
                    hidden_path = Path(task.hidden_test_command[-1])
                    if task.hidden_test_source is not None:
                        hidden_path.write_text(task.hidden_test_source)
                    try:
                        hidden = subprocess.run(
                            task.hidden_test_command,
                            cwd=task.repo_path,
                            env=TASK_ENV,
                            text=True,
                            capture_output=True,
                            timeout=120,
                        )
                    finally:
                        if task.hidden_test_source is not None:
                            hidden_path.unlink(missing_ok=True)
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
                    # Truncated and budget-exhausted episodes are not model
                    # failures in the same sense as a wrong patch; keep them
                    # visible so the failure mix drives the data decision.
                    "terminations": {
                        reason: sum(1 for row in trajectories if row["termination"] == reason)
                        for reason in sorted({row["termination"] for row in trajectories})
                    },
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
                    results_repo = f"{HF_USERNAME}/qwen38-code-pilot-results"
                    require_private_repo(results_repo, "dataset")
                    api = HfApi(token=hf_token)
                    api.create_repo(results_repo, repo_type="dataset", private=True, exist_ok=True)
                    api.upload_folder(repo_id=results_repo, repo_type="dataset", folder_path=str(RESULTS_DIR))
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
                from datasets import Dataset, load_dataset
                import numpy as np
                import subprocess

                from transformers import AutoTokenizer

                MODEL_ID = "unsloth/Qwen3.8-27B"
                SOURCE_DATASET_IDS = []  # Native-schema datasets only.
                # The bootstrap corpus lives in this repository; the notebook
                # clones it, so nothing has to be uploaded or pointed at.
                REPO_URL = "https://github.com/CodeHalwell/qwen3.8-27B-code"
                REPO_DIR = Path("/content/qwen3.8-27B-code")
                if not REPO_DIR.exists():
                    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
                SOURCE_LOCAL_JSONL = str(REPO_DIR / "data" / "native_sft" / "trajectories.jsonl")
                if str(REPO_DIR / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_DIR / "src"))
                from qwen3_8_27b_code.public_sources import (
                    SOURCE_OPEN_CODE_INSTRUCT,
                    SOURCE_OPEN_CODE_REASONING,
                    SOURCE_OPEN_SWE,
                    collect_public_rows,
                )

                # Public sources, streamed from the Hub at pinned commits and
                # converted to the native schema (docs/data-strategy.md, public
                # seed sources): resolved Open-SWE-Traces trajectories cut to
                # the budget with bash mapped onto the shell tool,
                # OpenCodeInstruct answers whose unit tests all passed, and
                # OpenCodeReasoning answers, which nothing executed: they are
                # the corpus's one unverified slice, labelled as such. The
                # value is the number of native rows each source contributes;
                # 0 skips it.
                # Sized against one training epoch of about four hours on an
                # A100: the first run measured 2.5 seconds a row. The agentic
                # source takes the largest share because it is the only one
                # that teaches the tool protocol this model is being
                # specialised for.
                PUBLIC_SOURCES = {
                    SOURCE_OPEN_SWE: 3_000,
                    SOURCE_OPEN_CODE_INSTRUCT: 2_000,
                    SOURCE_OPEN_CODE_REASONING: 1_200,
                }
                # Content tokens per row; the rendered prompt and tool schema add
                # about 1,500, so this fits notebook 03's 8,192 window.
                PUBLIC_TOKEN_BUDGET = 6_000
                OUTPUT_DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                # True runs the two-row format fixture as a plumbing check and
                # refuses to publish it. The default is the real corpus.
                DEMO_MODE = False
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
                        "tool_schema_version": TOOL_SCHEMA_VERSION,
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
                        "tool_schema_version": TOOL_SCHEMA_VERSION,
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

                def read_jsonl(path):
                    with open(path) as handle:
                        return [json.loads(line) for line in handle if line.strip()]

                def count_tokens(text):
                    return len(tokenizer(text=text, add_special_tokens=False)["input_ids"])

                # Reset on every run of this cell, so a rerun with the public
                # sources off cannot publish the report of an earlier run.
                public_report = None
                if DEMO_MODE:
                    raw_dataset = Dataset.from_list(demo_rows)
                else:
                    # Plain rows from every source, then one Dataset: building it
                    # from the whole list lets Arrow infer one schema across rows
                    # whose nested tool-call arguments differ.
                    rows = []
                    if SOURCE_LOCAL_JSONL:
                        rows += read_jsonl(SOURCE_LOCAL_JSONL)
                    for dataset_id in SOURCE_DATASET_IDS:
                        rows += load_dataset(dataset_id, split="train").to_list()
                    if any(PUBLIC_SOURCES.values()):
                        public_rows, public_report = collect_public_rows(
                            PUBLIC_SOURCES, PUBLIC_TOKEN_BUDGET, count=count_tokens, token=hf_token
                        )
                        print(json.dumps(public_report, indent=2))
                        rows += public_rows
                    if not rows:
                        raise ValueError("Set SOURCE_LOCAL_JSONL, SOURCE_DATASET_IDS or PUBLIC_SOURCES to native-schema sources.")
                    # Every row gets every column first: Dataset.from_list
                    # names the columns from the first row, which is a
                    # bootstrap row with no lane, and the public rows would
                    # lose theirs and be read as agentic.
                    raw_dataset = Dataset.from_list(unify_columns(rows))

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
                    if row.get("tool_schema_version") != TOOL_SCHEMA_VERSION:
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
                    # docs/data-strategy.md defines a non-agentic lane for code
                    # and reasoning content that carries no tool supervision.
                    # Rejecting those rows here made the documented mixture
                    # impossible to build.
                    lane = row.get("lane") or "agentic"
                    if lane not in {"agentic", "non_agentic"}:
                        errors.append(f"unknown lane {lane!r}")
                    elif lane == "agentic" and not saw_tool_call:
                        errors.append("agentic trajectory has no tool call")
                    elif lane == "non_agentic" and saw_tool_call:
                        errors.append("non-agentic row supervises a tool call; label it agentic")
                    # Verified means the row says so. The one exception is
                    # an answer nothing executed (runner "none"), admitted to
                    # the non-agentic lane with that stated; never to the
                    # agentic lane, whose observations vouch for outcomes.
                    verification = row.get("verification") or {}
                    verified = verification.get("all_required_tests_pass")
                    unverified_answer = (
                        lane == "non_agentic" and verified is None and verification.get("runner") == "none"
                    )
                    if verified is not True and not unverified_answer:
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
                    text = render_chat(
                        messages,
                        add_generation_prompt=False,
                        reasoning_effort=row.get("reasoning_effort") or "medium",
                    )
                    token_count = len(tokenizer(text=text, add_special_tokens=False)["input_ids"])
                    return {
                        "messages": messages,
                        "text": text,
                        "token_count": token_count,
                        "tools": TOOLS,
                        "tool_schema_version": TOOL_SCHEMA_VERSION,
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
                import hashlib

                repo_families = sorted(set(prepared["repo_family"]))
                if len(repo_families) < 2:
                    raise ValueError(
                        "At least two repository families are required to create disjoint train and validation splits."
                    )
                # Families are wildly unequal: one Open-SWE repository is a
                # single row and a bucketed non-agentic family is dozens, so
                # holding out a tenth of the families held out a fortieth of
                # the rows, every one of them from the lane with the most
                # families. Whole families still move together, but they are
                # taken until a tenth of the rows are held out, so the
                # validation loss measures the corpus rather than one lane.
                VALIDATION_ROW_SHARE = 0.10
                family_sizes = Counter(prepared["repo_family"])
                ranked_families = sorted(
                    repo_families,
                    key=lambda family: hashlib.sha256(family.encode()).hexdigest(),
                )
                validation_target = max(1, round(len(prepared) * VALIDATION_ROW_SHARE))
                validation_families, held_out_rows = set(), 0
                for family in ranked_families:
                    if held_out_rows >= validation_target:
                        break
                    if len(validation_families) == len(repo_families) - 1:
                        break  # every corpus keeps at least one training family
                    validation_families.add(family)
                    held_out_rows += family_sizes[family]

                def split_name(repo_family: str) -> str:
                    return "validation" if repo_family in validation_families else "train"

                prepared = prepared.map(lambda row: {"split": split_name(row["repo_family"])})
                split_counts = Counter(prepared["split"])
                print(json.dumps({
                    "splits": dict(sorted(split_counts.items())),
                    "families": {"total": len(repo_families), "validation": len(validation_families)},
                    "validation_lanes": dict(sorted(Counter(
                        row.get("lane") or "agentic"
                        for row in prepared
                        if row["split"] == "validation"
                    ).items())),
                }, indent=2))

                from datasets import DatasetDict
                dataset_dict = DatasetDict({
                    split: prepared.filter(lambda row, expected=split: row["split"] == expected)
                    for split in ("train", "validation")
                })
                if not dataset_dict["train"] or not dataset_dict["validation"]:
                    raise RuntimeError(f"Split construction produced an empty partition: {split_counts}")

                PUSH_DATASET = True
                if DEMO_MODE:
                    PUSH_DATASET = False  # a smoke run never publishes the fixture
                if PUSH_DATASET:
                    # The flag says what was asked for; the ids say what is in
                    # memory. Flipping DEMO_MODE and rerunning only this cell
                    # would otherwise publish the fixture the loading cell built.
                    fixture_rows = sum(
                        str(row_id).startswith("fixture/")
                        for split in dataset_dict.values()
                        for row_id in (split["id"] if "id" in split.column_names else [])
                    )
                    if fixture_rows:
                        raise RuntimeError(
                            "Refusing to publish the synthetic format fixture as training data. "
                            "Set DEMO_MODE=False and rerun from the loading cell so the corpus is what is in memory."
                        )
                    require_private_repo(OUTPUT_DATASET_ID, "dataset")
                    dataset_dict.push_to_hub(OUTPUT_DATASET_ID, private=True)
                    print(f"Pushed {OUTPUT_DATASET_ID}")
                    if public_report:
                        # Which public commits and how many rows of each went
                        # in, kept beside the corpus so it can be rebuilt.
                        from huggingface_hub import HfApi

                        report_path = RUN_ROOT / "public_sources.json"
                        report_path.write_text(json.dumps(public_report, indent=2))
                        HfApi(token=hf_token).upload_file(
                            path_or_fileobj=str(report_path),
                            path_in_repo="public_sources.json",
                            repo_id=OUTPUT_DATASET_ID,
                            repo_type="dataset",
                            commit_message="public source commits and row counts",
                        )
                else:
                    print("Demo mode: the fixture stays local." if DEMO_MODE else "PUSH_DATASET is off; nothing published.")
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

                This stage is agentic-coding-only by policy: the corpus contains
                no general-capability replay slice, and drift on non-coding chat
                is an accepted trade rather than a gate. Specialisation comes
                from concentrated coding gradients and adapter capacity — never
                from unlearning objectives on general text, which regress the
                coding gates themselves. Acceptance still requires coding,
                tool-protocol and harness-safety non-regression against the
                frozen baseline.

                **Input:** a private dataset from notebook 02.
                **Output:** a versioned LoRA adapter and, at the end of a full run,
                the merged SFT checkpoint that notebook 04 starts from.
                """
            ),
            markdown("## Install the pinned day-zero environment"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            markdown("## Deployment tool surface"),
            code(TOOLS_CELL),
            markdown("## Run configuration"),
            code(
                r"""
                from unsloth import FastModel
                from unsloth.chat_templates import train_on_responses_only
                from datasets import Dataset, load_dataset
                from trl import SFTConfig, SFTTrainer

                MODEL_ID = "unsloth/Qwen3.8-27B"
                DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                DATASET_REVISION = "main"  # the dataset notebook 02 pushed; pin a commit to repeat a run exactly
                OUTPUT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                # Notebook 04 starts from the merged SFT weights, so its KL
                # reference is the SFT policy rather than the base, and a DPO
                # adapter trained on them is loadable elsewhere only if they are
                # on the Hub. Published at the end of training, about 55 GB, with
                # the repo history squashed so only the latest merge is stored.
                MERGED_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-merged"
                # Notebook 04's adapter is trained on the merged weights above. A
                # new SFT run replaces them, so that adapter stops being the latest
                # finished stage: its completion marker is removed when training
                # starts, and notebook 07 gates this adapter until 04 reruns.
                DPO_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-dpo-lora"
                MAX_SEQ_LENGTH = 8_192       # public rows are windowed to fit this; the 4k run measured the headroom
                # One pass over whatever the dataset holds; the trainer counts
                # the updates. A positive MAX_STEPS would override the epochs.
                # The 3,207-row run measured the second epoch: validation sat at
                # 0.240 from the end of the first to the end of the second, so
                # those hours now buy fresh rows instead of a repeat.
                NUM_TRAIN_EPOCHS = 1
                MAX_STEPS = -1
                # True trains two local smoke steps on the fixture and publishes nothing.
                DEMO_MODE = False
                RUN_TRAINING = True
                PUSH_ADAPTER = True
                PUSH_MERGED_SFT = True
                # docs/training-plan.md, Stage 1: 2e-5 is the main-run default
                # and 5e-5, 1e-4 the sweep points. 1e-4 was for the 204-row
                # bootstrap, where a rank-16 adapter had a few dozen steps to
                # move at all. The corpus notebook 02 now builds is thousands
                # of rows and hundreds of steps per epoch, so this drops to the
                # middle of the band.
                LEARNING_RATE = 5e-5
                # Every step in demo mode so the smoke exercises save and eval.
                # In a real run each eval reads the whole held-out split and each
                # save pushes the adapter to the Hub, so the cadence is set
                # against the step count: every ten steps costs more in eval and
                # upload than in training once a run is hundreds of steps long.
                EVAL_EVERY_STEPS = 1 if DEMO_MODE else 50
                SAVE_EVERY_STEPS = 1 if DEMO_MODE else 50
                # Every eval reads the whole held-out split at batch size one,
                # so its cost grows with the corpus while its job, drawing a
                # loss curve, does not. A fixed sample keeps the run's wall
                # clock tied to the training rows: at the step and forward-pass
                # costs the 3,207-row run measured, a capped run lands at the
                # same four and a half hours, where reading the whole 640-row
                # split every time would add well over an hour. The sample is
                # seeded, so the curve is comparable between runs.
                EVAL_ROW_CAP = 256

                if DEMO_MODE:
                    # A smoke run: two local steps on the fixture, nothing published.
                    MAX_STEPS, PUSH_ADAPTER, PUSH_MERGED_SFT = 2, False, False
                # The trainer creates the Hub repo when it is built, so an
                # existing public repo is caught here, before that happens.
                if PUSH_ADAPTER:
                    require_private_repo(OUTPUT_ADAPTER_ID)
                if PUSH_MERGED_SFT:
                    require_private_repo(MERGED_MODEL_ID)

                run_manifest = {
                    "stage": "sft",
                    "objective": "agentic-coding",
                    "general_retention_share": 0.0,
                    "model_id": MODEL_ID,
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "merged_model_id": MERGED_MODEL_ID,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "num_train_epochs": NUM_TRAIN_EPOCHS,
                    "max_steps": MAX_STEPS,
                    "learning_rate": LEARNING_RATE,
                    "gradient_accumulation_steps": 8,
                    "optimizer": "adamw_8bit",
                    "eval_every_steps": EVAL_EVERY_STEPS,
                    "eval_row_cap": EVAL_ROW_CAP,
                    "save_every_steps": SAVE_EVERY_STEPS,
                    "demo_mode": DEMO_MODE,
                    "tool_schema_version": TOOL_SCHEMA_VERSION,
                    "harness_version": "pilot-local-v1",
                    "run_training": RUN_TRAINING,
                }
                print(json.dumps(run_manifest, indent=2))
                """
            ),
            markdown("## Load the model and discover supported LoRA targets"),
            code(
                r"""
                require_free_vram(60.0)
                model, tokenizer = FastModel.from_pretrained(
                    model_name=MODEL_ID,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    full_finetuning=False,
                    token=hf_token,
                )
                assert_model_fully_resident(model)

                from collections import Counter

                # Qwen3.8 repeats three Gated DeltaNet layers per full-attention
                # layer, and the DeltaNet projections are named in_proj_qkv,
                # in_proj_z, in_proj_a and in_proj_b. None of those matched the
                # earlier suffix list, so 48 of the 64 layers had no adapter on
                # their attention path at all. Discover the set, then refuse to
                # train if it is not the reviewed one.
                REVIEWED_TARGET_SUFFIXES = {
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                    "gate_proj", "up_proj", "down_proj",
                }
                # The vision tower is frozen by project policy. The MTP head and
                # lm_head are excluded because nothing here trains a multi-token
                # objective and the merged checkpoint should keep its original
                # output layer.
                EXCLUDED_MODULE_MARKERS = ("visual", "vision", "image", "mtp.", "lm_head")

                def is_excluded_module(name: str) -> bool:
                    lowered = name.lower()
                    return any(marker in lowered for marker in EXCLUDED_MODULE_MARKERS)

                language_linear_names = [
                    name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear) and not is_excluded_module(name)
                ]
                discovered_suffixes = {name.rsplit(".", 1)[-1] for name in language_linear_names}
                expected_module_counts = Counter(name.rsplit(".", 1)[-1] for name in language_linear_names)
                missing_suffixes = sorted(REVIEWED_TARGET_SUFFIXES - discovered_suffixes)
                unexpected_suffixes = sorted(discovered_suffixes - REVIEWED_TARGET_SUFFIXES)
                if missing_suffixes:
                    raise RuntimeError(
                        f"Reviewed LoRA targets are absent from the loaded model: {missing_suffixes}. "
                        "The architecture or the loader changed; re-derive the target list before training."
                    )
                if unexpected_suffixes:
                    raise RuntimeError(
                        f"The model exposes language linear modules this suite has not reviewed: {unexpected_suffixes}. "
                        "Decide explicitly whether they belong in the adapter, then update REVIEWED_TARGET_SUFFIXES."
                    )
                target_modules = sorted(discovered_suffixes)
                print(json.dumps({
                    "lora_targets": target_modules,
                    "language_linear_modules": dict(sorted(expected_module_counts.items())),
                }, indent=2))

                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,  # text-only specialisation; the reviewed list below decides the rest
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

                # PEFT matches target_modules by name suffix, so the MTP head's
                # own q_proj/o_proj would otherwise receive adapters that no loss
                # ever trains. Freeze everything outside the language decoder.
                for name, parameter in model.named_parameters():
                    if is_excluded_module(name):
                        parameter.requires_grad_(False)
                trainable_outside_decoder = [
                    name for name, parameter in model.named_parameters()
                    if parameter.requires_grad and is_excluded_module(name)
                ]
                assert not trainable_outside_decoder, trainable_outside_decoder[:20]

                # Prove the adapter actually reaches every module discovery found,
                # rather than trusting that suffix matching did what was intended.
                adapted_counts = Counter(
                    name.rsplit(".", 1)[-1]
                    for name, module in model.named_modules()
                    if not is_excluded_module(name) and len(getattr(module, "lora_A", {}) or {})
                )
                if adapted_counts != expected_module_counts:
                    raise RuntimeError(
                        "LoRA coverage does not match module discovery. "
                        f"expected={dict(sorted(expected_module_counts.items()))} "
                        f"adapted={dict(sorted(adapted_counts.items()))}"
                    )
                print(json.dumps({"adapted_modules": dict(sorted(adapted_counts.items()))}, indent=2))
                model.print_trainable_parameters()
                """
            ),
            markdown("## Load native-schema data and render the exact deployment template"),
            code(
                r"""
                def demo_rows():
                    return [
                        {
                            "repo_family": "fixture/clamp",
                            "tool_schema_version": TOOL_SCHEMA_VERSION,
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
                            "tool_schema_version": TOOL_SCHEMA_VERSION,
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
                    fixture_rows = sum(
                        str(row_id).startswith("fixture/")
                        for split in ("train", "validation")
                        for row_id in (loaded[split]["id"] if "id" in loaded[split].column_names else [])
                    )
                    if fixture_rows:
                        raise ValueError(
                            f"{DATASET_ID}@{DATASET_REVISION} holds notebook 02's format fixture ({fixture_rows} rows), "
                            "not a corpus. Rerun notebook 02 with DEMO_MODE=False and push again."
                        )
                    train_raw = loaded["train"]
                    eval_raw = loaded["validation"]

                def render_row(row):
                    if row.get("tool_schema_version") != TOOL_SCHEMA_VERSION:
                        raise ValueError("Dataset schema version differs from this notebook.")
                    if row.get("tool_schema_json") != TOOL_SCHEMA_JSON:
                        raise ValueError("Dataset canonical tool fingerprint differs from this notebook.")
                    if canonical_tool_schema(row.get("tools") or []) != TOOL_SCHEMA_JSON:
                        raise ValueError("Dataset tools differ from the deployment tool surface.")
                    return {
                        "text": render_chat(
                            row["messages"],
                            add_generation_prompt=False,
                            reasoning_effort=row.get("reasoning_effort") or "medium",
                        )
                    }

                eval_rows_available = len(eval_raw)
                if EVAL_ROW_CAP and eval_rows_available > EVAL_ROW_CAP:
                    eval_raw = eval_raw.shuffle(seed=3407).select(range(EVAL_ROW_CAP))
                train_dataset = train_raw.map(render_row)
                eval_dataset = eval_raw.map(render_row)
                print(json.dumps({
                    "train_rows": len(train_dataset),
                    "eval_rows": len(eval_dataset),
                    "eval_rows_available": eval_rows_available,
                }, indent=2))
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
                    learning_rate=LEARNING_RATE,
                    warmup_ratio=0.05,
                    lr_scheduler_type="cosine",
                    num_train_epochs=NUM_TRAIN_EPOCHS,
                    max_steps=MAX_STEPS,
                    bf16=True,
                    fp16=False,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    logging_steps=1,
                    eval_strategy="steps",
                    eval_steps=EVAL_EVERY_STEPS,
                    save_strategy="steps",
                    save_steps=SAVE_EVERY_STEPS,
                    save_total_limit=2,
                    seed=3407,
                    report_to="trackio",
                    run_name="qwen38-code-sft-smoke" if USE_DEMO_DATA else "qwen38-code-sft",
                    push_to_hub=PUSH_ADAPTER,
                    hub_model_id=OUTPUT_ADAPTER_ID,
                    hub_strategy="every_save",
                    hub_private_repo=True,
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

                # Checking for one phrase from the demo fixture made this gate
                # pass only in demo mode: with real data it failed before
                # training could start. Derive the expectation from each row.
                #
                # The chat template wraps every observation in this tag, so the
                # tag inside the loss is an observation inside the loss. Looking
                # for the observation's own text instead refused real agentic
                # rows: an observation prints a path, the assistant's next
                # command names that path, and the path is then in the loss
                # while the observation is masked exactly as it should be.
                OBSERVATION_TAG = "<tool_response>"

                def masking_problems(tokenized_split, source_split) -> list[str]:
                    if len(tokenized_split) != len(source_split):
                        return [
                            f"trainer holds {len(tokenized_split)} rows but the source has "
                            f"{len(source_split)}; rows cannot be compared positionally"
                        ]
                    problems = []
                    for index, (row, source_row) in enumerate(zip(tokenized_split, source_split)):
                        supervised_ids = [
                            token_id for token_id, label in zip(row["input_ids"], row["labels"])
                            if label != -100
                        ]
                        if not supervised_ids:
                            problems.append(f"row {index}: no supervised tokens remain")
                            continue
                        supervised = tokenizer.decode(supervised_ids, skip_special_tokens=False)
                        if OBSERVATION_TAG in supervised:
                            problems.append(f"row {index}: tool observation leaked into the loss")
                        # A right-truncated row legitimately loses its tail, so
                        # only assert completeness where nothing was cut.
                        complete = len(row["input_ids"]) < MAX_SEQ_LENGTH
                        if not complete:
                            continue
                        for message in source_row["messages"]:
                            content = (message.get("content") or "").strip()
                            if message["role"] == "assistant" and content and content not in supervised:
                                problems.append(f"row {index}: assistant content is masked out of the loss")
                    return problems

                masking_failures = (
                    masking_problems(trainer.train_dataset, train_dataset)
                    + masking_problems(trainer.eval_dataset, eval_dataset)
                )
                if masking_failures:
                    raise RuntimeError(f"Assistant-only masking is wrong (first 10): {masking_failures[:10]}")

                first_supervised = tokenizer.decode(
                    [
                        token_id for token_id, label in zip(
                            trainer.train_dataset[0]["input_ids"], trainer.train_dataset[0]["labels"]
                        )
                        if label != -100
                    ],
                    skip_special_tokens=False,
                )
                print({
                    "batch_shape": tuple(labels.shape),
                    "first_trained_token": first_trainable,
                    "checked_rows": len(trainer.train_dataset) + len(trainer.eval_dataset),
                    "supervision_preview": first_supervised[:2000],
                })
                """
            ),
            markdown("## Train, resume, and publish the adapter"),
            code(
                r"""
                # checkpoint-10 sorts before checkpoint-9 lexicographically, so a
                # plain sort silently resumes from a stale step.
                def latest_checkpoint(directory):
                    numbered = [
                        (int(path.name.rsplit("-", 1)[-1]), path)
                        for path in directory.glob("checkpoint-*")
                        if path.is_dir() and path.name.rsplit("-", 1)[-1].isdigit()
                    ]
                    return max(numbered)[1] if numbered else None

                if RUN_TRAINING:
                    resume_from = latest_checkpoint(RUN_ROOT / "sft")
                    torch.cuda.reset_peak_memory_stats()
                    start_reserved_gib = torch.cuda.memory_reserved() / 1024**3
                    if PUSH_ADAPTER:
                        # An earlier run's completion marker must not survive into
                        # this run's intermediate pushes, or notebook 07 would take
                        # a half-trained adapter for a finished one. Removed here,
                        # once training is certain to start, so a dry run or a
                        # failure before this point leaves a valid adapter alone.
                        from huggingface_hub import HfApi

                        hub = HfApi(token=hf_token)
                        if hub.repo_exists(OUTPUT_ADAPTER_ID) and hub.file_exists(OUTPUT_ADAPTER_ID, "run_manifest.json"):
                            hub.delete_file(
                                "run_manifest.json", OUTPUT_ADAPTER_ID,
                                commit_message="training started: completion marker removed",
                            )
                        # The DPO adapter descends from the merged weights this run
                        # replaces, so it is no longer the latest finished stage.
                        if hub.repo_exists(DPO_ADAPTER_ID) and hub.file_exists(DPO_ADAPTER_ID, "run_manifest.json"):
                            hub.delete_file(
                                "run_manifest.json", DPO_ADAPTER_ID,
                                commit_message="SFT lineage replaced: completion marker removed",
                            )
                    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
                    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
                    run_manifest["train_runtime_seconds"] = result.metrics.get("train_runtime")
                    run_manifest["peak_reserved_gib"] = round(peak_reserved_gib, 3)
                    run_manifest["training_memory_delta_gib"] = round(peak_reserved_gib - start_reserved_gib, 3)
                    trainer.save_model(str(RUN_ROOT / "sft" / "final_adapter"))
                    tokenizer.save_pretrained(str(RUN_ROOT / "sft" / "final_adapter"))
                    (RUN_ROOT / "sft" / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
                    if PUSH_ADAPTER:
                        from huggingface_hub import HfApi

                        trainer.push_to_hub(commit_message="SFT adapter with native six-tool schema")
                        # A completion marker. The trainer created the repo before
                        # training, so its existence proves nothing; notebook 07
                        # gates an adapter only once this file is on the Hub.
                        HfApi(token=hf_token).upload_file(
                            path_or_fileobj=str(RUN_ROOT / "sft" / "run_manifest.json"),
                            path_in_repo="run_manifest.json",
                            repo_id=OUTPUT_ADAPTER_ID,
                            commit_message="run manifest: training completed",
                        )
                    if PUSH_MERGED_SFT:
                        # The merged weights notebook 04 starts from. The marker
                        # goes up after the weights, and the history is squashed
                        # so the repo holds one copy (about 55 GB), not one per
                        # run. That discards the parent of any DPO adapter
                        # trained on the previous merge, which is why this run
                        # removed that adapter's completion marker when training
                        # started: nothing downstream treats it as current, and
                        # its manifest keeps the commit it was trained on.
                        from huggingface_hub import HfApi

                        hub = HfApi(token=hf_token)
                        hub.create_repo(MERGED_MODEL_ID, repo_type="model", private=True, exist_ok=True)
                        if hub.file_exists(MERGED_MODEL_ID, "run_manifest.json"):
                            hub.delete_file(
                                "run_manifest.json", MERGED_MODEL_ID,
                                commit_message="merge started: completion marker removed",
                            )
                        model.push_to_hub_merged(MERGED_MODEL_ID, tokenizer, save_method="merged_16bit", token=hf_token)
                        hub.upload_file(
                            path_or_fileobj=str(RUN_ROOT / "sft" / "run_manifest.json"),
                            path_in_repo="run_manifest.json",
                            repo_id=MERGED_MODEL_ID,
                            commit_message="run manifest: merge completed",
                        )
                        hub.super_squash_history(MERGED_MODEL_ID, commit_message="keep only the latest merge")
                        print(f"Published the merged SFT checkpoint to {MERGED_MODEL_ID}.")
                    print(result.metrics)
                    print({
                        "peak_reserved_gib": round(peak_reserved_gib, 3),
                        "training_memory_delta_gib": round(peak_reserved_gib - start_reserved_gib, 3),
                    })
                else:
                    print("Dry run complete. Set RUN_TRAINING=True only after inspecting labels and memory.")
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

                **Input:** the *merged* SFT checkpoint notebook 03 publishes at
                the end of training, not the adapter. The model-loading cell
                explains why the distinction decides what the KL reference is.
                """
            ),
            markdown("## Install and authenticate"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            code(
                r"""
                import hashlib
                import subprocess

                from unsloth import FastModel
                from datasets import Dataset, load_dataset
                from trl import DPOConfig, DPOTrainer

                # DPO starts from the *merged* accepted SFT weights, not the SFT
                # adapter; see the model-loading cell for why the distinction
                # decides what the KL reference actually is.
                MERGED_SFT_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-merged"
                MERGED_SFT_REVISION = "main"  # pin a commit to repeat a run exactly
                # Demo mode loads the stock model instead, so the two-step smoke
                # needs nothing published; the KL reference is then the base.
                SMOKE_MODEL_ID = "unsloth/Qwen3.8-27B"
                SFT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                SFT_ADAPTER_REVISION = "main"  # recorded for lineage only
                PREFERENCE_DATASET_ID = f"{HF_USERNAME}/qwen38-code-preferences"
                PREFERENCE_DATASET_REVISION = "main"
                # The execution-derived bootstrap pairs live in this repository;
                # the notebook clones it, so nothing has to be uploaded.
                REPO_URL = "https://github.com/CodeHalwell/qwen3.8-27B-code"
                REPO_DIR = Path("/content/qwen3.8-27B-code")
                if not REPO_DIR.exists():
                    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
                PREFERENCE_LOCAL_JSONL = str(REPO_DIR / "data" / "preferences" / "pairs.jsonl")
                # Reasoning-length pairs from notebook 07 or collect_trajectories.py.
                # They teach brevity only (both sides succeeded), so they stay a
                # minority next to the execution-derived pairs: docs/thinking-budget.md
                # says no more than roughly a third of the mixture.
                LENGTH_PAIRS_LOCAL_JSONL = ""  # e.g. "/content/length_pairs.jsonl"
                MAX_LENGTH_PAIR_SHARE = 1 / 3
                OUTPUT_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-dpo-lora"
                MAX_SEQ_LENGTH = 4_096
                # Two passes over the training split; the trainer counts the
                # updates. A positive MAX_STEPS would override the epochs.
                NUM_TRAIN_EPOCHS = 2
                MAX_STEPS = -1
                # True trains two local smoke steps on synthetic pairs and publishes nothing.
                DEMO_MODE = False
                RUN_TRAINING = True
                PUSH_ADAPTER = True
                if DEMO_MODE:
                    MAX_STEPS, PUSH_ADAPTER = 2, False
                # Every step in demo mode; every ten in a real run, since each
                # save pushes the adapter and each eval scores the held-out split.
                EVAL_EVERY_STEPS = 1 if DEMO_MODE else 10
                SAVE_EVERY_STEPS = 1 if DEMO_MODE else 10
                # 5e-7 is a full-fine-tuning DPO rate. A rank-16 adapter sees
                # a small fraction of the parameters and barely moves at that
                # rate; 5e-6 is the conservative end of the usual LoRA band.
                # Sweep it alongside beta rather than treating it as settled.
                LEARNING_RATE = 5e-6
                DPO_BETA = 0.1

                # The commit the merged checkpoint resolves to, pinned here so
                # the marker check, the load and the manifest all name the same
                # weights even if notebook 03 republishes meanwhile. Notebook 03
                # keeps one merge on the Hub, so a later SFT run replaces this
                # parent and, at the same time, removes this adapter's
                # completion marker.
                MERGED_SFT_COMMIT = None
                if RUN_TRAINING and not DEMO_MODE:
                    from huggingface_hub import HfApi

                    api = HfApi(token=hf_token)
                    if not api.repo_exists(MERGED_SFT_MODEL_ID):
                        raise RuntimeError(
                            f"{MERGED_SFT_MODEL_ID} does not exist. Notebook 03 publishes it at the end of "
                            "training; run notebook 03 to completion first."
                        )
                    MERGED_SFT_COMMIT = api.repo_info(MERGED_SFT_MODEL_ID, revision=MERGED_SFT_REVISION).sha
                    # The run manifest is uploaded last, after the weights, so it
                    # proves the merge finished; a repo alone does not.
                    if not api.file_exists(MERGED_SFT_MODEL_ID, "run_manifest.json", revision=MERGED_SFT_COMMIT):
                        raise RuntimeError(
                            f"{MERGED_SFT_MODEL_ID}@{MERGED_SFT_COMMIT[:12]} has no completed merge. Notebook 03 "
                            "publishes it at the end of training; run notebook 03 to completion first."
                        )
                # The trainer creates the Hub repo when it is built, so an
                # existing public repo is caught here, before that happens.
                if PUSH_ADAPTER:
                    require_private_repo(OUTPUT_ADAPTER_ID)

                run_manifest = {
                    "stage": "dpo",
                    "objective": "agentic-coding",
                    "model_id": MERGED_SFT_MODEL_ID,
                    "model_revision": MERGED_SFT_REVISION,
                    "model_commit": MERGED_SFT_COMMIT,
                    "sft_adapter_id": SFT_ADAPTER_ID,
                    "sft_adapter_revision": SFT_ADAPTER_REVISION,
                    # Filled in by the loading cell from what is actually read:
                    # a local file with its digest, or the Hub dataset at the
                    # commit its revision resolved to.
                    "preference_sources": None,
                    "max_seq_length": MAX_SEQ_LENGTH,
                    "num_train_epochs": NUM_TRAIN_EPOCHS,
                    "max_steps": MAX_STEPS,
                    "learning_rate": LEARNING_RATE,
                    "beta": DPO_BETA,
                    "loss_type": "sigmoid",
                    "gradient_accumulation_steps": 8,
                    "optimizer": "adamw_8bit",
                    "eval_every_steps": EVAL_EVERY_STEPS,
                    "save_every_steps": SAVE_EVERY_STEPS,
                    "max_length_pair_share": MAX_LENGTH_PAIR_SHARE,
                    "demo_mode": DEMO_MODE,
                    "run_training": RUN_TRAINING,
                }
                print(json.dumps(run_manifest, indent=2))
                """
            ),
            markdown("## Load the merged SFT checkpoint"),
            code(
                r"""
                from collections import Counter

                require_free_vram(60.0)
                # `ref_model=None` does not mean "no reference": TRL uses the
                # policy with its adapters disabled. Loading the SFT *adapter*
                # here therefore made the reference the pre-SFT base model, so
                # the KL term pulled the policy back toward exactly the
                # behaviour SFT had just trained away. Loading the merged SFT
                # weights and attaching a fresh adapter makes the reference the
                # accepted SFT policy, which is what this stage should move from.
                model, tokenizer = FastModel.from_pretrained(
                    model_name=SMOKE_MODEL_ID if DEMO_MODE else MERGED_SFT_MODEL_ID,
                    revision=MERGED_SFT_COMMIT,  # None in demo mode; the pinned commit otherwise
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                    token=hf_token,
                )
                assert_model_fully_resident(model)
                # FastModel returns a processor; padding is a text-tokenizer setting.
                getattr(tokenizer, "tokenizer", tokenizer).padding_side = "left"

                # Same reviewed target set as notebook 03; a mismatch between the
                # stages would silently train a different subnetwork.
                REVIEWED_TARGET_SUFFIXES = {
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                    "gate_proj", "up_proj", "down_proj",
                }
                EXCLUDED_MODULE_MARKERS = ("visual", "vision", "image", "mtp.", "lm_head")

                def is_excluded_module(name: str) -> bool:
                    lowered = name.lower()
                    return any(marker in lowered for marker in EXCLUDED_MODULE_MARKERS)

                language_linear_names = [
                    name for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear) and not is_excluded_module(name)
                ]
                discovered_suffixes = {name.rsplit(".", 1)[-1] for name in language_linear_names}
                if discovered_suffixes != REVIEWED_TARGET_SUFFIXES:
                    raise RuntimeError(
                        "Merged checkpoint exposes a different language module set than notebook 03 reviewed: "
                        f"missing={sorted(REVIEWED_TARGET_SUFFIXES - discovered_suffixes)} "
                        f"unexpected={sorted(discovered_suffixes - REVIEWED_TARGET_SUFFIXES)}"
                    )
                expected_module_counts = Counter(name.rsplit(".", 1)[-1] for name in language_linear_names)

                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,
                    r=16,
                    target_modules=sorted(discovered_suffixes),
                    lora_alpha=32,
                    lora_dropout=0,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                )
                for name, parameter in model.named_parameters():
                    if is_excluded_module(name):
                        parameter.requires_grad_(False)
                adapted_counts = Counter(
                    name.rsplit(".", 1)[-1]
                    for name, module in model.named_modules()
                    if not is_excluded_module(name) and len(getattr(module, "lora_A", {}) or {})
                )
                if adapted_counts != expected_module_counts:
                    raise RuntimeError(
                        "LoRA coverage does not match module discovery. "
                        f"expected={dict(sorted(expected_module_counts.items()))} "
                        f"adapted={dict(sorted(adapted_counts.items()))}"
                    )

                trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                total = sum(parameter.numel() for parameter in model.parameters())
                if not trainable:
                    raise RuntimeError("The fresh DPO adapter has no trainable parameters; inspect PEFT loading before DPO.")
                print({
                    "reference": "accepted SFT weights (adapters disabled)",
                    "trainable": trainable,
                    "total": total,
                    "fraction": trainable / total,
                })
                """
            ),
            markdown("## Render prompt/chosen/rejected with the native Qwen3.8 template"),
            code(TOOLS_CELL),
            code(
                r"""
                demo_preferences = Dataset.from_list([
                    {
                        "repo_family": "fixture/bounds",
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
                        "repo_family": "fixture/parser",
                        "reasoning_effort": "low",
                        "prompt_messages": [
                            {"role": "developer", "content": "Inspect evidence before proposing a patch."},
                            {"role": "user", "content": "A parser test fails only for empty input."},
                        ],
                        # Tool-call continuation: exercises the same rendering
                        # path the execution-derived corpus pairs rely on.
                        "chosen_message": {"role": "assistant", "reasoning_content": "I should look at the failing branch before editing.", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "src/parser.py"}}}]},
                        "rejected_message": {"role": "assistant", "content": "Delete the failing test."},
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "infra_status": "ok",
                    },
                ])

                # Keep reasoning-length pairs a minority beside the correctness
                # pairs. Both sides of a length pair succeeded, so those rows
                # teach brevity and nothing about software engineering; capped at
                # max_share of the mixture, sampled deterministically, the stage
                # still learns 'right' more strongly than 'shorter'.
                import random

                def cap_length_pair_share(rows, max_share, seed=3407):
                    length = [row for row in rows if row.get("contrast_type") == "reasoning_length"]
                    correctness = [row for row in rows if row.get("contrast_type") != "reasoning_length"]
                    if length and not correctness:
                        raise ValueError(
                            "Reasoning-length pairs need execution-derived pairs beside them; "
                            "supply PREFERENCE_LOCAL_JSONL or the Hub preference dataset."
                        )
                    allowed = len(length) if max_share >= 1 else int(max_share * len(correctness) / (1 - max_share))
                    dropped = 0
                    if len(length) > allowed:
                        kept = set(random.Random(seed).sample(range(len(length)), allowed))
                        dropped = len(length) - allowed
                        length = [row for index, row in enumerate(length) if index in kept]
                    mixture = {
                        "correctness_pairs": len(correctness),
                        "length_pairs_kept": len(length),
                        "length_pairs_dropped": dropped,
                        "length_share": round(len(length) / (len(length) + len(correctness)), 3) if rows else 0.0,
                    }
                    print(mixture)
                    return correctness + length, mixture

                def read_jsonl(path):
                    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

                USE_DEMO_DATA = DEMO_MODE
                PREFERENCE_MIXTURE = None  # recorded in the run manifest for a real run
                PREFERENCE_SOURCES = []    # what was actually read, recorded in the run manifest

                def local_source(path: str, rows: list) -> dict:
                    return {
                        "kind": "local",
                        "path": path,
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        "rows": len(rows),
                    }

                if USE_DEMO_DATA:
                    raw = demo_preferences
                    print("Using synthetic plumbing preferences; this is not a capability run.")
                else:
                    if PREFERENCE_LOCAL_JSONL:
                        rows = read_jsonl(PREFERENCE_LOCAL_JSONL)
                        PREFERENCE_SOURCES.append(local_source(PREFERENCE_LOCAL_JSONL, rows))
                    else:
                        from huggingface_hub import HfApi

                        rows = load_dataset(
                            PREFERENCE_DATASET_ID,
                            split="train",
                            revision=PREFERENCE_DATASET_REVISION,
                            token=hf_token,
                        ).to_list()
                        PREFERENCE_SOURCES.append({
                            "kind": "hub",
                            "dataset_id": PREFERENCE_DATASET_ID,
                            "revision": PREFERENCE_DATASET_REVISION,
                            "resolved_revision": HfApi(token=hf_token).dataset_info(
                                PREFERENCE_DATASET_ID, revision=PREFERENCE_DATASET_REVISION
                            ).sha,
                            "rows": len(rows),
                        })
                    if LENGTH_PAIRS_LOCAL_JSONL:
                        length_rows = read_jsonl(LENGTH_PAIRS_LOCAL_JSONL)
                        PREFERENCE_SOURCES.append(local_source(LENGTH_PAIRS_LOCAL_JSONL, length_rows))
                        rows += length_rows
                    # One Dataset from plain rows: Arrow unions the struct keys of
                    # the different pair sources, and the render below strips
                    # the nulls that union inserts.
                    rows, PREFERENCE_MIXTURE = cap_length_pair_share(rows, MAX_LENGTH_PAIR_SHARE)
                    raw = Dataset.from_list(rows)

                def render_preference(row):
                    if row.get("infra_status") != "ok":
                        raise ValueError("Infrastructure failures must not become preferences.")
                    if not row["chosen_reward"] > row["rejected_reward"]:
                        raise ValueError("Chosen reward must be strictly greater than rejected reward.")
                    # Render under the effort the continuations were generated at.
                    # The template injects an instruction for low and xhigh and
                    # nothing for medium, so a low pair rendered at medium would
                    # lose the instruction its reasoning was written under.
                    effort = row.get("reasoning_effort") or "medium"
                    prompt = canonical_to_qwen(row["prompt_messages"])
                    prompt_text = text_tokenizer_of(tokenizer).apply_chat_template(
                        prompt,
                        tools=TOOLS,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                        reasoning_effort=effort,
                    )

                    def completion(message):
                        # An Arrow round trip unions struct keys across rows, so a
                        # content-only message gains null tool_calls/reasoning
                        # fields (and null argument keys) that the template would
                        # render as spurious values. Strip them before rendering.
                        message = _without_arrow_nulls(message)
                        full = text_tokenizer_of(tokenizer).apply_chat_template(
                            prompt + [message],
                            tools=TOOLS,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=True,
                            reasoning_effort=effort,
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

                # A random split puts variants of the same repository family on
                # both sides, so the evaluation half measures memorisation of the
                # family rather than transfer. Split on families, as notebook 02
                # does for SFT.
                if "repo_family" not in raw.column_names:
                    raise ValueError("Preference rows must carry repo_family so the split can be family-disjoint.")
                row_families = list(raw["repo_family"])
                families = sorted(set(row_families))
                if len(families) < 2:
                    raise ValueError(f"At least two repository families are required; found {families}.")
                evaluation_family_count = min(max(1, round(len(families) * 0.10)), len(families) - 1)
                ranked_families = sorted(families, key=lambda family: hashlib.sha256(family.encode()).hexdigest())
                evaluation_families = set(ranked_families[:evaluation_family_count])

                preferences = raw.map(render_preference, remove_columns=raw.column_names)
                evaluation_indices = [
                    index for index, family in enumerate(row_families) if family in evaluation_families
                ]
                train_indices = [
                    index for index, family in enumerate(row_families) if family not in evaluation_families
                ]
                split = {
                    "train": preferences.select(train_indices),
                    "test": preferences.select(evaluation_indices),
                }
                if not len(split["train"]) or not len(split["test"]):
                    raise RuntimeError(f"Family split produced an empty partition: {sorted(evaluation_families)}")
                print({
                    "train_rows": len(split["train"]),
                    "evaluation_rows": len(split["test"]),
                    "evaluation_families": sorted(evaluation_families),
                })
                print({key: split["train"][0][key][:1000] for key in ["prompt", "chosen", "rejected"]})
                """
            ),
            markdown("## Configure and optionally run DPO"),
            code(
                r"""
                dpo_args = DPOConfig(
                    output_dir=str(RUN_ROOT / "dpo"),
                    max_length=MAX_SEQ_LENGTH,
                    beta=DPO_BETA,
                    loss_type="sigmoid",
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=8,
                    learning_rate=LEARNING_RATE,
                    warmup_ratio=0.05,
                    lr_scheduler_type="cosine",
                    num_train_epochs=NUM_TRAIN_EPOCHS,
                    max_steps=MAX_STEPS,
                    bf16=True,
                    optim="adamw_8bit",
                    logging_steps=1,
                    eval_strategy="steps",
                    eval_steps=EVAL_EVERY_STEPS,
                    save_strategy="steps",
                    save_steps=SAVE_EVERY_STEPS,
                    save_total_limit=2,
                    precompute_ref_log_probs=True,
                    report_to="trackio",
                    run_name="qwen38-code-dpo-smoke" if USE_DEMO_DATA else "qwen38-code-dpo",
                    push_to_hub=PUSH_ADAPTER,
                    hub_model_id=OUTPUT_ADAPTER_ID,
                    hub_strategy="every_save",
                    hub_private_repo=True,
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
                    (RUN_ROOT / "dpo" / "preference_mixture.json").parent.mkdir(parents=True, exist_ok=True)
                    (RUN_ROOT / "dpo" / "preference_mixture.json").write_text(json.dumps({
                        "mixture": PREFERENCE_MIXTURE,
                        "max_length_pair_share": MAX_LENGTH_PAIR_SHARE,
                        "reasoning_effort": dict(sorted(Counter(
                            row.get("reasoning_effort") or "medium" for row in raw
                        ).items())),
                    }, indent=2))
                    if PUSH_ADAPTER:
                        # As in notebook 03: an earlier run's completion marker must
                        # not survive into this run's intermediate pushes.
                        from huggingface_hub import HfApi

                        hub = HfApi(token=hf_token)
                        if hub.repo_exists(OUTPUT_ADAPTER_ID) and hub.file_exists(OUTPUT_ADAPTER_ID, "run_manifest.json"):
                            hub.delete_file(
                                "run_manifest.json", OUTPUT_ADAPTER_ID,
                                commit_message="training started: completion marker removed",
                            )
                    result = trainer.train()
                    run_manifest["tool_schema_version"] = TOOL_SCHEMA_VERSION
                    run_manifest["preference_sources"] = PREFERENCE_SOURCES
                    run_manifest["preference_mixture"] = PREFERENCE_MIXTURE
                    run_manifest["train_runtime_seconds"] = result.metrics.get("train_runtime")
                    (RUN_ROOT / "dpo" / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
                    trainer.save_model(str(RUN_ROOT / "dpo" / "final_adapter"))
                    if PUSH_ADAPTER:
                        trainer.push_to_hub(commit_message="DPO adapter from verifier-backed preferences")
                        # Completion marker, after the final push: notebook 07 gates
                        # this adapter only once it is there.
                        hub.upload_file(
                            path_or_fileobj=str(RUN_ROOT / "dpo" / "run_manifest.json"),
                            path_in_repo="run_manifest.json",
                            repo_id=OUTPUT_ADAPTER_ID,
                            commit_message="run manifest: training completed",
                        )
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
                NeMo Gym/Harbor rollout backend is validated. Upstream Unsloth
                PR #8810 raises the TRL cap to 1.10.0; once it is released, pin
                that Unsloth revision in the install cell and re-run this
                notebook's compatibility probe before enabling training.
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

                from unsloth import FastModel
                from datasets import Dataset
                from packaging.version import Version
                from transformers import __version__ as transformers_version
                from trl import GRPOConfig, GRPOTrainer

                ACCEPTED_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                ACCEPTED_REVISION = "main"  # pin a commit to repeat a run exactly
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
                if PUSH_ADAPTER:
                    require_private_repo(OUTPUT_ADAPTER_ID)
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
            markdown(
                """
                ## Brevity term: correctness first, then fewer reasoning tokens

                docs/thinking-budget.md wants the policy to reach the same
                verified outcome with less thinking. In RL that is a
                group-relative length reward (the long2short term of Kimi
                k1.5), gated on correctness: within one GRPO group the shortest
                correct sample earns a small bonus, the longest correct sample
                a small penalty, and an incorrect sample can never gain, so
                brevity never rewards giving up. The weight stays well below
                the gap between a hidden pass and a hidden fail, which keeps
                correctness in charge of the ranking.

                This is the importable twin of `qwen3_8_27b_code.thinking.length_rewards`;
                the test suite pins the two together. Reasoning length is the
                number of completion tokens before `</think>`, which the
                trainer has for every sample in a group. Wire it as an extra
                reward per group once the `environment_factory` path is open;
                until then the fixtures below are the contract.
                """
            ),
            code(
                r"""
                def length_rewards(lengths, succeeded, weight=0.1):
                    # Group-relative brevity reward, gated on correctness; twin of
                    # qwen3_8_27b_code.thinking.length_rewards.
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

                brevity = length_rewards([100, 300, 200, 50], [True, True, False, False])
                # Correct samples: shortest up, longest down.
                assert brevity[0] > 0 > brevity[1]
                # Incorrect samples never gain, however short.
                assert brevity[2] <= 0 and brevity[3] == 0.0
                assert length_rewards([80, 80], [True, True]) == [0.0, 0.0]
                # With the environment's 0.8 hidden-pass weight, correctness
                # still decides the order across the pass/fail boundary.
                correctness = [0.8, 0.8, 0.2, 0.2]
                totals = [score + bonus for score, bonus in zip(correctness, brevity)]
                assert min(totals[:2]) > max(totals[2:])
                print("Brevity reward fixtures passed: correctness dominates, shorter correct samples rank first.")
                """
            ),
            markdown("## Load the accepted adapter and configure multi-turn GRPO"),
            code(
                r"""
                use_quantized_policy = ROLLOUT_POLICY_PRECISION == "bnb4"
                policy_manifest = {
                    "objective": "agentic-coding",
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
                    require_free_vram(24.0 if use_quantized_policy else 60.0)
                    model, tokenizer = FastModel.from_pretrained(
                        model_name=ACCEPTED_ADAPTER_ID,
                        revision=None if ACCEPTED_REVISION.startswith("REPLACE_") else ACCEPTED_REVISION,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=None if use_quantized_policy else torch.bfloat16,
                        load_in_4bit=use_quantized_policy,
                        token=hf_token,
                    )
                    assert_model_fully_resident(model)
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
                        hub_private_repo=True,
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
                from unsloth import FastModel
                from unsloth.chat_templates import train_on_responses_only
                from datasets import Dataset, load_dataset
                from trl import SFTConfig, SFTTrainer

                ACCEPTED_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                ACCEPTED_REVISION = "main"  # pin a commit to repeat an export exactly
                MERGED_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-merged"  # published by notebook 03
                QAT_OUTPUT_ID = f"{HF_USERNAME}/qwen38-27b-code-qat-int4"
                GGUF_OUTPUT_ID = f"{HF_USERNAME}/qwen38-27b-code-gguf"
                DATASET_ID = f"{HF_USERNAME}/qwen38-code-native-sft-v0"
                DATASET_REVISION = "main"
                MAX_SEQ_LENGTH = 4_096

                RUN_QAT = False
                PUSH_QAT = False
                RUN_STANDARD_GGUF_EXPORT = False
                BUILD_CALIBRATION_CORPUS = False

                # An existing public destination is found here, before the
                # QAT run or the GGUF conversion spends the GPU.
                if PUSH_QAT:
                    require_private_repo(QAT_OUTPUT_ID)
                if RUN_STANDARD_GGUF_EXPORT:
                    require_private_repo(GGUF_OUTPUT_ID)
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

                    require_free_vram(60.0)
                    qat_model, qat_tokenizer = FastModel.from_pretrained(
                        model_name=MERGED_MODEL_ID,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=torch.bfloat16,
                        load_in_4bit=False,
                        token=hf_token,
                    )
                    assert_model_fully_resident(qat_model)
                    # Same reviewed set as notebooks 03 and 04. The Gated
                    # DeltaNet projections are in_proj_qkv/z/a/b; a bare
                    # "in_proj" matches none of them, which would leave three of
                    # every four layers unadapted during the recovery pass.
                    QAT_TARGET_MODULES = [
                        "down_proj", "gate_proj",
                        "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
                        "k_proj", "o_proj", "out_proj", "q_proj", "up_proj", "v_proj",
                    ]
                    EXCLUDED_MODULE_MARKERS = ("visual", "vision", "image", "mtp.", "lm_head")
                    qat_model = FastModel.get_peft_model(
                        qat_model,
                        finetune_vision_layers=False,
                        r=16,
                        target_modules=QAT_TARGET_MODULES,
                        lora_alpha=32,
                        lora_dropout=0,
                        bias="none",
                        use_gradient_checkpointing="unsloth",
                        random_state=3407,
                        qat_scheme="int4",
                    )
                    for name, parameter in qat_model.named_parameters():
                        if any(marker in name.lower() for marker in EXCLUDED_MODULE_MARKERS):
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
                        require_private_repo(QAT_OUTPUT_ID)
                        HfApi(token=hf_token).create_repo(QAT_OUTPUT_ID, repo_type="model", private=True, exist_ok=True)
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
                    require_free_vram(60.0)
                    export_model, export_tokenizer = FastModel.from_pretrained(
                        model_name=ACCEPTED_ADAPTER_ID,
                        revision=ACCEPTED_REVISION,
                        max_seq_length=MAX_SEQ_LENGTH,
                        dtype=torch.bfloat16,
                        load_in_4bit=False,
                        token=hf_token,
                    )
                    assert_model_fully_resident(export_model)
                    from huggingface_hub import HfApi
                    require_private_repo(GGUF_OUTPUT_ID)
                    # push_to_hub_gguf creates a missing repo with its own
                    # default visibility; create it private first.
                    HfApi(token=hf_token).create_repo(GGUF_OUTPUT_ID, repo_type="model", private=True, exist_ok=True)
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


def build_07_collect_and_evaluate():
    return notebook(
        "07 · Collect trajectories and run the held-out gate",
        [
            markdown(
                """
                # 07 · Collection and the held-out gate

                Two things this suite could not do before: generate training
                trajectories from the model's own attempts, and measure whether
                a trained checkpoint is actually better than the stock one.

                **This notebook imports the shared code instead of restating
                it.** Notebooks 00-06 keep self-contained cells because each one
                needs only a handful of definitions. The episode loop, the
                collection filters and the scorecard are none of those: three
                hand-copied copies would drift, and a gate that drifts from the
                collector it grades is worse than no gate. The GPU-specific part
                — turning messages into one generated turn — is the only thing
                defined here.

                **Held-out means held out.** The evaluation families in
                `qwen3_8_27b_code.tasks` share no bug class, module or family
                name with the SFT fixtures, and the test suite enforces that.
                Never collect training data from them.
                """
            ),
            markdown("## Install the pinned day-zero environment"),
            code(INSTALL_CORE),
            markdown("After the first install, restart the runtime and rerun the notebook from the top; the install marker skips the pip work."),
            code(AUTH_AND_RUNTIME),
            markdown("## Bring in the shared harness, collector and gate"),
            code(
                r"""
                import subprocess

                REPO_URL = "https://github.com/CodeHalwell/qwen3.8-27B-code"
                REPO_REVISION = "main"  # Pin an immutable commit before a run that produces artifacts.
                REPO_DIR = Path("/content/qwen3.8-27B-code")

                if not REPO_DIR.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--branch", REPO_REVISION, REPO_URL, str(REPO_DIR)],
                        check=True,
                    )
                if str(REPO_DIR / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_DIR / "src"))

                from qwen3_8_27b_code.collection import collect, write_corpus
                from qwen3_8_27b_code.episodes import EpisodeBudget, TurnResult
                from qwen3_8_27b_code.evaluation import (
                    DEFAULT_SEEDS,
                    build_provenance,
                    compare,
                    effort_ladder,
                    evaluate,
                    gate,
                    gate_passed,
                    pairing_problems,
                    provenance_mismatches,
                    read_report,
                    write_report,
                )
                from qwen3_8_27b_code.fixtures import iter_tasks
                from qwen3_8_27b_code.long_horizon import training_tasks
                from qwen3_8_27b_code.tasks import evaluation_tasks, task_from_fixture
                from qwen3_8_27b_code.thinking import build_reasoning_length_pairs, write_length_pairs

                repo_revision = subprocess.run(
                    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                    text=True, capture_output=True, check=True,
                ).stdout.strip()
                print(json.dumps({"repo": REPO_URL, "revision": repo_revision}, indent=2))
                """
            ),
            markdown("## Run configuration"),
            code(
                r"""
                from unsloth import FastModel

                MODEL_ID = "unsloth/Qwen3.8-27B"
                # A Hub id is mutable; the baseline is loaded at this revision and
                # recorded at the commit it resolves to. Pin an immutable commit
                # for a run that produces artifacts.
                MODEL_REVISION = "main"
                ACCEPTED_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-lora"
                ACCEPTED_REVISION = "main"  # pin a commit to repeat a gate exactly
                # Gate the latest stage that finished: notebook 04's adapter when
                # it has pushed one, else notebook 03's. False gates
                # ACCEPTED_ADAPTER_ID as configured.
                DPO_ADAPTER_ID = f"{HF_USERNAME}/qwen38-27b-code-dpo-lora"
                DPO_REVISION = "main"  # a commit of the DPO repo; ACCEPTED_REVISION pins the SFT repo
                GATE_LATEST_STAGE = True
                # Every think block after the request stays in context for the rest
                # of the episode, so a thirty-call episode carries thirty turns of
                # reasoning: this is where thinking and horizon meet, and why the
                # window is sized for the long band rather than for one turn.
                MAX_SEQUENCE_LENGTH = 32_768
                # Reasoning counts against max_new_tokens, and a turn cut off inside
                # its think block returns no action. One cap sized for medium turns
                # the xhigh rung into a truncation measurement, so each effort gets
                # its own; set each above the p95 reasoning length measured at that
                # effort (docs/training-plan.md, Stage 0). `high` is deliberately
                # absent: the template aliases it to xhigh.
                MAX_NEW_TOKENS_BY_EFFORT = {"low": 2_048, "medium": 4_096, "xhigh": 8_192}
                REASONING_EFFORT = "medium"
                # docs/thinking-budget.md: a candidate may spend at most this
                # fraction more reasoning tokens per turn than the baseline.
                # Quality is gated first and separately; this only stops a
                # "better" checkpoint that got there by thinking longer.
                MAX_REASONING_GROWTH = 0.10

                # docs/evaluation.md funnel: the sentinel tier is the cheap one
                # every candidate runs. Widen only for a candidate or release
                # gate, and price it before starting.
                EVAL_VARIANTS_PER_FAMILY = 1     # 9 held-out tasks: 6 short, 2 medium, 1 long
                EVAL_ATTEMPTS = 2                # two seeds per task; a candidate must match its baseline
                # Ceilings, not targets: the long band runs to 30 tool calls and a
                # smaller budget excludes the pipeline tasks by construction.
                EPISODE_BUDGET = EpisodeBudget(tool_calls=30, wall_seconds=900.0)

                RUN_BASELINE_EVAL = None         # None: only when no baseline could be pulled from the Hub
                # The stock model at low, medium and xhigh, each with its own cap;
                # picks the deployment effort and the gate baseline
                # (docs/thinking-budget.md, lever 1).
                RUN_EFFORT_LADDER = False
                EFFORT_LADDER_TOLERANCE = 0.0    # a rung must match the best success to be eligible
                RUN_CANDIDATE_EVAL = None        # None: when the adapter notebook 03 pushes exists on the Hub
                RUN_COLLECTION = False           # expensive; read the cost note below first
                PUSH_ARTIFACTS = True

                COLLECTION_ATTEMPTS = 3
                COLLECTION_VARIANTS_PER_FAMILY = 2
                COLLECTION_SEEDS = (3407, 9176, 20261)

                REPORT_DIR = RUN_ROOT / "gate"
                REPORT_DIR.mkdir(parents=True, exist_ok=True)

                # Colab runtimes are per notebook and per session, so a baseline
                # measured today is gone before the candidate exists. The reports
                # live in a private dataset repo; the last cell pushes this
                # session's REPORT_DIR and this pulls the earlier ones into a
                # separate directory. Nothing pulled is ever mistaken for this
                # session's work: the gate takes the baseline by name from
                # either place and the candidate only from this session.
                GATE_REPORTS_REPO = f"{HF_USERNAME}/qwen38-code-gate-reports"
                PULL_REPORTS_FROM_HUB = True
                HUB_REPORT_DIR = RUN_ROOT / "gate_hub"
                # The frozen baseline the gate pairs with this session's candidate:
                # baseline.json, or a ladder rung such as ladder_medium.json.
                GATE_BASELINE_FILE = "baseline.json"
                if PULL_REPORTS_FROM_HUB:
                    from huggingface_hub import HfApi, snapshot_download

                    if HfApi(token=hf_token).repo_exists(GATE_REPORTS_REPO, repo_type="dataset"):
                        snapshot_download(
                            GATE_REPORTS_REPO,
                            repo_type="dataset",
                            local_dir=str(HUB_REPORT_DIR),
                            allow_patterns=["*.json", "*.jsonl"],
                            token=hf_token,
                        )
                        print(f"pulled earlier reports: {sorted(p.name for p in HUB_REPORT_DIR.iterdir())}")
                    else:
                        print(f"no earlier reports at {GATE_REPORTS_REPO}; starting fresh.")

                from huggingface_hub import HfApi

                def resolved_revision(repo_id: str, revision: str) -> str:
                    # The commit a branch or tag points at now, so the record
                    # names the checkpoint that was measured, not a moving ref.
                    return HfApi(token=hf_token).model_info(repo_id, revision=revision).sha

                stock_model_ref = f"{MODEL_ID}@{resolved_revision(MODEL_ID, MODEL_REVISION)}"
                print(json.dumps({"stock_model": stock_model_ref}, indent=2))

                # Recorded on every report this notebook writes, with the same
                # writer the CLI uses. The gate refuses to pair two reports
                # whose settings differ, and records a harness revision that does.
                def report_provenance(model_ref: str, reasoning_effort: str = REASONING_EFFORT) -> dict:
                    return build_provenance(
                        model=model_ref,
                        harness_revision=repo_revision,
                        reasoning_effort=reasoning_effort,
                        max_new_tokens=MAX_NEW_TOKENS_BY_EFFORT[reasoning_effort],
                        max_sequence_length=MAX_SEQUENCE_LENGTH,
                        episode_budget=EPISODE_BUDGET,
                        attempts_per_task=EVAL_ATTEMPTS,
                        seeds=DEFAULT_SEEDS[:EVAL_ATTEMPTS],
                        variants_per_family=EVAL_VARIANTS_PER_FAMILY,
                    )

                # Publishing is the last cell, but an existing public target
                # is found now, before any GPU time is spent.
                if PUSH_ARTIFACTS:
                    require_private_repo(GATE_REPORTS_REPO, "dataset")

                # What this session does follows from what already exists: a
                # baseline is measured once and reused; a candidate is gated as
                # soon as notebook 03 has pushed an adapter.
                from huggingface_hub import HfApi

                if RUN_BASELINE_EVAL is None:
                    pulled_baseline = HUB_REPORT_DIR / GATE_BASELINE_FILE
                    RUN_BASELINE_EVAL = True
                    if pulled_baseline.exists():
                        # Reuse it only if it was measured the way this session
                        # measures: otherwise the candidate would be evaluated
                        # in full and then refused at the gate.
                        mismatches = provenance_mismatches(
                            read_report(pulled_baseline).metadata, report_provenance(stock_model_ref)
                        )
                        RUN_BASELINE_EVAL = bool(mismatches)
                        for line in mismatches:
                            print(f"pulled baseline differs, measuring a new one: {line}")
                if not RUN_BASELINE_EVAL:
                    # The gate prefers this session's file over the pulled copy,
                    # so one left by an earlier run in this runtime must go.
                    (REPORT_DIR / GATE_BASELINE_FILE).unlink(missing_ok=True)
                # The candidate revision is pinned here, before the evaluation,
                # so a push during it cannot change what the report and the
                # merge name. Notebook 03 removes its run manifest when training
                # starts and uploads it after the final adapter push, so a
                # revision that carries it is a training run that finished.
                api = HfApi(token=hf_token)
                CANDIDATE_REVISION = None
                candidates = (
                    ((DPO_ADAPTER_ID, DPO_REVISION), (ACCEPTED_ADAPTER_ID, ACCEPTED_REVISION))
                    if GATE_LATEST_STAGE else ((ACCEPTED_ADAPTER_ID, ACCEPTED_REVISION),)
                )
                for adapter_id, pinned in candidates:
                    if not api.repo_exists(adapter_id):
                        continue
                    revision = resolved_revision(adapter_id, pinned)
                    if api.file_exists(adapter_id, "run_manifest.json", revision=revision):
                        ACCEPTED_ADAPTER_ID, CANDIDATE_REVISION = adapter_id, revision
                        break
                if RUN_CANDIDATE_EVAL is None:
                    RUN_CANDIDATE_EVAL = CANDIDATE_REVISION is not None
                if RUN_CANDIDATE_EVAL and CANDIDATE_REVISION is None:
                    raise RuntimeError("No finished adapter to gate; run notebook 03 to completion first.")
                print(json.dumps({"run_baseline_eval": RUN_BASELINE_EVAL, "run_candidate_eval": RUN_CANDIDATE_EVAL}, indent=2))

                # Six single-file families (short band) plus the three multi-file
                # families from long_horizon: two coupled-module (medium band) and
                # one four-stage pipeline (long band).
                evaluation_suite = evaluation_tasks(variants_per_family=EVAL_VARIANTS_PER_FAMILY)
                print(json.dumps({
                    "held_out_tasks": len(evaluation_suite),
                    "families": sorted({task.family for task in evaluation_suite}),
                    "multi_file_tasks": sum(len(task.gold_files) > 1 for task in evaluation_suite),
                    "attempts_each": EVAL_ATTEMPTS,
                }, indent=2))
                """
            ),
            markdown("## The only GPU-specific piece: messages in, one turn out"),
            code(
                r"""
                # Everything else in this notebook is shared code. A policy is a
                # callable that renders the history, generates one assistant
                # turn, and reports whether generation finished or was cut off.
                def build_policy_factory(model, tokenizer, reasoning_effort=REASONING_EFFORT):
                    max_new_tokens = MAX_NEW_TOKENS_BY_EFFORT[reasoning_effort]
                    text_tokenizer = text_tokenizer_of(tokenizer)
                    generation_eos = model.generation_config.eos_token_id
                    eos_token_ids = {
                        token_id
                        for token_id in (
                            *(generation_eos if isinstance(generation_eos, (list, tuple)) else [generation_eos]),
                            text_tokenizer.eos_token_id,
                        )
                        if token_id is not None
                    }
                    if not eos_token_ids:
                        raise RuntimeError("No end-of-turn token id is available; truncation cannot be detected.")
                    # The thinking budget is measured in tokens generated before
                    # the think block closes. Count them from the ids, not from
                    # decoded text, so the number is exact.
                    think_end_id = text_tokenizer.convert_tokens_to_ids("</think>")
                    if think_end_id is None or think_end_id == text_tokenizer.unk_token_id:
                        raise RuntimeError("The tokenizer has no </think> token; reasoning tokens cannot be counted.")

                    def policy_factory(task, seed):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)

                        def policy(messages):
                            rendered = render_chat(
                                messages,
                                add_generation_prompt=True,
                                reasoning_effort=reasoning_effort,
                            )
                            inputs = tokenizer(
                                text=rendered, return_tensors="pt", add_special_tokens=False
                            ).to("cuda")
                            prompt_tokens = int(inputs["input_ids"].numel())
                            if prompt_tokens + max_new_tokens > MAX_SEQUENCE_LENGTH:
                                return TurnResult(
                                    text="", prompt_tokens=prompt_tokens, fault="context_budget"
                                )
                            with torch.inference_mode():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=max_new_tokens,
                                    temperature=1.0,
                                    top_p=0.95,
                                    top_k=20,
                                    do_sample=True,
                                    use_cache=True,
                                )
                            new_ids = outputs[0, inputs["input_ids"].shape[1]:]
                            completion_tokens = int(new_ids.numel())
                            stopped_on_eos = completion_tokens > 0 and int(new_ids[-1]) in eos_token_ids
                            text = tokenizer.decode(new_ids, skip_special_tokens=False)
                            closes = (new_ids == think_end_id).nonzero()
                            if len(closes):
                                # Everything up to and including </think> was reasoning.
                                reasoning_tokens = int(closes[0].item()) + 1
                            elif rendered.rstrip().endswith("<think>") or "<think>" in text:
                                # The block never closed: the whole turn was
                                # reasoning, which is the overrun the budget
                                # has to see rather than hide.
                                reasoning_tokens = completion_tokens
                            else:
                                reasoning_tokens = 0
                            return TurnResult(
                                text=text,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                fault=None if stopped_on_eos else "output_truncated",
                                reasoning_tokens=reasoning_tokens,
                            )

                        return policy

                    return policy_factory
                """
            ),
            code(TOOLS_CELL),
            markdown("## Baseline: the stock model on the held-out suite"),
            code(
                r"""
                baseline_report_path = REPORT_DIR / "baseline.json"

                if RUN_BASELINE_EVAL:
                    # A baseline from an earlier run of this cell must not
                    # survive an evaluation that fails before it writes.
                    baseline_report_path.unlink(missing_ok=True)
                    require_free_vram(60.0)
                    model, tokenizer = FastModel.from_pretrained(
                        model_name=MODEL_ID,
                        revision=MODEL_REVISION,
                        max_seq_length=MAX_SEQUENCE_LENGTH,
                        load_in_4bit=False,
                        full_finetuning=False,
                        token=hf_token,
                    )
                    assert_model_fully_resident(model)
                    FastModel.for_inference(model)

                    baseline = evaluate(
                        evaluation_suite,
                        build_policy_factory(model, tokenizer),
                        label="upstream-bf16",
                        attempts_per_task=EVAL_ATTEMPTS,
                        budget=EPISODE_BUDGET,
                    )
                    baseline.metadata = report_provenance(stock_model_ref)
                    write_report(baseline, baseline_report_path)
                    print(json.dumps(baseline.scorecard(), indent=2))
                    print(f"wrote {baseline_report_path}")
                else:
                    print("Baseline evaluation is off. It is the comparison point for every later claim.")
                """
            ),
            markdown(
                """
                ## Effort ladder: the stock model at low, medium and xhigh

                Lever 1 of docs/thinking-budget.md, and it costs no training.
                Each rung runs with its own per-turn cap, so the table
                measures effort rather than truncation. The recommendation
                is the rung that thinks least among those that keep the best
                success; set `REASONING_EFFORT` to it and use its report as
                the gate baseline. Read `success_by_task_horizon` before the
                aggregate: a rung that holds the short tasks and loses the
                pipeline is not a cheaper rung.
                """
            ),
            code(
                r"""
                if RUN_EFFORT_LADDER:
                    if "model" not in globals():
                        raise RuntimeError("Load the stock model in the baseline cell first.")
                    ladder_reports = {}
                    for effort in ("low", "medium", "xhigh"):
                        ladder_reports[effort] = evaluate(
                            evaluation_suite,
                            build_policy_factory(model, tokenizer, reasoning_effort=effort),
                            label=f"upstream-bf16-{effort}",
                            attempts_per_task=EVAL_ATTEMPTS,
                            budget=EPISODE_BUDGET,
                        )
                        ladder_reports[effort].metadata = report_provenance(stock_model_ref, reasoning_effort=effort)
                        write_report(ladder_reports[effort], REPORT_DIR / f"ladder_{effort}.json")
                    ladder = effort_ladder(ladder_reports, success_tolerance=EFFORT_LADDER_TOLERANCE)
                    (REPORT_DIR / "effort_ladder.json").write_text(json.dumps(ladder, indent=2))
                    print(json.dumps(ladder, indent=2))
                    if ladder["recommended"] is None:
                        print(f"{ladder['note']}. Do not set REASONING_EFFORT from this ladder.")
                    else:
                        print(
                            f"Recommended deployment effort: {ladder['recommended']}. Set REASONING_EFFORT to it "
                            f"and use ladder_{ladder['recommended']}.json as the frozen baseline for the gate."
                        )
                else:
                    print("Effort ladder is off. Run it once on the stock model before choosing REASONING_EFFORT.")
                """
            ),
            markdown(
                """
                ## Candidate: the accepted adapter on the same frozen suite

                Release the baseline model first. Two 27B checkpoints do not
                coexist on one card, and a partially offloaded second load
                crawls or dies mid-episode.
                """
            ),
            code(
                r"""
                candidate_report_path = REPORT_DIR / "candidate.json"
                # Set only when this cell writes the report; the gate reads it
                # rather than inferring from the flag and a file that may be
                # left over from an earlier run in the same runtime.
                candidate_written = False

                if RUN_CANDIDATE_EVAL:
                    candidate_report_path.unlink(missing_ok=True)
                    release_stale_gpu_state()
                    require_free_vram(60.0)
                    model, tokenizer = FastModel.from_pretrained(
                        model_name=ACCEPTED_ADAPTER_ID,
                        revision=CANDIDATE_REVISION,
                        max_seq_length=MAX_SEQUENCE_LENGTH,
                        load_in_4bit=False,
                        token=hf_token,
                    )
                    assert_model_fully_resident(model)
                    FastModel.for_inference(model)

                    candidate = evaluate(
                        evaluation_suite,
                        build_policy_factory(model, tokenizer),
                        label="sft-lora-candidate",
                        attempts_per_task=EVAL_ATTEMPTS,
                        budget=EPISODE_BUDGET,
                    )
                    candidate_model_ref = f"{ACCEPTED_ADAPTER_ID}@{CANDIDATE_REVISION}"
                    candidate.metadata = report_provenance(candidate_model_ref)
                    write_report(candidate, candidate_report_path)
                    candidate_written = True
                    print(json.dumps(candidate.scorecard(), indent=2))
                else:
                    print("Candidate evaluation is off. Turn it on once an adapter revision is accepted.")
                """
            ),
            markdown("## Apply the gate"),
            code(
                r"""
                # The candidate is always the one measured in this session. The
                # baseline is GATE_BASELINE_FILE from this session if it wrote
                # one, else the pulled copy. A pulled candidate is never used:
                # a fresh baseline gated against a stale candidate would
                # republish a verdict nobody asked for.
                comparison_path = REPORT_DIR / "comparison.json"
                # A verdict from an earlier run of this cell in the same runtime
                # must not survive a gate that is skipped or refused now, or the
                # persist cell would push it as if it were this run's.
                comparison_path.unlink(missing_ok=True)
                # The same goes for an acceptance: it is written only by a gate
                # that passed in this run.
                accepted_path = REPORT_DIR / "accepted.json"
                accepted_path.unlink(missing_ok=True)
                # The baseline is, in order: the named file this session wrote (a
                # ladder rung), the baseline this session measured, then the pulled
                # copy of the named file. A fresh measurement always outranks the
                # pulled copy it was measured to replace.
                baseline_candidates = [REPORT_DIR / GATE_BASELINE_FILE]
                if RUN_BASELINE_EVAL:
                    baseline_candidates.append(baseline_report_path)
                baseline_candidates.append(HUB_REPORT_DIR / GATE_BASELINE_FILE)
                baseline_for_gate = next((path for path in baseline_candidates if path.exists()), None)
                if not (RUN_CANDIDATE_EVAL and globals().get("candidate_written") and candidate_report_path.exists()):
                    print("The gate needs a candidate measured in this session (RUN_CANDIDATE_EVAL).")
                elif baseline_for_gate is None:
                    print(f"No {GATE_BASELINE_FILE} in this session or on {GATE_REPORTS_REPO}; measure a baseline first.")
                elif (blocking := pairing_problems(
                    baseline_report := read_report(baseline_for_gate),
                    candidate_report := read_report(candidate_report_path),
                ))[0]:
                    for problem in blocking[0]:
                        print(f"  [REFUSED] {problem}")
                    print("GATE NOT RUN: the two reports were not measured the same way.")
                else:
                    advisory = blocking[1]
                    for note in advisory:
                        print(f"  [NOTE] {note}")
                    comparison = compare(baseline_report, candidate_report)
                    comparison["provenance"] = {
                        "baseline": {**baseline_report.metadata, "path": str(baseline_for_gate)},
                        "candidate": candidate_report.metadata,
                        "notes": advisory,
                    }
                    checks = gate(comparison, max_reasoning_growth=MAX_REASONING_GROWTH)
                    comparison["gate"] = [
                        {"name": check.name, "passed": check.passed, "detail": check.detail}
                        for check in checks
                    ]
                    comparison["gate_passed"] = gate_passed(checks)
                    comparison_path.write_text(json.dumps(comparison, indent=2))

                    print(json.dumps(comparison["deltas"], indent=2))
                    # Reasoning tokens per turn, share of generation spent thinking,
                    # and success by horizon band, baseline against candidate.
                    print(json.dumps(comparison["thinking"], indent=2))
                    for check in checks:
                        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
                    task_level = comparison["task_level"]
                    # A suite this small reports paired outcomes; it cannot
                    # support a percentage-point significance claim.
                    print(
                        f"{task_level['wins']} improved, {task_level['losses']} regressed, "
                        f"{task_level['ties']} unchanged, of {task_level['tasks']} tasks."
                    )
                    print("GATE PASSED" if comparison["gate_passed"] else "GATE FAILED")

                    if comparison["gate_passed"]:
                        # The acceptance record, pushed with the reports: which
                        # adapter, at which commit, passed against which baseline.
                        accepted_path.write_text(json.dumps({
                            "adapter": candidate_model_ref,
                            "baseline": baseline_for_gate.name,
                            "harness_revision": repo_revision,
                            "provenance": candidate_report.metadata,
                        }, indent=2))
                """
            ),
            markdown(
                """
                ## Collect training trajectories by rejection sampling

                This is the route off the scripted bootstrap corpus. The model
                attempts each training task several times, every attempt is
                graded from outside its workspace, and only verified attempts
                become rows — carrying the model's own reasoning at the effort
                it ran at, which is what the scripted corpus cannot supply.

                Cost first: attempts × tasks × mean episode seconds. Measure one
                task before enabling the full sweep, and use the acceptance rate
                in the report to decide whether more attempts or easier tasks
                are the better next move.
                """
            ),
            code(
                r"""
                if RUN_COLLECTION:
                    if "model" not in globals():
                        raise RuntimeError("Load a model in one of the cells above before collecting.")
                    # Single-file fixtures plus the multi-file training families,
                    # so the corpus carries the medium-horizon shape as well.
                    collection_tasks = [
                        task_from_fixture(fixture)
                        for fixture in iter_tasks(COLLECTION_VARIANTS_PER_FAMILY)
                    ] + training_tasks(COLLECTION_VARIANTS_PER_FAMILY)
                    result = collect(
                        collection_tasks,
                        build_policy_factory(model, tokenizer),
                        attempts_per_task=COLLECTION_ATTEMPTS,
                        seeds=COLLECTION_SEEDS,
                        budget=EPISODE_BUDGET,
                        reasoning_effort=REASONING_EFFORT,
                        max_rows_per_task=2,
                        # Of the attempts that verified, keep the ones that
                        # thought least: the model's own shortest working path.
                        selection="shortest_reasoning",
                    )
                    corpus_path = REPORT_DIR / "collected_trajectories.jsonl"
                    report = write_corpus(result, corpus_path, REPORT_DIR / "collection_report.json")
                    print(json.dumps(report, indent=2))
                    print(f"wrote {corpus_path}; feed it to notebook 02 as SOURCE_LOCAL_JSONL.")

                    # Verified attempts that thought more than another verified
                    # attempt at the same action become brevity preferences.
                    length_pairs = build_reasoning_length_pairs(result.attempts)
                    pairs_report = write_length_pairs(
                        length_pairs,
                        REPORT_DIR / "length_pairs.jsonl",
                        REPORT_DIR / "length_pairs_report.json",
                    )
                    print(json.dumps(pairs_report, indent=2))
                    print(
                        f"wrote {len(length_pairs)} reasoning-length pairs; feed length_pairs.jsonl to "
                        "notebook 04 as LENGTH_PAIRS_LOCAL_JSONL next to the execution-derived pairs."
                    )
                else:
                    print("Collection is off. Enable it once the baseline scorecard shows the failure mix.")
                """
            ),
            markdown(
                """
                ## Persist the reports

                Everything this session wrote under `REPORT_DIR` — baseline,
                candidate, comparison, ladder rungs, any collected corpus and
                its length pairs — goes to one private dataset repo, tagged
                with the harness revision that produced it. Pulled copies live
                in `HUB_REPORT_DIR` and are not pushed back. The configuration
                cell pulls the same repo at the start of the next session.
                """
            ),
            code(
                r"""
                if PUSH_ARTIFACTS:
                    from huggingface_hub import HfApi

                    require_private_repo(GATE_REPORTS_REPO, "dataset")
                    api = HfApi(token=hf_token)
                    api.create_repo(GATE_REPORTS_REPO, repo_type="dataset", private=True, exist_ok=True)
                    commit = api.upload_folder(
                        repo_id=GATE_REPORTS_REPO,
                        repo_type="dataset",
                        folder_path=str(REPORT_DIR),
                        allow_patterns=["*.json", "*.jsonl"],
                        # An earlier verdict or acceptance on the Hub must not
                        # outlive a gate that was skipped, refused or failed here:
                        # the remote file is deleted unless this session's copy
                        # replaces it (a file uploaded in the same commit is kept).
                        delete_patterns=["comparison.json", "accepted.json"],
                        commit_message=f"gate reports from {repo_revision[:12]}",
                    )
                    print(f"pushed {sorted(p.name for p in REPORT_DIR.iterdir())} to {GATE_REPORTS_REPO}")
                    print(commit)
                else:
                    print("PUSH_ARTIFACTS is off; the reports stay in this runtime and vanish with it.")
                """
            ),
            markdown(
                """
                ## What the numbers mean

                Read the acceptance rate and the rejection breakdown before the
                row count. A corpus of 500 rows whose rejections are dominated
                by `completed_without_verification` is telling you the policy
                does not verify, and training on the survivors will not fix that.

                The difficulty bands come from docs/data-strategy.md: tasks in
                the trivial band are protocol smoke tests, the learnable band is
                the useful curriculum, and frontier tasks are for later.

                The `thinking` section of the report splits reasoning per turn
                by outcome. If the attempts that failed thought far more than
                the ones that verified, the model is spending tokens on tasks
                it cannot do and a tighter budget costs little; if the reverse,
                brevity is being bought with correctness and the thinking gate
                in the comparison above is the thing to watch.

                Feed the collected JSONL to notebook 02, which remains the
                publisher that validates, splits and pushes the dataset that
                notebooks 03 and 06 consume. The reasoning-length pairs go to
                notebook 04 as `LENGTH_PAIRS_LOCAL_JSONL`, where they are
                capped to a minority of the mixture (see docs/thinking-budget.md).

                Two long-horizon numbers are on every scorecard now:
                `peak_prompt_tokens_max`, the largest context any turn needed,
                and `context_budget_rate`, how often an episode ran out of
                window. When either climbs towards `MAX_SEQUENCE_LENGTH` the
                next lever is less thinking per turn, then observation
                compaction, in that order.
                """
            ),
        ],
    )

TEACHER_RUNTIME = r"""
import json
import os
from pathlib import Path

from huggingface_hub import login, whoami

try:
    from google.colab import userdata
except ImportError:
    userdata = None

def secret(name: str):
    value = userdata.get(name) if userdata is not None else None
    return value or os.getenv(name)

hf_token = secret("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Add HF_TOKEN to Colab Secrets: it pushes artifacts and authenticates the Hugging Face router.")
login(token=hf_token, add_to_git_credential=False)
HF_USERNAME = whoami()["name"]
os.environ["HF_TOKEN"] = hf_token

def require_private_repo(repo_id: str, repo_type: str = "model") -> None:
    # Refuse to publish into a Hub repo that already exists and is public.
    # private=True on create_repo, push_to_hub and hub_private_repo applies
    # only when the repo is created; an existing public repo stays public
    # and every later push lands in the open.
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    if not api.repo_exists(repo_id, repo_type=repo_type):
        return
    if not api.repo_info(repo_id, repo_type=repo_type).private:
        raise RuntimeError(
            f"{repo_type} repo {repo_id} exists and is public. Make it private first with "
            f"HfApi(token=hf_token).update_repo_settings(repo_id={repo_id!r}, repo_type={repo_type!r}, "
            "private=True), or publish under a new id."
        )

# Vendor endpoints read their own key. Copy each one that exists in Secrets
# into the environment name its preset expects; the task harness strips
# anything named like a key from the environment repository code runs under.
for key_env in ("MOONSHOT_API_KEY", "ZAI_API_KEY", "TEACHER_API_KEY"):
    value = secret(key_env)
    if value:
        os.environ[key_env] = value

RUN_ROOT = Path("/content/qwen38_runs")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Authenticated as {HF_USERNAME}")
"""


def build_08_distil():
    return notebook(
        "08 · Distil from a larger open model",
        [
            markdown(
                """
                # 08 · Distil from a larger open model

                A larger open model (a bigger Qwen3.8, Kimi K3, GLM 5.3 or 5.3
                Flash) acts as the *teacher*: it attempts the training tasks
                through the exact six-tool harness, every attempt is graded from
                outside its workspace, and only verified attempts become student
                data. Nothing is translated or fabricated; this is the
                "Regenerable" lane of docs/data-strategy.md applied to a model.

                Three artifacts come out. Verified trajectories for notebook 02,
                with the teacher's own reasoning at the effort you label them
                with. Reasoning-length pairs from teacher attempts that verified
                but thought more than another. And outcome pairs (teacher
                verified, student did not, from the same state) for notebook 04
                when a student attempts file is supplied.

                **No GPU is needed.** The teacher runs behind an OpenAI-compatible
                endpoint; this notebook only drives the CPU harness and the
                endpoint. Read docs/distillation.md first: it covers what this
                buys over logit distillation, the reasoning-visibility
                requirement, how to label effort, and the vendor terms to check
                before training on API output.
                """
            ),
            markdown("## Authenticate"),
            code(TEACHER_RUNTIME),
            markdown("## Bring in the shared harness, collector and teacher adapter"),
            code(
                r"""
                import subprocess
                import sys

                REPO_URL = "https://github.com/CodeHalwell/qwen3.8-27B-code"
                REPO_REVISION = "main"  # Pin an immutable commit before a run that produces artifacts.
                REPO_DIR = Path("/content/qwen3.8-27B-code")

                if not REPO_DIR.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "--branch", REPO_REVISION, REPO_URL, str(REPO_DIR)],
                        check=True,
                    )
                if str(REPO_DIR / "src") not in sys.path:
                    sys.path.insert(0, str(REPO_DIR / "src"))

                from qwen3_8_27b_code.collection import collect, read_attempts, write_attempts, write_corpus
                from qwen3_8_27b_code.distillation import build_outcome_pairs, write_outcome_pairs
                from qwen3_8_27b_code.episodes import EpisodeBudget
                from qwen3_8_27b_code.fixtures import iter_tasks
                from qwen3_8_27b_code.long_horizon import training_tasks
                from qwen3_8_27b_code.tasks import task_from_fixture
                from qwen3_8_27b_code.teachers import PRESETS, TeacherConfig, probe_teacher, teacher_policy_factory
                from qwen3_8_27b_code.thinking import build_reasoning_length_pairs, write_length_pairs

                repo_revision = subprocess.run(
                    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
                    text=True, capture_output=True, check=True,
                ).stdout.strip()
                print(json.dumps({"repo": REPO_URL, "revision": repo_revision, "presets": sorted(PRESETS)}, indent=2))
                """
            ),
            markdown("## Choose the teacher"),
            code(
                r"""
                TEACHER_PRESET = "hf-inference-providers"   # or "moonshot", "zai", "vllm-local", "llama-cpp-local"
                TEACHER_MODEL = "REPLACE_WITH_TEACHER_MODEL_ID"  # the id as the endpoint names it
                TEACHER_BASE_URL = None                     # overrides the preset when set
                TEACHER_API_KEY_ENV = None                  # overrides the preset's key variable when set
                SEND_REASONING_EFFORT = None                # e.g. "medium" for Qwen3.8 endpoints; None sends nothing
                EXTRA_BODY = {}                             # vendor switches, e.g. a thinking flag
                REQUIRE_REASONING = True                    # refuse turns whose reasoning the endpoint hides
                MAX_TOKENS_PER_TURN = 4_096

                # The label stored on every row. It tells the student what this
                # much reasoning is worth; compare the teacher's reasoning per
                # turn (in the collection report) with the student's effort
                # ladder before settling on it. See docs/distillation.md.
                EFFORT_LABEL = "medium"

                RUN_PROBE = True
                RUN_TEACHER_COLLECTION = False              # cost first: attempts x tasks x teacher price
                COLLECTION_ATTEMPTS = 3
                COLLECTION_VARIANTS_PER_FAMILY = 2
                COLLECTION_SEEDS = (3407, 9176, 20261)
                EPISODE_BUDGET = EpisodeBudget(tool_calls=30, wall_seconds=900.0)  # long band: up to 30 calls
                STUDENT_ATTEMPTS_JSONL = ""                 # attempts.jsonl from notebook 07 / collect_trajectories.py
                PUSH_ARTIFACTS = False

                if TEACHER_MODEL.startswith("REPLACE_"):
                    raise RuntimeError("Set TEACHER_MODEL to the teacher's model id before continuing.")
                overrides = {
                    "reasoning_effort": SEND_REASONING_EFFORT,
                    "extra_body": EXTRA_BODY,
                    "require_reasoning": REQUIRE_REASONING,
                    "max_tokens": MAX_TOKENS_PER_TURN,
                }
                if TEACHER_BASE_URL:
                    teacher = TeacherConfig(
                        model=TEACHER_MODEL, base_url=TEACHER_BASE_URL, api_key_env=TEACHER_API_KEY_ENV, **overrides
                    )
                else:
                    teacher = TeacherConfig.from_preset(TEACHER_PRESET, TEACHER_MODEL, **overrides)
                    if TEACHER_API_KEY_ENV:
                        teacher = TeacherConfig(**{**teacher.__dict__, "api_key_env": TEACHER_API_KEY_ENV})
                TEACHER_DIR = RUN_ROOT / "teacher" / TEACHER_MODEL.replace("/", "-")
                TEACHER_DIR.mkdir(parents=True, exist_ok=True)
                TEACHER_REPO = f"{HF_USERNAME}/qwen38-code-teacher-{TEACHER_MODEL.replace('/', '-')}"
                # An existing public destination is found here, before any
                # teacher calls are paid for.
                if PUSH_ARTIFACTS:
                    require_private_repo(TEACHER_REPO, "dataset")
                print(json.dumps({"teacher": teacher.label, "endpoint": teacher.base_url, "out": str(TEACHER_DIR)}, indent=2))
                """
            ),
            markdown(
                """
                ## Probe: can this endpoint be a teacher?

                One round trip with the deployment tools. A teacher is usable
                when it answers with a native tool call and exposes its
                reasoning. If reasoning is hidden, do not switch
                `REQUIRE_REASONING` off to get past this cell: rows with empty
                think blocks teach the student to skip thinking at the labelled
                effort. Find an endpoint that returns `reasoning_content`.
                """
            ),
            code(
                r"""
                if RUN_PROBE:
                    probe = probe_teacher(teacher)
                    print(json.dumps(probe, indent=2))
                    if not probe["usable"]:
                        raise RuntimeError("This endpoint cannot be a teacher as configured; see the probe report.")
                else:
                    print("Probe skipped.")
                """
            ),
            markdown(
                """
                ## Collect verified teacher trajectories

                Same collector as notebook 07: single-file fixtures plus the
                multi-file training families, several attempts per task, only
                verified attempts kept, and of those the ones that reasoned
                least. Every attempt is persisted for the outcome pairs below.
                """
            ),
            code(
                r"""
                if RUN_TEACHER_COLLECTION:
                    collection_tasks = [
                        task_from_fixture(fixture)
                        for fixture in iter_tasks(COLLECTION_VARIANTS_PER_FAMILY)
                    ] + training_tasks(COLLECTION_VARIANTS_PER_FAMILY)
                    result = collect(
                        collection_tasks,
                        teacher_policy_factory(teacher),
                        attempts_per_task=COLLECTION_ATTEMPTS,
                        seeds=COLLECTION_SEEDS,
                        budget=EPISODE_BUDGET,
                        reasoning_effort=EFFORT_LABEL,
                        max_rows_per_task=2,
                        selection="shortest_reasoning",
                        policy_label=teacher.label,
                        source=teacher.label,
                        provenance={
                            "teacher": {
                                "model": teacher.model,
                                "endpoint": teacher.base_url,
                                "reasoning_effort_parameter": teacher.reasoning_effort,
                                "extra_body": teacher.extra_body,
                            }
                        },
                    )
                    report = write_corpus(result, TEACHER_DIR / "trajectories.jsonl", TEACHER_DIR / "quality_report.json")
                    print(json.dumps(report, indent=2))
                    print(f"persisted {write_attempts(result, TEACHER_DIR / 'attempts.jsonl')} attempts")

                    length_pairs = build_reasoning_length_pairs(result.attempts)
                    print(json.dumps(write_length_pairs(
                        length_pairs, TEACHER_DIR / "length_pairs.jsonl", TEACHER_DIR / "length_pairs_report.json"
                    ), indent=2))
                else:
                    print("Teacher collection is off. Price one task first, then enable it.")
                """
            ),
            markdown(
                """
                ## Outcome pairs: teacher against the student

                Where a teacher attempt verified and a student attempt at the
                same task did not, the pair prefers the teacher's continuation
                at the first divergent action. Supply the student's
                `attempts.jsonl` (notebook 07 or `scripts/collect_trajectories.py`)
                to get teacher-versus-student pairs; without it the pairs are
                teacher-versus-teacher across seeds.
                """
            ),
            code(
                r"""
                if RUN_TEACHER_COLLECTION:
                    attempts = list(result.attempts)
                    if STUDENT_ATTEMPTS_JSONL:
                        attempts += read_attempts(Path(STUDENT_ATTEMPTS_JSONL))
                    outcome_pairs = build_outcome_pairs(attempts)
                    outcome_report = write_outcome_pairs(
                        outcome_pairs, TEACHER_DIR / "outcome_pairs.jsonl", TEACHER_DIR / "outcome_pairs_report.json"
                    )
                    print(json.dumps(outcome_report, indent=2))

                    if PUSH_ARTIFACTS:
                        from huggingface_hub import HfApi

                        require_private_repo(TEACHER_REPO, "dataset")
                        api = HfApi(token=hf_token)
                        api.create_repo(TEACHER_REPO, repo_type="dataset", private=True, exist_ok=True)
                        api.upload_folder(repo_id=TEACHER_REPO, repo_type="dataset", folder_path=str(TEACHER_DIR))
                else:
                    print("No teacher attempts in this session; nothing to pair.")
                """
            ),
            markdown(
                """
                ## What next

                Feed `trajectories.jsonl` to notebook 02 as `SOURCE_LOCAL_JSONL`,
                and the two pairs files to notebook 04 as `PREFERENCE_LOCAL_JSONL`
                alongside the execution-derived pairs, keeping the length pairs a
                minority. Read the collection report's `thinking` section before
                either: if the teacher thinks several times longer per turn than
                the student at the same label, the rows will move the student's
                budget up, and the thinking gate in notebook 07 will say so.

                Check the teacher's weight licence and, for a vendor API, its
                terms on training other models with its output, before any
                artifact from this notebook is published.
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
        NOTEBOOKS / "07_collect_and_evaluate.ipynb": build_07_collect_and_evaluate(),
        NOTEBOOKS / "08_distil_from_teacher.ipynb": build_08_distil(),
        ROOT / "src" / "qwen3_8_27b_code" / "train.ipynb": build_legacy_pointer(),
    }
    for path, nb in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        validate_notebook(nb, path)
        nbf.write(nb, path)
        print(f"wrote {path.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
