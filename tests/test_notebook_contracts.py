"""Regression tests for contracts embedded in the generated Colab notebooks."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import unquote

from datasets import Dataset
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "build_notebooks.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("build_notebooks", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code_cell_containing(notebook, marker: str) -> str:
    return next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and marker in cell.source
    )


class FakeTokenizer:
    """Minimal tokenizer surface for data-contract cells that do not need a model."""

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "<tool_call>fixture</tool_call>\n<tool_response>ok</tool_response>"

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split())))}


def test_arrow_round_trip_preserves_semantic_tool_schema():
    generator = load_generator()
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)

    expected_tools = namespace["TOOLS"]
    round_tripped = Dataset.from_list([{"tools": expected_tools}])[0]["tools"]

    assert namespace["canonical_tool_schema"](round_tripped) == namespace["TOOL_SCHEMA_JSON"]
    changed = deepcopy(round_tripped)
    changed[0]["function"]["description"] = "Different semantics"
    assert namespace["canonical_tool_schema"](changed) != namespace["TOOL_SCHEMA_JSON"]


def test_rendered_tool_block_preserves_semantic_tool_schema():
    generator = load_generator()
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)

    rendered = (
        "<|im_start|>system\n<tools>\n"
        + "\n".join(json.dumps(tool) for tool in namespace["TOOLS"])
        + "\n</tools><|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert namespace["rendered_tool_schema"](rendered) == namespace["TOOL_SCHEMA_JSON"]


def test_repository_family_split_always_has_two_nonempty_partitions():
    generator = load_generator()
    split_cell = code_cell_containing(generator.build_02_data(), "validation_family_count")
    namespace = {
        "prepared": Dataset.from_list(
            [
                {"repo_family": "family-a", "value": 1},
                {"repo_family": "family-b", "value": 2},
            ]
        ),
        "Counter": Counter,
        "hashlib": hashlib,
        "PUSH_DATASET": False,
        "DEMO_MODE": True,
    }
    exec(split_cell, namespace)

    dataset_dict = namespace["dataset_dict"]
    assert set(dataset_dict) == {"train", "validation"}
    assert len(dataset_dict["train"]) == 1
    assert len(dataset_dict["validation"]) == 1
    assert set(dataset_dict["train"]["repo_family"]).isdisjoint(
        dataset_dict["validation"]["repo_family"]
    )


def test_notebooks_02_and_03_demo_data_execute_after_arrow_round_trip():
    generator = load_generator()
    notebook_02 = generator.build_02_data()
    namespace_02 = {
        "json": json,
        "tokenizer": FakeTokenizer(),
        "Dataset": Dataset,
        "DEMO_MODE": True,
        "SOURCE_DATASET_IDS": [],
        "Counter": Counter,
        "hashlib": hashlib,
        "np": np,
    }
    exec(generator.TOOLS_CELL, namespace_02)
    exec(code_cell_containing(notebook_02, "demo_rows = ["), namespace_02)
    exec(code_cell_containing(notebook_02, "def validate_row"), namespace_02)
    exec(code_cell_containing(notebook_02, "def render_row(row: dict)"), namespace_02)
    exec(code_cell_containing(notebook_02, "validation_family_count"), namespace_02)
    assert len(namespace_02["dataset_dict"]["train"]) == 1
    assert len(namespace_02["dataset_dict"]["validation"]) == 1

    notebook_03 = generator.build_03_sft()
    namespace_03 = {
        "json": json,
        "tokenizer": FakeTokenizer(),
        "Dataset": Dataset,
        "DEMO_MODE": True,
    }
    exec(generator.TOOLS_CELL, namespace_03)
    exec(code_cell_containing(notebook_03, "def demo_rows()"), namespace_03)
    assert len(namespace_03["train_dataset"]) == 1
    assert len(namespace_03["eval_dataset"]) == 1


def test_baseline_search_uses_bounded_python_fallback(tmp_path):
    generator = load_generator()
    executor_cell = code_cell_containing(generator.build_01_baseline(), "class PilotTask")
    namespace = {
        "dataclass": dataclass,
        "Path": Path,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "json": json,
        "re": re,
        "DEMO_MODE": False,
        "PILOT_MANIFEST": tmp_path / "unused.jsonl",
        "TASK_ENV": {},
    }
    exec(executor_cell, namespace)

    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("def clamp(value):\n    return value\n")
    task = namespace["PilotTask"](
        task_id="search-test",
        repo_path=str(tmp_path),
        request="Find clamp",
        visible_test_command=[sys.executable, "-V"],
        hidden_test_command=[sys.executable, "-V"],
    )

    result = namespace["execute_tool"](task, "search", {"query": "clamp"})
    assert "src/example.py:1:def clamp(value):" in result
    assert namespace["execute_tool"](task, "search", {"query": "["}).startswith(
        "invalid regular expression:"
    )


def test_baseline_apply_patch_accepts_parsed_model_patches(tmp_path):
    """The XML parser strips the patch's final newline, and git apply rejects
    such a patch as corrupt. The executor must normalise it back."""
    generator = load_generator()
    baseline = generator.build_01_baseline()
    namespace = {
        "dataclass": dataclass,
        "Path": Path,
        "shutil": shutil,
        "subprocess": subprocess,
        "sys": sys,
        "json": json,
        "re": re,
        "DEMO_MODE": False,
        "PILOT_MANIFEST": tmp_path / "unused.jsonl",
        "TASK_ENV": {},
    }
    exec(code_cell_containing(baseline, "def parse_tool_calls"), namespace)
    exec(code_cell_containing(baseline, "class PilotTask"), namespace)

    source = tmp_path / "src"
    source.mkdir()
    (source / "clamp.py").write_text(
        "def clamp(value, lower, upper):\n"
        "    return min(lower, max(upper, value))\n"
    )
    raw = (
        "<tool_call>\n<function=apply_patch>\n"
        "<parameter=patch>\n"
        "--- a/src/clamp.py\n+++ b/src/clamp.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def clamp(value, lower, upper):\n"
        "-    return min(lower, max(upper, value))\n"
        "+    return max(lower, min(upper, value))\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    arguments = namespace["parse_tool_calls"](raw)[1][0]["function"]["arguments"]
    task = namespace["PilotTask"](
        task_id="patch-test",
        repo_path=str(tmp_path),
        request="Fix clamp",
        visible_test_command=[sys.executable, "-V"],
        hidden_test_command=[sys.executable, "-V"],
    )

    assert namespace["execute_tool"](task, "apply_patch", arguments) == "patch applied"
    assert "max(lower, min(upper, value))" in (source / "clamp.py").read_text()


def test_baseline_parser_preserves_unified_diff_context_whitespace():
    generator = load_generator()
    parser_cell = code_cell_containing(generator.build_01_baseline(), "def parse_tool_calls")
    namespace = {"re": re}
    exec(parser_cell, namespace)

    raw = (
        "<tool_call>\n<function=apply_patch>\n"
        "<parameter=patch>\n"
        "--- a/src/clamp.py\n+++ b/src/clamp.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def clamp(value, lower, upper):\n"
        "-    return min(lower, max(upper, value))\n"
        "+    return max(lower, min(upper, value))\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    patch = namespace["parse_tool_calls"](raw)[1][0]["function"]["arguments"]["patch"]
    assert "\n def clamp(value, lower, upper):\n" in patch
    assert patch.startswith("--- a/src/clamp.py\n")


def test_reward_fixture_rejects_deleted_visible_tests():
    generator = load_generator()
    environment_cell = code_cell_containing(generator.build_05_grpo(), "class ToyCodingEnv")
    namespace = {
        "hashlib": hashlib,
        "Path": Path,
        "re": re,
        "shutil": shutil,
        "tempfile": tempfile,
    }
    exec(environment_cell, namespace)

    environment = namespace["ToyCodingEnv"]()
    environment.reset()
    assert environment.apply_patch(
        "--- a/src/clamp.py\n+++ b/src/clamp.py\n@@ -1,2 +1,2 @@\n"
        " def clamp(value, low, high):\n-    return value\n"
        "+    return max(low, min(high, value))\n"
    ) == "Done!"
    (environment.root / "tests" / "test_clamp.py").unlink()

    assert environment.run_tests("unit").startswith("Test integrity failure")
    assert environment.get_reward() == 0.0


def test_generated_notebooks_have_restart_and_schema_guards():
    generator = load_generator()
    notebooks = [
        generator.build_00_preflight(),
        generator.build_01_baseline(),
        generator.build_02_data(),
        generator.build_03_sft(),
        generator.build_04_dpo(),
        generator.build_05_grpo(),
        generator.build_06_qat_export(),
        generator.build_07_collect_and_evaluate(),
    ]

    all_source = "\n".join(cell.source for notebook in notebooks for cell in notebook.cells)
    assert "row.get(\"tools\") != TOOLS" not in all_source
    assert "then resume here" not in all_source
    assert "then continue from the runtime/authentication cell" not in all_source
    assert "This runtime was restarted. Rerun the notebook from the first cell" in all_source
    assert "TOOL_SCHEMA_JSON" in all_source
    assert 'assert "<function=read_file>" in rendered_probe' not in all_source
    assert "rendered_tool_schema(rendered_probe) == TOOL_SCHEMA_JSON" in all_source
    assert "tokenizer(rendered, return_tensors=\"pt\"" not in all_source
    assert "text=rendered_probe" in all_source
    assert "tokenizer(text=text, add_special_tokens=False)" in all_source
    assert 'globals().pop(_stale_name, None)' in all_source
    assert "torch.cuda.empty_cache()" in all_source
    assert "torch._dynamo.reset()" in all_source
    # The masking gate must not be pinned to a phrase only the demo row has.
    assert 'assert "Implemented the bounded clamp" in joined_supervision' not in all_source
    assert "masking_problems(" in all_source
    # A bare "in_proj" matches none of Qwen3.8's DeltaNet projections.
    assert '"gate_proj", "up_proj", "down_proj", "in_proj", "out_proj",' not in all_source

    for notebook in notebooks:
        generator.validate_notebook(notebook, Path("generated.ipynb"))


def test_every_model_load_is_guarded_against_silent_offload():
    """A load without enough free VRAM makes accelerate offload modules to
    CPU, which then OOMs mid-episode when a spilled tensor is copied back."""
    generator = load_generator()
    notebooks = {
        "00": generator.build_00_preflight(),
        "01": generator.build_01_baseline(),
        "02": generator.build_02_data(),
        "03": generator.build_03_sft(),
        "04": generator.build_04_dpo(),
        "05": generator.build_05_grpo(),
        "06": generator.build_06_qat_export(),
        "07": generator.build_07_collect_and_evaluate(),
    }

    load_cells = 0
    for name, notebook in notebooks.items():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or "FastLanguageModel.from_pretrained" not in cell.source:
                continue
            load_cells += 1
            assert "require_free_vram(" in cell.source, f"notebook {name} cell {index}"
            assert "assert_model_fully_resident(" in cell.source, f"notebook {name} cell {index}"
    assert load_cells == 9

    auth = generator.AUTH_AND_RUNTIME
    assert "def release_stale_gpu_state" in auth
    assert "def require_free_vram" in auth
    assert "def assert_model_fully_resident" in auth
    assert '"last_traceback"' in auth
    assert "\nrelease_stale_gpu_state()" in auth


def test_core_install_uses_a_resolvable_unsloth_compatibility_set():
    generator = load_generator()
    install = generator.INSTALL_CORE

    for package, expected in {
        "transformers": "5.3.0",
        "trl": "0.22.2",
        "datasets": "4.3.0",
        "peft": "0.19.0",
    }.items():
        assert f'"{package}": "{expected}"' in install
    assert "c49429ed1f8b89749de77c0ec930ef19685c9ae5" not in install
    assert "b39c2276567639b93ca5b53658751e0f9c09b92f" not in install
    assert "--quiet" not in install
    assert "/content/qwen38_pip_install.log" in install


def test_g4_preflight_accepts_decimal_96_gb_and_rejects_decimal_48_gb():
    generator = load_generator()
    auth = generator.AUTH_AND_RUNTIME

    decimal_96_gb_in_gib = 96_000_000_000 / 1024**3
    decimal_48_gb_in_gib = 48_000_000_000 / 1024**3
    assert decimal_96_gb_in_gib < 90
    assert decimal_96_gb_in_gib >= 85
    assert decimal_48_gb_in_gib < 85

    assert "MIN_G4_TOTAL_GIB = 85.0" in auth
    assert "gpu_total_gib < MIN_G4_TOTAL_GIB" in auth
    assert "gpu_gib < 90" not in auth
    assert '"gpu_total_gib": round(gpu_total_gib, 2)' in auth


def test_generated_cell_ids_are_deterministic():
    generator = load_generator()
    first = generator.build_02_data()
    second = generator.build_02_data()
    first_ids = [cell.id for cell in first.cells]
    second_ids = [cell.id for cell in second.cells]

    assert first_ids == second_ids
    assert len(first_ids) == len(set(first_ids))


def test_local_markdown_links_resolve():
    markdown_files = [ROOT / "README.md", ROOT / "notebooks" / "README.md"]
    markdown_files.extend(sorted((ROOT / "docs").glob("*.md")))
    failures = []
    for markdown_file in markdown_files:
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown_file.read_text()):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if local_target and not (markdown_file.parent / local_target).resolve().exists():
                failures.append(f"{markdown_file.relative_to(ROOT)} -> {target}")
    assert not failures, failures


class _FakeParameter:
    def __init__(self, requires_grad: bool = True):
        self.requires_grad = requires_grad

    def requires_grad_(self, flag: bool):
        self.requires_grad = flag
        return self

    def numel(self) -> int:
        return 1


class _FakeLinear:
    """Stands in for torch.nn.Linear; gains lora_A when an adapter attaches."""

    def __init__(self):
        self.lora_A: dict = {}


class _FakeNonLinear:
    """A norm or embedding: discovery must never target it."""


class _FakeTorch:
    bfloat16 = "bfloat16"

    class nn:
        Linear = _FakeLinear


def qwen38_linear_module_names() -> list[str]:
    """The linear modules the published Qwen3.8-27B weight index actually has.

    Three of every four layers are Gated DeltaNet (`linear_attn`), whose
    projections are named in_proj_qkv/z/a/b and out_proj. config.json places
    the full-attention layer at every fourth position.
    """
    names = []
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}"
        if layer % 4 == 3:
            names += [f"{prefix}.self_attn.{projection}_proj" for projection in ("q", "k", "v", "o")]
        else:
            names += [
                f"{prefix}.linear_attn.{projection}"
                for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")
            ]
        names += [f"{prefix}.mlp.{projection}_proj" for projection in ("gate", "up", "down")]
    names += [f"mtp.layers.0.self_attn.{projection}_proj" for projection in ("q", "k", "v", "o")]
    names += [f"mtp.layers.0.mlp.{projection}_proj" for projection in ("gate", "up", "down")]
    names.append("mtp.fc")
    for block in range(27):
        names += [
            f"model.visual.blocks.{block}.attn.qkv",
            f"model.visual.blocks.{block}.attn.proj",
            f"model.visual.blocks.{block}.mlp.linear_fc1",
            f"model.visual.blocks.{block}.mlp.linear_fc2",
        ]
    names += ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2", "lm_head"]
    return names


class _FakeQwen38Model:
    def __init__(self):
        self._modules = {name: _FakeLinear() for name in qwen38_linear_module_names()}
        self._modules["model.language_model.norm"] = _FakeNonLinear()
        self._parameters = {f"{name}.weight": _FakeParameter(False) for name in self._modules}

    def named_modules(self):
        return list(self._modules.items())

    def named_parameters(self):
        return list(self._parameters.items())

    def parameters(self):
        return list(self._parameters.values())

    def print_trainable_parameters(self):
        return None

    def attach_lora(self, target_modules: set[str]) -> None:
        # PEFT matches target_modules by name suffix, so the MTP head's own
        # q_proj/o_proj are adapted too unless something freezes them.
        for name, module in self._modules.items():
            if isinstance(module, _FakeLinear) and name.rsplit(".", 1)[-1] in target_modules:
                module.lora_A = {"default": object()}
                self._parameters[f"{name}.lora_A.default.weight"] = _FakeParameter(True)


class _FakeFastLanguageModel:
    @staticmethod
    def from_pretrained(**kwargs):
        # Notebook 04 sets tokenizer.padding_side, so the stand-in must accept
        # attribute assignment.
        return _FakeQwen38Model(), SimpleNamespace()

    @staticmethod
    def get_peft_model(model, target_modules, **kwargs):
        model.attach_lora(set(target_modules))
        return model


REVIEWED_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "gate_proj", "up_proj", "down_proj",
}


def run_lora_discovery(cell: str) -> dict:
    namespace = {
        "json": json,
        "torch": _FakeTorch,
        "FastLanguageModel": _FakeFastLanguageModel,
        "require_free_vram": lambda *_: 90.0,
        "assert_model_fully_resident": lambda *_, **__: None,
        "MODEL_ID": "unsloth/Qwen3.8-27B",
        "MERGED_SFT_MODEL_ID": "user/merged",
        "MERGED_SFT_REVISION": "REPLACE_WITH_ACCEPTED_COMMIT",
        "MAX_SEQ_LENGTH": 4096,
        "hf_token": "token",
    }
    exec(cell, namespace)
    return namespace


def test_lora_adapters_cover_the_gated_deltanet_layers():
    """Three of four Qwen3.8 layers are linear attention. A suffix list built
    for a standard transformer misses every one of their projections."""
    generator = load_generator()
    namespace = run_lora_discovery(
        code_cell_containing(generator.build_03_sft(), "REVIEWED_TARGET_SUFFIXES")
    )

    assert set(namespace["target_modules"]) == REVIEWED_SUFFIXES
    adapted = namespace["adapted_counts"]
    for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        assert adapted[projection] == 48, projection
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert adapted[projection] == 16, projection
    for projection in ("gate_proj", "up_proj", "down_proj"):
        assert adapted[projection] == 64, projection
    assert sum(adapted.values()) == 496


def test_lora_leaves_the_mtp_head_and_vision_tower_frozen():
    generator = load_generator()
    namespace = run_lora_discovery(
        code_cell_containing(generator.build_03_sft(), "REVIEWED_TARGET_SUFFIXES")
    )
    parameters = dict(namespace["model"].named_parameters())

    # Suffix matching does reach the MTP head, so the freeze is load-bearing.
    mtp_adapters = [name for name in parameters if name.startswith("mtp.") and "lora_A" in name]
    assert mtp_adapters, "expected suffix matching to reach the MTP head"
    assert not [name for name in mtp_adapters if parameters[name].requires_grad]
    assert not [
        name for name, parameter in parameters.items()
        if parameter.requires_grad and "visual" in name
    ]


def test_sft_and_dpo_attach_adapters_to_the_same_module_set():
    """A different subnetwork per stage would make the DPO delta uninterpretable."""
    generator = load_generator()
    sft = run_lora_discovery(code_cell_containing(generator.build_03_sft(), "REVIEWED_TARGET_SUFFIXES"))
    dpo = run_lora_discovery(code_cell_containing(generator.build_04_dpo(), "REVIEWED_TARGET_SUFFIXES"))

    assert sft["REVIEWED_TARGET_SUFFIXES"] == dpo["REVIEWED_TARGET_SUFFIXES"] == REVIEWED_SUFFIXES
    assert sft["adapted_counts"] == dpo["adapted_counts"]


def test_sft_resume_picks_the_highest_numbered_checkpoint(tmp_path):
    """checkpoint-10 sorts before checkpoint-9 as a string."""
    generator = load_generator()
    namespace = {"RUN_TRAINING": False}
    exec(code_cell_containing(generator.build_03_sft(), "def latest_checkpoint"), namespace)
    latest_checkpoint = namespace["latest_checkpoint"]

    for name in ("checkpoint-2", "checkpoint-9", "checkpoint-10", "checkpoint-final"):
        (tmp_path / name).mkdir()
    assert latest_checkpoint(tmp_path).name == "checkpoint-10"
    assert latest_checkpoint(tmp_path / "absent") is None


def test_baseline_treats_truncated_and_overlong_turns_as_terminations():
    """A turn cut off at the token cap parses as 'no tool calls', so storing it
    as the final answer scores a truncation as a completed episode."""
    generator = load_generator()
    baseline = generator.build_01_baseline()
    generation_cell = code_cell_containing(baseline, "def generate_turn")
    episode_cell = code_cell_containing(baseline, "def run_episode")

    assert "EOS_TOKEN_IDS" in generation_cell
    assert '"fault": "context_budget"' in generation_cell
    assert '"output_truncated"' in generation_cell
    assert "prompt_tokens + MAX_NEW_TOKENS_PER_TURN > MAX_SEQUENCE_LENGTH" in generation_cell
    assert 'if turn["fault"] is not None:' in episode_cell
    # The old shape silently promoted a truncated prefix to a final answer.
    assert "raw, prompt_count, completion_count = generate_turn" not in episode_cell


def test_notebook_02_accepts_the_documented_non_agentic_lane():
    generator = load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(code_cell_containing(generator.build_02_data(), "def validate_row"), namespace)
    validate_row = namespace["validate_row"]

    base = {
        "tool_schema_version": namespace["TOOL_SCHEMA_VERSION"],
        "tool_schema_json": namespace["TOOL_SCHEMA_JSON"],
        "tools": namespace["TOOLS"],
        "verification": {"all_required_tests_pass": True},
    }
    reasoning_row = dict(base, lane="non_agentic", messages=[
        {"role": "user", "content": "Implement a bounded cache."},
        {"role": "assistant", "content": "Here is the implementation."},
    ])
    assert validate_row(reasoning_row) == []

    # An agentic row still has to call a tool, and a non-agentic row must not.
    assert validate_row(dict(reasoning_row, lane="agentic"))
    mislabelled = dict(base, lane="non_agentic", messages=[
        {"role": "user", "content": "Fix it."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "read_file", "arguments": {"path": "src/a.py"}}}
        ]},
        {"role": "tool", "name": "read_file", "content": "x = 1"},
    ])
    assert validate_row(mislabelled)


def test_notebook_07_imports_the_shared_loop_instead_of_restating_it():
    """Three hand-copied episode loops would drift, and a gate that drifts
    from the collector it grades is worse than no gate."""
    generator = load_generator()
    source = "\n".join(cell.source for cell in generator.build_07_collect_and_evaluate().cells)

    assert "from qwen3_8_27b_code.evaluation import" in source
    assert "from qwen3_8_27b_code.collection import" in source
    for redefinition in ("def run_episode", "def execute_tool", "def rejection_reason", "def scorecard"):
        assert redefinition not in source, redefinition
    # The GPU-specific part is the only thing the notebook defines.
    assert "def build_policy_factory" in source
    assert "TurnResult(" in source


def test_notebook_07_never_collects_from_the_held_out_suite():
    generator = load_generator()
    notebook = generator.build_07_collect_and_evaluate()
    collection_cell = code_cell_containing(notebook, "result = collect(")

    assert "task_from_fixture" in collection_cell
    assert "evaluation_tasks(" not in collection_cell
    evaluation_cell = code_cell_containing(notebook, "label=\"upstream-bf16\"")
    assert "evaluation_suite" in evaluation_cell


def test_notebook_07_applies_the_gate_and_persists_the_comparison():
    generator = load_generator()
    gate_cell = code_cell_containing(
        generator.build_07_collect_and_evaluate(), "comparison[\"gate_passed\"]"
    )
    assert "compare(" in gate_cell
    assert "gate_passed(checks)" in gate_cell
    assert "comparison.json" in gate_cell
    # A suite this small reports paired outcomes, not a significance claim.
    assert "task_level" in gate_cell


def test_notebook_07_releases_the_baseline_before_loading_the_candidate():
    """Two 27B checkpoints do not coexist on one card."""
    generator = load_generator()
    candidate_cell = code_cell_containing(
        generator.build_07_collect_and_evaluate(), "RUN_CANDIDATE_EVAL:"
    )
    assert "release_stale_gpu_state()" in candidate_cell
    assert candidate_cell.index("release_stale_gpu_state()") < candidate_cell.index(
        "FastLanguageModel.from_pretrained"
    )
