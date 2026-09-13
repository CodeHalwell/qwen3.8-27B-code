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
        generator.build_08_distil(),
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
        "08": generator.build_08_distil(),
    }

    load_cells = 0
    for name, notebook in notebooks.items():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or "FastModel.from_pretrained" not in cell.source:
                continue
            load_cells += 1
            assert "require_free_vram(" in cell.source, f"notebook {name} cell {index}"
            assert "assert_model_fully_resident(" in cell.source, f"notebook {name} cell {index}"
    assert load_cells == 9
    # Unsloth's own Qwen3.8 notebook loads with FastModel; the text-only
    # loader must not creep back in through a copied cell.
    for name, notebook in notebooks.items():
        for cell in notebook.cells:
            assert "FastLanguageModel" not in cell.source, name

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


class _FakeFastModel:
    @staticmethod
    def from_pretrained(**kwargs):
        # Notebook 04 sets tokenizer.padding_side, so the stand-in must accept
        # attribute assignment.
        return _FakeQwen38Model(), SimpleNamespace()

    @staticmethod
    def get_peft_model(model, target_modules, finetune_vision_layers=True, **kwargs):
        # The suite passes the reviewed list and switches the vision tower
        # off in the loader's own terms as well.
        assert finetune_vision_layers is False
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
        "FastModel": _FakeFastModel,
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
        "FastModel.from_pretrained"
    )


def test_processor_aware_tokenizer_helper_is_shared():
    """FastModel returns a processor; token-level reads go through the text tokenizer."""
    generator = load_generator()
    assert "def text_tokenizer_of(tokenizer)" in generator.TOOLS_CELL
    assert "text_tokenizer_of(tokenizer).apply_chat_template(" in generator.TOOLS_CELL
    namespace = {"json": json}
    exec(generator.TOOLS_CELL, namespace)
    plain = SimpleNamespace()
    processor = SimpleNamespace(tokenizer=plain)
    assert namespace["text_tokenizer_of"](plain) is plain
    assert namespace["text_tokenizer_of"](processor) is plain

    policy_cell = code_cell_containing(generator.build_07_collect_and_evaluate(), "def build_policy_factory")
    assert "text_tokenizer.eos_token_id" in policy_cell
    assert 'text_tokenizer.convert_tokens_to_ids("</think>")' in policy_cell
    baseline_cell = code_cell_containing(generator.build_01_baseline(), "EOS_TOKEN_IDS = {")
    assert "text_tokenizer_of(tokenizer).eos_token_id" in baseline_cell


def test_notebook_07_caps_generation_per_effort_and_runs_the_ladder():
    generator = load_generator()
    notebook = generator.build_07_collect_and_evaluate()
    config_cell = code_cell_containing(notebook, "MAX_NEW_TOKENS_BY_EFFORT")
    namespace = {"EpisodeBudget": lambda **kwargs: kwargs}
    for line in config_cell.splitlines():
        if line.startswith(("MAX_", "REASONING_EFFORT", "EPISODE_BUDGET", "RUN_EFFORT_LADDER", "EFFORT_LADDER")):
            exec(line, namespace)
    assert set(namespace["MAX_NEW_TOKENS_BY_EFFORT"]) == {"low", "medium", "xhigh"}
    caps = namespace["MAX_NEW_TOKENS_BY_EFFORT"]
    assert caps["low"] < caps["medium"] < caps["xhigh"]
    # Ten xhigh turns of reasoning must fit the window with the caps as set.
    assert namespace["MAX_SEQUENCE_LENGTH"] >= caps["xhigh"] + 3 * caps["medium"]
    assert namespace["EPISODE_BUDGET"] == {"tool_calls": 30, "wall_seconds": 900.0}
    assert namespace["RUN_EFFORT_LADDER"] is False

    policy_cell = code_cell_containing(notebook, "def build_policy_factory")
    assert "max_new_tokens = MAX_NEW_TOKENS_BY_EFFORT[reasoning_effort]" in policy_cell
    assert "MAX_NEW_TOKENS_PER_TURN" not in policy_cell
    assert "prompt_tokens + max_new_tokens > MAX_SEQUENCE_LENGTH" in policy_cell

    ladder_cell = code_cell_containing(notebook, "RUN_EFFORT_LADDER:")
    for effort in ("low", "medium", "xhigh"):
        assert f'"{effort}"' in ladder_cell
    assert "effort_ladder(ladder_reports, success_tolerance=EFFORT_LADDER_TOLERANCE)" in ladder_cell
    assert "reasoning_effort=effort" in ladder_cell
    assert "effort_ladder.json" in ladder_cell


def test_training_notebooks_publish_privately_and_save_on_a_real_cadence():
    """A real run must not push an adapter to a public repo, and must not
    save-and-push after every optimiser step."""
    generator = load_generator()
    for build, config_marker, args_marker in (
        (generator.build_03_sft, "PUSH_MERGED_BF16 = False", "training_args = SFTConfig("),
        (generator.build_04_dpo, "LENGTH_PAIRS_LOCAL_JSONL", "dpo_args = DPOConfig("),
    ):
        notebook = build()
        config_cell = code_cell_containing(notebook, config_marker)
        for demo_mode, expected in ((True, 1), (False, 10)):
            namespace = {"DEMO_MODE": demo_mode}
            for line in config_cell.splitlines():
                if line.startswith(("EVAL_EVERY_STEPS", "SAVE_EVERY_STEPS")):
                    exec(line, namespace)
            assert namespace["EVAL_EVERY_STEPS"] == expected
            assert namespace["SAVE_EVERY_STEPS"] == expected

        args_cell = code_cell_containing(notebook, args_marker)
        assert "eval_steps=EVAL_EVERY_STEPS," in args_cell
        assert "save_steps=SAVE_EVERY_STEPS," in args_cell
        assert "hub_private_repo=True," in args_cell
        assert "eval_steps=1," not in args_cell
        assert "save_steps=1," not in args_cell

    sft_args = code_cell_containing(generator.build_03_sft(), "training_args = SFTConfig(")
    assert "learning_rate=LEARNING_RATE," in sft_args
    sft_config = code_cell_containing(generator.build_03_sft(), "PUSH_MERGED_BF16 = False")
    assert "LEARNING_RATE = 2e-5" in sft_config

    # The run inputs a checkpoint must be attributed to are in its manifest.
    for cell, keys in (
        (sft_config, ("learning_rate", "eval_every_steps", "save_every_steps", "optimizer")),
        (
            code_cell_containing(generator.build_04_dpo(), "LENGTH_PAIRS_LOCAL_JSONL"),
            (
                "learning_rate", "beta", "eval_every_steps", "save_every_steps", "optimizer",
                "max_length_pair_share", "preference_sources",
            ),
        ),
    ):
        manifest = cell[cell.index("run_manifest = {"):]
        for key in keys:
            assert f'"{key}":' in manifest, key
    dpo_args = code_cell_containing(generator.build_04_dpo(), "dpo_args = DPOConfig(")
    assert "learning_rate=LEARNING_RATE," in dpo_args
    assert "beta=DPO_BETA," in dpo_args
    dpo_train = code_cell_containing(generator.build_04_dpo(), "preference_mixture.json")
    assert '"dpo" / "run_manifest.json"' in dpo_train
    # The manifest names the rows actually read, not the configured Hub id.
    assert 'run_manifest["preference_sources"] = PREFERENCE_SOURCES' in dpo_train
    dpo_load = code_cell_containing(generator.build_04_dpo(), "demo_preferences = Dataset.from_list")
    assert "PREFERENCE_SOURCES.append(local_source(PREFERENCE_LOCAL_JSONL, rows))" in dpo_load
    assert "PREFERENCE_SOURCES.append(local_source(LENGTH_PAIRS_LOCAL_JSONL, length_rows))" in dpo_load
    assert '"resolved_revision": HfApi(token=hf_token).dataset_info(' in dpo_load
    assert '"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()' in dpo_load


PUBLISH_MARKERS = (
    "push_to_hub(",
    "push_to_hub_merged(",
    "push_to_hub_gguf(",
    "upload_folder(",
    "training_args = SFTConfig(",
    "dpo_args = DPOConfig(",
    "GRPOConfig(",
)


def test_every_hub_publish_is_guarded_against_an_existing_public_repo():
    """`private=True` and `hub_private_repo` apply only when a repo is
    created; an existing public repo stays public. Every publish site must
    check first, and a trainer's check must run before the trainer is built,
    which is when it creates the repo."""
    generator = load_generator()
    for runtime in (generator.AUTH_AND_RUNTIME, generator.TEACHER_RUNTIME):
        assert "def require_private_repo(" in runtime
        assert "repo_info(repo_id, repo_type=repo_type).private" in runtime
    notebooks = {
        "01": generator.build_01_baseline(),
        "02": generator.build_02_data(),
        "03": generator.build_03_sft(),
        "04": generator.build_04_dpo(),
        "05": generator.build_05_grpo(),
        "06": generator.build_06_qat_export(),
        "07": generator.build_07_collect_and_evaluate(),
        "08": generator.build_08_distil(),
    }
    guarded = 0
    for name, notebook in notebooks.items():
        cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
        seen_guard = False
        for index, source in enumerate(cells):
            if "require_private_repo(" in source and "def require_private_repo" not in source:
                seen_guard = True
            if not any(marker in source for marker in PUBLISH_MARKERS):
                continue
            if "Config(" in source and "push_to_hub=PUSH_ADAPTER" in source:
                # The guard for a trainer lives in an earlier cell.
                assert seen_guard, f"notebook {name} cell {index}: trainer built before the repo check"
                assert "hub_private_repo=True" in source, f"notebook {name} cell {index}"
            elif "trainer.push_to_hub(" in source:
                assert seen_guard, f"notebook {name} cell {index}"
            else:
                assert "require_private_repo(" in source, f"notebook {name} cell {index}"
            if "upload_folder(" in source or "push_to_hub_gguf(" in source:
                # Neither call creates a private repo on its own: upload_folder
                # has no private flag and push_to_hub_gguf uses its own default.
                assert "private=True, exist_ok=True" in source, f"notebook {name} cell {index}"
                assert "private=True,\n" not in source.split("upload_folder(")[-1], f"notebook {name} cell {index}"
            guarded += 1
    assert guarded >= 12

    # Where the publish follows an expensive job, an existing public target
    # is found at configuration time, not after the GPU or the teacher bill.
    for name, marker, guard in (
        ("03", "PUSH_MERGED_BF16 = False", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("04", "LENGTH_PAIRS_LOCAL_JSONL", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("05", "ROLLOUT_POLICY_PRECISION = ", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("06", "RUN_STANDARD_GGUF_EXPORT = False", "require_private_repo(QAT_OUTPUT_ID)"),
        ("06", "RUN_STANDARD_GGUF_EXPORT = False", "require_private_repo(GGUF_OUTPUT_ID)"),
        ("07", "GATE_REPORTS_REPO =", 'require_private_repo(GATE_REPORTS_REPO, "dataset")'),
        ("08", "TEACHER_REPO = ", 'require_private_repo(TEACHER_REPO, "dataset")'),
    ):
        config_cell = code_cell_containing(notebooks[name], marker)
        assert guard in config_cell, (name, guard)


def test_notebook_07_persists_reports_across_colab_sessions():
    """The gate pairs a candidate with a baseline measured in an earlier
    runtime, so the reports must round-trip through the Hub."""
    generator = load_generator()
    notebook = generator.build_07_collect_and_evaluate()
    config_cell = code_cell_containing(notebook, "GATE_REPORTS_REPO =")
    assert "PULL_REPORTS_FROM_HUB = True" in config_cell
    assert "repo_exists(GATE_REPORTS_REPO" in config_cell
    assert "snapshot_download(" in config_cell
    # Pulled copies never land where this session writes: a stale report
    # must be chosen by name, not found by accident.
    assert 'HUB_REPORT_DIR = RUN_ROOT / "gate_hub"' in config_cell
    assert "local_dir=str(HUB_REPORT_DIR)" in config_cell
    assert "local_dir=str(REPORT_DIR)" not in config_cell
    assert 'GATE_BASELINE_FILE = "baseline.json"' in config_cell
    assert "def report_provenance(" in config_cell
    # One writer for the notebook and the CLI: the notebook passes its
    # settings to the shared builder rather than assembling the dict itself.
    assert "return build_provenance(" in config_cell
    for key in ("model", "harness_revision", "reasoning_effort", "max_new_tokens", "episode_budget", "attempts_per_task"):
        assert f"{key}=" in config_cell
    assert '"measured_at"' not in config_cell

    # Every report this notebook writes records how it was measured.
    # The model reference is pinned: a Hub id resolved to the commit that was
    # loaded, never the moving branch name.
    assert 'MODEL_REVISION = "main"' in config_cell
    assert "def resolved_revision(" in config_cell
    assert 'stock_model_ref = f"{MODEL_ID}@{resolved_revision(MODEL_ID, MODEL_REVISION)}"' in config_cell
    for marker, model_ref in (
        ("RUN_BASELINE_EVAL:", "report_provenance(stock_model_ref)"),
        ("RUN_EFFORT_LADDER:", "report_provenance(stock_model_ref, reasoning_effort=effort)"),
        ("RUN_CANDIDATE_EVAL:", "report_provenance(candidate_model_ref)"),
    ):
        cell = code_cell_containing(notebook, marker)
        assert model_ref in cell
        assert cell.index(".metadata = report_provenance(") < cell.index("write_report(")
    assert "seeds=DEFAULT_SEEDS[:EVAL_ATTEMPTS]," in config_cell
    baseline_cell = code_cell_containing(notebook, "RUN_BASELINE_EVAL:")
    assert "revision=MODEL_REVISION," in baseline_cell
    assert baseline_cell.index("baseline_report_path.unlink(missing_ok=True)") < baseline_cell.index("evaluate(")
    candidate_cell = code_cell_containing(notebook, "RUN_CANDIDATE_EVAL:")
    assert "resolved_revision(ACCEPTED_ADAPTER_ID, ACCEPTED_REVISION)" in candidate_cell
    # A candidate counts only when this cell wrote it: the file is removed
    # before the evaluation and the flag set after the write.
    assert candidate_cell.index("candidate_written = False") < candidate_cell.index("if RUN_CANDIDATE_EVAL:")
    assert candidate_cell.index("candidate_report_path.unlink(missing_ok=True)") < candidate_cell.index("evaluate(")
    assert candidate_cell.index("write_report(candidate, candidate_report_path)") < candidate_cell.index(
        "candidate_written = True"
    )

    # The gate pairs only this session's candidate with a named baseline,
    # and refuses reports measured differently.
    gate_cell = code_cell_containing(notebook, 'comparison["gate_passed"]')
    assert 'RUN_CANDIDATE_EVAL and globals().get("candidate_written") and candidate_report_path.exists()' in gate_cell
    assert "HUB_REPORT_DIR / GATE_BASELINE_FILE" in gate_cell
    assert "HUB_REPORT_DIR / \"candidate.json\"" not in gate_cell
    assert "pairing_problems(" in gate_cell
    assert "GATE NOT RUN" in gate_cell
    assert 'comparison["provenance"]' in gate_cell
    # A rerun in the same runtime never republishes an earlier verdict: the
    # old comparison is removed before the gate decides whether to run.
    assert "comparison_path.unlink(missing_ok=True)" in gate_cell
    assert gate_cell.index("comparison_path.unlink(missing_ok=True)") < gate_cell.index("pairing_problems(")
    assert "comparison_path.write_text(" in gate_cell

    persist_cell = code_cell_containing(notebook, "create_repo(GATE_REPORTS_REPO")
    assert "private=True" in persist_cell
    assert "exist_ok=True" in persist_cell
    assert "folder_path=str(REPORT_DIR)" in persist_cell
    assert "repo_revision" in persist_cell
    # A remote verdict is deleted unless this session's copy replaces it.
    assert 'delete_patterns=["comparison.json"]' in persist_cell
    # The persist cell is the last code cell, after the collection cell,
    # so it carries everything the session produced.
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells[-1] == persist_cell
    collection_cell = code_cell_containing(notebook, "if RUN_COLLECTION:")
    assert "upload_folder" not in collection_cell


NOTEBOOK_GLOBALS = {"display", "get_ipython"}  # injected by the Colab kernel


def _cell_symbols(index: int, source: str) -> tuple[set[str], set[str], set[str]]:
    """Names a cell binds at module scope, reads at module scope, and reads
    from inside nested scopes (function and lambda bodies, class bodies)."""
    import symtable

    python = "\n".join(
        "pass  # magic" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()
    )
    table = symtable.symtable(python, f"cell {index}", "exec")
    binds = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
    top_reads = {
        s.get_name() for s in table.get_symbols()
        if s.is_referenced() and not (s.is_assigned() or s.is_imported())
    }
    nested_reads: set[str] = set()

    def walk(scope) -> None:
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                nested_reads.add(symbol.get_name())
        for child in scope.get_children():
            walk(child)

    for child in table.get_children():
        walk(child)
    return binds, top_reads, nested_reads


def undefined_notebook_names(cells: list[str]) -> list[str]:
    """Names a notebook reads that it never binds in time.

    Each cell is compiled on its own with ``symtable``. A name read at a
    cell's module scope must be bound by that cell or an earlier one, since
    the cell runs when it is reached. A name read inside a function, lambda
    or class body resolves when that body runs, which may be after a later
    cell binds it, so it must be bound somewhere in the notebook. Order
    inside one cell is not modelled, and a function called before a later
    cell binds its global is not caught.
    """
    import builtins

    parsed = [_cell_symbols(index, source) for index, source in enumerate(cells)]
    bound_anywhere = set(dir(builtins)) | NOTEBOOK_GLOBALS
    for binds, _, _ in parsed:
        bound_anywhere |= binds
    bound_so_far = set(dir(builtins)) | NOTEBOOK_GLOBALS
    problems: list[str] = []
    for index, (binds, top_reads, nested_reads) in enumerate(parsed):
        bound_so_far |= binds
        missing = (top_reads - bound_so_far) | (nested_reads - bound_anywhere)
        problems.extend(f"cell {index}: {name}" for name in sorted(missing))
    return problems


def test_undefined_name_checker_models_cell_boundaries():
    # Module-scope use before the import: the whole-file view passes it,
    # the notebook raises at cell 0.
    assert undefined_notebook_names(["digest = hashlib.sha256(b'x')", "import hashlib"]) == ["cell 0: hashlib"]
    assert undefined_notebook_names(["import hashlib", "digest = hashlib.sha256(b'x')"]) == []
    assert undefined_notebook_names(["!pip install x\nimport json", "print(json.dumps(rows))"]) == ["cell 1: rows"]
    # A function body may read a name a later cell binds; one nothing binds is a bug.
    assert undefined_notebook_names(["def render():\n    return tokenizer.name", "tokenizer = object()"]) == []
    assert undefined_notebook_names(["def g():\n    return helper()", "x = 1"]) == ["cell 0: helper"]


def test_every_notebook_cell_uses_only_names_defined_earlier():
    """A cell that uses a module it never imports raises NameError on Colab,
    while the contract tests above, which hand cells a ready namespace,
    still pass. Check every notebook cell against what came before it."""
    generator = load_generator()
    builders = {
        "00": generator.build_00_preflight,
        "01": generator.build_01_baseline,
        "02": generator.build_02_data,
        "03": generator.build_03_sft,
        "04": generator.build_04_dpo,
        "05": generator.build_05_grpo,
        "06": generator.build_06_qat_export,
        "07": generator.build_07_collect_and_evaluate,
        "08": generator.build_08_distil,
    }
    for name, build in builders.items():
        cells = [cell.source for cell in build().cells if cell.cell_type == "code"]
        assert undefined_notebook_names(cells) == [], f"notebook {name}"
