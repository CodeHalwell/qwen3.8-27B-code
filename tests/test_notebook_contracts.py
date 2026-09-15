"""Regression tests for contracts embedded in the generated Colab notebooks."""

from __future__ import annotations

import ast
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
    split_cell = code_cell_containing(generator.build_02_data(), "VALIDATION_ROW_SHARE")
    namespace = {
        "prepared": Dataset.from_list(
            [
                {"repo_family": "family-a", "value": 1},
                {"repo_family": "family-b", "value": 2},
            ]
        ),
        "Counter": Counter,
        "hashlib": hashlib,
        "json": json,
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


def test_validation_split_is_a_share_of_rows_not_of_families():
    """One bucketed family holds dozens of rows and one repository holds one,
    so counting families held out a fortieth of the corpus, all of it agentic."""
    generator = load_generator()
    split_cell = code_cell_containing(generator.build_02_data(), "VALIDATION_ROW_SHARE")
    rows = [{"repo_family": f"swe:repo-{i}", "lane": "agentic"} for i in range(300)]
    rows += [
        {"repo_family": f"instruct:generic/{i:02d}", "lane": "non_agentic"}
        for i in range(32) for _ in range(20)
    ]
    namespace = {
        "prepared": Dataset.from_list(rows), "Counter": Counter, "hashlib": hashlib,
        "json": json, "PUSH_DATASET": False, "DEMO_MODE": True,
    }
    exec(split_cell, namespace)
    dataset_dict = namespace["dataset_dict"]
    held_out = len(dataset_dict["validation"])
    target = round(len(rows) * 0.10)
    # Whole families still move together, so the count lands near the target
    # rather than on it; the old rule held out a quarter of this.
    assert target <= held_out <= target + 20
    # Both lanes are measured, which is the point of the change.
    assert set(dataset_dict["validation"]["lane"]) == {"agentic", "non_agentic"}
    assert set(dataset_dict["train"]["repo_family"]).isdisjoint(dataset_dict["validation"]["repo_family"])

    def split(rows):
        namespace = {
            "prepared": Dataset.from_list(rows), "Counter": Counter, "hashlib": hashlib,
            "json": json, "PUSH_DATASET": False, "DEMO_MODE": True,
        }
        exec(split_cell, namespace)
        return namespace["dataset_dict"]

    # A lane held in few, fat families can be walked past before the row
    # target is met. Given a family to spare, it is held out anyway.
    agentic = [{"repo_family": f"swe:repo-{i}", "lane": "agentic"} for i in range(300)]
    spare = split(agentic + [
        {"repo_family": f"opencodeinstruct:generic/{i:02d}", "lane": "non_agentic"}
        for i in range(2) for _ in range(5)
    ])
    assert set(spare["validation"]["lane"]) == {"agentic", "non_agentic"}
    assert set(spare["train"]["lane"]) == {"agentic", "non_agentic"}

    # With one family, holding it out would leave the lane with no training
    # rows at all. Unmeasured beats untrained, so it stays in training.
    # This family ranks ninth of 301, well inside the walk, so the walk
    # itself would take it were it not for the rule.
    sole = split(agentic + [
        {"repo_family": "opencodeinstruct:generic/00", "lane": "non_agentic"} for _ in range(5)
    ])
    assert set(sole["validation"]["lane"]) == {"agentic"}
    assert sole["train"]["lane"].count("non_agentic") == 5

    # A corpus of one lane stays a corpus of one lane; nothing is invented.
    single = split([{"repo_family": f"swe:repo-{i}", "lane": "agentic"} for i in range(40)])
    assert set(single["validation"]["lane"]) == {"agentic"}
    assert len(single["train"]) > 0


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
    exec(code_cell_containing(notebook_02, "VALIDATION_ROW_SHARE"), namespace_02)
    assert len(namespace_02["dataset_dict"]["train"]) == 1
    assert len(namespace_02["dataset_dict"]["validation"]) == 1

    notebook_03 = generator.build_03_sft()
    namespace_03 = {
        "json": json,
        "tokenizer": FakeTokenizer(),
        "Dataset": Dataset,
        "DEMO_MODE": True,
        "EVAL_ROW_CAP": 256,
        "MODEL_COMMIT": "c" * 40,
    }
    exec(generator.TOOLS_CELL, namespace_03)
    exec(code_cell_containing(notebook_03, "def demo_rows()"), namespace_03)
    assert len(namespace_03["train_dataset"]) == 1
    assert len(namespace_03["eval_dataset"]) == 1

    # A rerun whose corpus or schedule changed must not resume the previous
    # run's checkpoints: the directory is keyed by what decides the schedule.
    args_cell = code_cell_containing(notebook_03, "training_args = SFTConfig(")
    assert "output_dir=str(SFT_RUN_DIR)" in args_cell
    for key in ("dataset_revision", "train_rows", "num_train_epochs", "learning_rate"):
        assert f'"{key}":' in args_cell[args_cell.index("SFT_RUN_KEY = "):], key
    train_cell = code_cell_containing(notebook_03, "resume_from = latest_checkpoint(")
    assert "latest_checkpoint(SFT_RUN_DIR)" in train_cell
    for cell in notebook_03.cells:
        assert 'RUN_ROOT / "sft"' not in cell.source or "SFT_RUN_KEY" in cell.source

    # Two schedules must not share a directory, and one schedule must keep it.
    def run_key(**overrides):
        namespace = {
            "json": json, "Path": Path, "RUN_ROOT": Path("/tmp/qwen38-key-probe"),
            "DATASET_ID": "x/y", "DATASET_REVISION": "main", "MAX_SEQ_LENGTH": 8192,
            "NUM_TRAIN_EPOCHS": 1, "MAX_STEPS": -1, "LEARNING_RATE": 5e-5,
            "train_dataset": range(5760), "eval_dataset": range(256),
            "DATASET_COMMIT": "a" * 40, "MODEL_COMMIT": "c" * 40,
            "LORA_RANK": 32, "LORA_ALPHA": 64,
            "run_manifest": {}, **overrides,
        }
        body = args_cell[: args_cell.index("training_args = SFTConfig(")]
        exec(body, namespace)
        return namespace["SFT_RUN_KEY"]

    assert run_key() == run_key()
    assert run_key() != run_key(train_dataset=range(3207))
    assert run_key() != run_key(NUM_TRAIN_EPOCHS=2)
    # The source caps are hit exactly, so a changed converter republishes the
    # same number of different rows. Only the resolved commit tells them apart.
    assert run_key() != run_key(DATASET_COMMIT="b" * 40)
    # The base repository is mutable too; adapter state belongs to one base.
    assert run_key() != run_key(MODEL_COMMIT="d" * 40)
    # A rank change makes the old checkpoints the wrong shape entirely.
    assert run_key() != run_key(LORA_RANK=16, LORA_ALPHA=32)
    load_cell = code_cell_containing(notebook_03, "loaded = load_dataset(DATASET_ID")
    assert "DATASET_COMMIT = HfApi(token=hf_token).dataset_info(" in load_cell
    assert "load_dataset(DATASET_ID, revision=DATASET_COMMIT, token=hf_token)" in load_cell
    assert load_cell.index("DATASET_COMMIT = HfApi(") < load_cell.index("loaded = load_dataset(")

    # The eval split is capped so a bigger corpus cannot stretch the run:
    # every eval reads the whole split, and the split grows with the corpus.
    sft_config = code_cell_containing(notebook_03, "LEARNING_RATE = 5e-5")
    assert "EVAL_ROW_CAP = 256" in sft_config
    load_cell = code_cell_containing(notebook_03, "eval_dataset = eval_raw.map(render_row)")
    assert "shuffled = eval_raw.shuffle(seed=3407)" in load_cell
    assert load_cell.index("EVAL_ROW_CAP") < load_cell.index("eval_dataset = eval_raw.map(")

    # The cap keeps every lane the split went to the trouble of holding out.
    cap_namespace = {
        "EVAL_ROW_CAP": 8,
        "eval_raw": Dataset.from_list(
            [{"lane": "agentic", "n": i} for i in range(200)]
            + [{"lane": "non_agentic", "n": 900 + i} for i in range(3)]
        ),
        "train_raw": Dataset.from_list([{"lane": "agentic", "n": 0}]),
        "render_row": lambda row: {"text": str(row["n"])},
        "json": json,
    }
    exec(load_cell[load_cell.index("eval_rows_available = len(eval_raw)"):], cap_namespace)
    assert len(cap_namespace["eval_dataset"]) == 8
    assert set(cap_namespace["eval_dataset"]["lane"]) == {"agentic", "non_agentic"}


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
    # Nor to the observation's own text: a path an observation printed and the
    # assistant's next command names is masked correctly and reads as a leak.
    assert "OBSERVATION_MATCH_FLOOR" not in all_source
    sft_masking = code_cell_containing(generator.build_03_sft(), "def masking_problems(")
    assert 'OBSERVATION_TAG = "<tool_response>"' in sft_masking
    assert "if OBSERVATION_TAG in supervised:" in sft_masking
    assert "tool observation leaked into the loss" in sft_masking
    assert 'message["role"] == "tool"' not in sft_masking
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


class _FakeHfApi:
    """Enough of HfApi for the load cell to resolve a revision."""

    def __init__(self, token=None):
        self.token = token

    def model_info(self, repo_id, revision=None):
        return SimpleNamespace(sha="c" * 40)


def run_lora_discovery(cell: str) -> dict:
    namespace = {
        "json": json,
        "HfApi": _FakeHfApi,
        "MODEL_REVISION": "main",
        "run_manifest": {},
        "LORA_RANK": 32,
        "LORA_ALPHA": 64,
        "torch": _FakeTorch,
        "FastModel": _FakeFastModel,
        "require_free_vram": lambda *_: 90.0,
        "assert_model_fully_resident": lambda *_, **__: None,
        "MODEL_ID": "unsloth/Qwen3.8-27B",
        "MERGED_SFT_MODEL_ID": "user/merged",
        "MERGED_SFT_REVISION": "main",
        "MERGED_SFT_COMMIT": None,
        "SMOKE_MODEL_ID": "unsloth/Qwen3.8-27B",
        "DEMO_MODE": False,
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


def test_notebook_02_accepts_a_segmented_trajectory_row():
    """A segment carries an elision note as a second user turn, so the
    validator must not require the roles to alternate. Rows that fail here
    fail at the top of a six-hour run, after the corpus is already built."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qwen3_8_27b_code import public_sources as ps

    generator = load_generator()
    namespace = {"json": json, "raw_dataset": []}
    exec(generator.TOOLS_CELL, namespace)
    exec(code_cell_containing(generator.build_02_data(), "def validate_row"), namespace)
    validate_row = namespace["validate_row"]

    # The source roles, as Open-SWE spells them: system, and the bash tool.
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the failing test. " + "context " * 40},
    ]
    for index in range(24):
        messages.append({
            "role": "assistant",
            "content": f"Step {index}.",
            "reasoning_content": f"Reasoning {index}. " * 30,
            "tool_calls": [{"type": "function", "id": "c", "function": {
                "name": "bash",
                "arguments": json.dumps({"command": f"pytest tests/test_{index}.py"}),
            }}],
        })
        messages.append({"role": "tool", "content": f"output {index} " + "y" * 1_500})

    rows = ps.convert_open_swe_row(
        {"resolved": 1, "messages": messages, "repo": "acme/widget", "trajectory_id": "t1"},
        budget_tokens=6_000,
    )
    kinds = [row["verification"]["window"] for row in rows]
    assert kinds[0] == "head" and "segment" in kinds, kinds
    for row in rows:
        row = dict(row, tool_schema_version=namespace["TOOL_SCHEMA_VERSION"],
                   tool_schema_json=namespace["TOOL_SCHEMA_JSON"], tools=namespace["TOOLS"])
        assert validate_row(row) == [], (row["id"], validate_row(row))


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

    # An answer nothing executed is admitted only when it says so, and only
    # to the non-agentic lane; a claimed failure or a bare omission is not.
    unverified = dict(reasoning_row, verification={"all_required_tests_pass": None, "runner": "none"})
    assert validate_row(unverified) == []
    assert validate_row(dict(unverified, verification={"all_required_tests_pass": None}))
    assert validate_row(dict(unverified, verification={"all_required_tests_pass": False, "runner": "none"}))
    assert validate_row(dict(mislabelled, lane="agentic", verification={"all_required_tests_pass": None, "runner": "none"}))


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
        (generator.build_03_sft, "LEARNING_RATE = 5e-5", "training_args = SFTConfig("),
        (generator.build_04_dpo, "LENGTH_PAIRS_LOCAL_JSONL", "dpo_args = DPOConfig("),
    ):
        notebook = build()
        config_cell = code_cell_containing(notebook, config_marker)
        for demo_mode, expected in ((True, 1), (False, 50 if build is generator.build_03_sft else 10)):
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
    sft_config = code_cell_containing(generator.build_03_sft(), "LEARNING_RATE = 5e-5")
    assert "LEARNING_RATE = 5e-5" in sft_config

    # The run inputs a checkpoint must be attributed to are in its manifest.
    for cell, keys in (
        (
            sft_config,
            (
                "learning_rate", "num_train_epochs", "eval_every_steps", "save_every_steps",
                "optimizer", "lora_rank", "lora_alpha",
            ),
        ),
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
    # Beside the trainer's output directory, not inside it; see
    # test_training_artefacts_are_saved_outside_the_trainer_output_directory.
    assert '(FINAL_DIR / "run_manifest.json").write_text' in dpo_train
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
    assert guarded >= 11

    # Where the publish follows an expensive job, an existing public target
    # is found at configuration time, not after the GPU or the teacher bill.
    for name, marker, guard in (
        ("03", "LEARNING_RATE = 5e-5", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("04", "LENGTH_PAIRS_LOCAL_JSONL", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("05", "ROLLOUT_POLICY_PRECISION = ", "require_private_repo(OUTPUT_ADAPTER_ID)"),
        ("06", "RUN_STANDARD_GGUF_EXPORT = False", "require_private_repo(QAT_OUTPUT_ID)"),
        ("06", "RUN_STANDARD_GGUF_EXPORT = False", "require_private_repo(GGUF_OUTPUT_ID)"),
        ("03", "LEARNING_RATE = 5e-5", "require_private_repo(MERGED_MODEL_ID)"),
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
        ("if RUN_BASELINE_EVAL:", "report_provenance(stock_model_ref)"),
        ("RUN_EFFORT_LADDER:", "report_provenance(stock_model_ref, reasoning_effort=effort)"),
        ("RUN_CANDIDATE_EVAL:", "report_provenance(candidate_model_ref)"),
    ):
        cell = code_cell_containing(notebook, marker)
        assert model_ref in cell
        assert cell.index(".metadata = report_provenance(") < cell.index("write_report(")
    assert "seeds=DEFAULT_SEEDS[:EVAL_ATTEMPTS]," in config_cell
    baseline_cell = code_cell_containing(notebook, "if RUN_BASELINE_EVAL:")
    assert "revision=MODEL_REVISION," in baseline_cell
    assert baseline_cell.index("baseline_report_path.unlink(missing_ok=True)") < baseline_cell.index("evaluate(")
    candidate_cell = code_cell_containing(notebook, "RUN_CANDIDATE_EVAL:")
    # The candidate commit is pinned in the configuration cell, before the
    # evaluation, and the same commit is loaded and recorded.
    assert "revision = resolved_revision(adapter_id, pinned)" in config_cell
    assert "ACCEPTED_ADAPTER_ID, CANDIDATE_REVISION = adapter_id, revision" in config_cell
    assert "revision=CANDIDATE_REVISION," in candidate_cell
    assert 'candidate_model_ref = f"{ACCEPTED_ADAPTER_ID}@{CANDIDATE_REVISION}"' in candidate_cell
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
    # Likewise an earlier acceptance: only a gate that passes now writes one.
    assert gate_cell.index("accepted_path.unlink(missing_ok=True)") < gate_cell.index("pairing_problems(")
    assert gate_cell.index("pairing_problems(") < gate_cell.index("accepted_path.write_text(")
    # A baseline measured this session outranks the pulled copy it replaced,
    # whatever name GATE_BASELINE_FILE selects.
    assert "baseline_candidates.append(baseline_report_path)" in gate_cell
    assert gate_cell.index("baseline_candidates.append(baseline_report_path)") < gate_cell.index(
        "baseline_candidates.append(HUB_REPORT_DIR / GATE_BASELINE_FILE)"
    )

    persist_cell = code_cell_containing(notebook, "create_repo(GATE_REPORTS_REPO")
    assert "private=True" in persist_cell
    assert "exist_ok=True" in persist_cell
    assert "folder_path=str(REPORT_DIR)" in persist_cell
    assert "repo_revision" in persist_cell
    # A remote verdict or acceptance is deleted unless this session's copy replaces it.
    assert 'delete_patterns=["comparison.json", "accepted.json"]' in persist_cell
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

    def walk(scope, deferred: bool) -> None:
        # A class body runs when its cell does; a function or lambda body
        # runs when called, and so does anything nested inside one.
        deferred = deferred or scope.get_type() != "class"
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                (nested_reads if deferred else top_reads).add(symbol.get_name())
        for child in scope.get_children():
            walk(child, deferred)

    for child in table.get_children():
        walk(child, False)
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
    # A class body runs with its cell; a method body does not.
    assert undefined_notebook_names(["class C:\n    digest = hashlib.sha256(b'x')", "import hashlib"]) == [
        "cell 0: hashlib"
    ]
    assert undefined_notebook_names(["class C:\n    def run(self):\n        return tokenizer", "tokenizer = 1"]) == []


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


def test_fixture_rows_are_refused_at_publish_and_at_training():
    """Flipping DEMO_MODE and rerunning only the publish cell once pushed
    the two-row fixture as the corpus. The guards read the rows, not the flag."""
    generator = load_generator()
    publish_cell = code_cell_containing(generator.build_02_data(), "PUSH_DATASET = True")
    assert 'str(row_id).startswith("fixture/")' in publish_cell
    assert "if DEMO_MODE:\n    PUSH_DATASET = False" in publish_cell
    assert "if fixture_rows:" in publish_cell
    assert publish_cell.index("if fixture_rows:") < publish_cell.index("require_private_repo(")
    # Rows without an id are valid; the scan tolerates a missing column.
    assert '"id" in split.column_names' in publish_cell
    load_cell = code_cell_containing(generator.build_03_sft(), "loaded = load_dataset(DATASET_ID")
    assert 'str(row_id).startswith("fixture/")' in load_cell
    assert '"id" in loaded[split].column_names' in load_cell
    assert "Rerun notebook 02 with DEMO_MODE=False" in load_cell
    # The fixture rows really are marked that way.
    demo_cell = code_cell_containing(generator.build_02_data(), "raw_dataset = Dataset.from_list(demo_rows)")
    assert demo_cell.count('"id": "fixture/') >= 2


def test_training_artefacts_are_saved_outside_the_trainer_output_directory():
    """Everything left in the trainer's output directory is swept into
    trainer.push_to_hub(): a local copy of the adapter saved there is published
    a second time under its own subdirectory, and a completion marker written
    there rides the same commit as the weights it is meant to follow."""
    generator = load_generator()
    stages = (
        ("03", generator.build_03_sft, "SFT_RUN_DIR"),
        ("04", generator.build_04_dpo, 'RUN_ROOT / "dpo"'),
        ("05", generator.build_05_grpo, "grpo_root"),
    )
    for name, build, run_dir in stages:
        cells = [cell.source for cell in build().cells if cell.cell_type == "code"]
        assert any(f"output_dir=str({run_dir})" in cell for cell in cells), name
        saves = [
            line.strip() for cell in cells for line in cell.splitlines()
            if "save_model(" in line or "run_manifest.json\").write_text" in line
            or "path_or_fileobj=" in line and "run_manifest" in line
        ]
        assert saves, name
        for line in saves:
            assert run_dir not in line, (name, line)
            assert "final_adapter" not in line, (name, line)


def test_dpo_records_the_shape_of_the_adapter_it_trains():
    """Notebook 04 trains a fresh adapter over the merged SFT weights, so its
    rank is its own. As a bare literal beside the model it never reached the
    manifest, and two adapters of different shapes looked alike in their own
    provenance records."""
    generator = load_generator()
    cells = [cell.source for cell in generator.build_04_dpo().cells if cell.cell_type == "code"]
    config = next(cell for cell in cells if "DPO_BETA = " in cell)
    assert "LORA_RANK = 16" in config
    assert "LORA_ALPHA = 2 * LORA_RANK" in config
    assert '"lora_rank": LORA_RANK' in config and '"lora_alpha": LORA_ALPHA' in config
    load = next(cell for cell in cells if "FastModel.get_peft_model(" in cell)
    assert "r=LORA_RANK," in load and "lora_alpha=LORA_ALPHA," in load
    assert "r=16," not in load and "lora_alpha=32," not in load


def test_every_notebook_checks_the_repository_out_the_same_way():
    """Four notebooks check this repository out, and the bugs found in one were
    always still in the other three. Assert it is literally one block."""
    generator = load_generator()
    builders = (
        ("02", generator.build_02_data), ("04", generator.build_04_dpo),
        ("07", generator.build_07_collect_and_evaluate), ("08", generator.build_08_distil),
    )
    blocks = {}
    for name, build in builders:
        cell = code_cell_containing(build(), 'REPO_DIR = Path("/content/qwen3.8-27B-code")')
        start = cell.index('REPO_URL = "https://github.com/')
        end = cell.index('subprocess.run(["git", "reset", "--hard", "FETCH_HEAD"], cwd=REPO_DIR, check=True)')
        blocks[name] = cell[start:end]
    assert len(set(blocks.values())) == 1, sorted(blocks)

    block = blocks["02"]
    # A cold runtime honours REPO_REVISION too: a clone would take the remote's
    # default branch, so the two paths would train from different data.
    assert '["git", "clone"' not in block
    assert '["git", "init", "-q", str(REPO_DIR)], check=True' in block
    assert '["git", "fetch", "--depth", "1", "origin", REPO_REVISION], cwd=REPO_DIR, check=True' in block
    # origin is set on every run, not only when the directory is new: a
    # checkout left by an earlier REPO_URL would otherwise be fetched from.
    top_level = [ast.unparse(node) for node in ast.parse(block).body]
    remote_calls = [statement for statement in top_level if "'remote'" in statement]
    assert len(remote_calls) == 2, top_level
    assert "'remote', 'remove', 'origin'" in remote_calls[0] and "check=False" in remote_calls[0]
    assert "'remote', 'add', 'origin', REPO_URL" in remote_calls[1] and "check=True" in remote_calls[1]


def test_notebooks_run_the_real_pipeline_as_shipped():
    """Open, Run all: no demo default, no publish flag to flip, no placeholder
    to fill in. Notebook 07 decides from the Hub what a session needs."""
    generator = load_generator()
    data_config = code_cell_containing(generator.build_02_data(), "SOURCE_LOCAL_JSONL")
    assert "DEMO_MODE = False" in data_config
    assert 'SOURCE_LOCAL_JSONL = str(REPO_DIR / "data" / "native_sft" / "trajectories.jsonl")' in data_config
    assert "PUSH_DATASET = True" in code_cell_containing(generator.build_02_data(), "PUSH_DATASET = ")
    # The package a stale checkout holds is dropped from sys.modules before the
    # import, or the refreshed checkout on disk is not the code that runs.
    assert 'name.split(".")[0] == "qwen3_8_27b_code"' in data_config
    for name, build in (
        ("04", generator.build_04_dpo), ("07", generator.build_07_collect_and_evaluate),
        ("08", generator.build_08_distil),
    ):
        joined = "\n".join(cell.source for cell in build().cells)
        if "qwen3_8_27b_code" in joined:
            assert 'name.split(".")[0] == "qwen3_8_27b_code"' in joined, name
    assert data_config.index("del sys.modules[module_name]") < data_config.index(
        "from qwen3_8_27b_code.public_sources import"
    )

    sft_config = code_cell_containing(generator.build_03_sft(), "LEARNING_RATE = 5e-5")
    for line in (
        "DEMO_MODE = False", "RUN_TRAINING = True", "PUSH_ADAPTER = True", "PUSH_MERGED_SFT = True",
        "NUM_TRAIN_EPOCHS = 1", "MAX_STEPS = -1",
    ):
        assert line in sft_config, line
    # Completion markers go up after the weights: the adapter's after its final
    # push, the merged checkpoint's after the merge, then the history is squashed.
    train_cell = code_cell_containing(generator.build_03_sft(), 'commit_message="SFT adapter')
    assert train_cell.index("trainer.push_to_hub(") < train_cell.index('path_in_repo="run_manifest.json"')
    merge_at = train_cell.index("push_to_hub_merged(")
    assert train_cell.index('path_in_repo="run_manifest.json"') < merge_at
    assert merge_at < train_cell.index('commit_message="run manifest: merge completed"') < train_cell.index(
        "super_squash_history("
    )
    # Demo mode clamps rather than raising, so a smoke needs one flag.
    assert "MAX_STEPS, PUSH_ADAPTER, PUSH_MERGED_SFT = 2, False, False" in sft_config
    # The rank is a configured lever, and alpha tracks it so that raising the
    # rank changes capacity without also changing the update scaling.
    assert "LORA_RANK = 32" in sft_config
    # The card reports about 89.4 GiB and the rank-16 run peaked at 88.5, so a
    # rank whose extra state exceeds that headroom cannot be the shipped default.
    assert "88.5 GiB" in sft_config and "89.4 GiB" in sft_config
    assert "LORA_ALPHA = 2 * LORA_RANK" in sft_config
    peft_cell = code_cell_containing(generator.build_03_sft(), "FastModel.get_peft_model(")
    assert "r=LORA_RANK," in peft_cell and "lora_alpha=LORA_ALPHA," in peft_cell
    assert "r=16," not in peft_cell

    dpo_config = code_cell_containing(generator.build_04_dpo(), "LENGTH_PAIRS_LOCAL_JSONL")
    for line in (
        "DEMO_MODE = False", "RUN_TRAINING = True", "PUSH_ADAPTER = True", "NUM_TRAIN_EPOCHS = 2", "MAX_STEPS = -1",
    ):
        assert line in dpo_config, line
    assert 'PREFERENCE_LOCAL_JSONL = str(REPO_DIR / "data" / "preferences" / "pairs.jsonl")' in dpo_config
    assert 'MERGED_SFT_MODEL_ID = f"{HF_USERNAME}/qwen38-27b-code-sft-merged"' in dpo_config
    dpo_train = code_cell_containing(generator.build_04_dpo(), 'commit_message="DPO adapter')
    assert dpo_train.index("hub.delete_file(") < dpo_train.index("trainer.train()")
    assert dpo_train.index("trainer.push_to_hub(") < dpo_train.index('path_in_repo="run_manifest.json"')
    assert 'file_exists(\n                        MERGED_SFT_MODEL_ID, "run_manifest.json"' in dpo_config.replace(
        "\n    ", "\n                    "
    ) or '"run_manifest.json", revision=MERGED_SFT_COMMIT' in dpo_config
    # The commit is resolved once and used for the marker check, the load
    # and the manifest, so all three name the same weights.
    assert dpo_config.index("MERGED_SFT_COMMIT = api.repo_info(") < dpo_config.index(
        'file_exists(MERGED_SFT_MODEL_ID, "run_manifest.json", revision=MERGED_SFT_COMMIT)'
    )
    assert '"model_commit": MERGED_SFT_COMMIT' in dpo_config
    # The smoke loads the stock model, so it needs nothing published.
    dpo_load = code_cell_containing(generator.build_04_dpo(), "model_name=SMOKE_MODEL_ID if DEMO_MODE else MERGED_SFT_MODEL_ID")
    assert "revision=MERGED_SFT_COMMIT," in dpo_load

    gate_config = code_cell_containing(generator.build_07_collect_and_evaluate(), "GATE_REPORTS_REPO =")
    assert "EVAL_ATTEMPTS = 2" in gate_config
    assert "PUSH_ARTIFACTS = True" in gate_config
    assert "RUN_BASELINE_EVAL = None" in gate_config
    assert "RUN_CANDIDATE_EVAL = None" in gate_config
    # A pulled baseline stands in only if it was measured the way this
    # session measures; otherwise the candidate would be refused at the gate.
    assert "RUN_BASELINE_EVAL = bool(mismatches)" in gate_config
    assert "read_report(pulled_baseline).metadata, report_provenance(stock_model_ref)" in gate_config
    # The decision keys on the completion marker at the pinned commit.
    assert 'api.file_exists(adapter_id, "run_manifest.json", revision=revision)' in gate_config
    # Notebook 03 removes an earlier marker before its first push, so an
    # intermediate checkpoint never inherits one.
    sft_config = code_cell_containing(generator.build_03_sft(), "LEARNING_RATE = 5e-5")
    # ...and only once training is certain to start: a dry run or an early
    # failure must leave a valid adapter's marker alone.
    assert "hub.delete_file(" not in sft_config
    train_cell = code_cell_containing(generator.build_03_sft(), 'commit_message="SFT adapter')
    assert '"run_manifest.json", OUTPUT_ADAPTER_ID,' in train_cell
    # A new SFT run supersedes the DPO adapter trained on the previous merge,
    # so its marker goes at the same moment and 07 gates the SFT adapter.
    assert train_cell.index('"run_manifest.json", DPO_ADAPTER_ID,') < train_cell.index("trainer.train(")
    assert train_cell.index("if RUN_TRAINING:") < train_cell.index("hub.delete_file(") < train_cell.index("trainer.train(")
    assert "RUN_CANDIDATE_EVAL = CANDIDATE_REVISION is not None" in gate_config
    # A baseline left by an earlier run in this runtime cannot shadow the pulled one.
    assert "(REPORT_DIR / GATE_BASELINE_FILE).unlink(missing_ok=True)" in gate_config
    assert gate_config.index("RUN_BASELINE_EVAL = bool(mismatches)") < gate_config.index(
        "(REPORT_DIR / GATE_BASELINE_FILE).unlink(missing_ok=True)"
    )
    # The decision comes after the pull and the provenance helper that inform it.
    assert gate_config.index("snapshot_download(") < gate_config.index("RUN_BASELINE_EVAL = bool(mismatches)")
    assert gate_config.index("def report_provenance(") < gate_config.index("RUN_BASELINE_EVAL = bool(mismatches)")
    # The gate records acceptance with the reports; it publishes no weights,
    # and it gates the latest finished stage.
    assert "GATE_LATEST_STAGE = True" in gate_config
    # Each candidate repo resolves its own pin; a commit of one repo is not a commit of the other.
    assert "((DPO_ADAPTER_ID, DPO_REVISION), (ACCEPTED_ADAPTER_ID, ACCEPTED_REVISION))" in gate_config
    assert "for adapter_id, pinned in candidates:" in gate_config
    for cell in generator.build_07_collect_and_evaluate().cells:
        assert "push_to_hub_merged(" not in cell.source
    accept_cell = code_cell_containing(generator.build_07_collect_and_evaluate(), '"accepted.json"')
    assert 'if comparison["gate_passed"]:' in accept_cell
    assert '"adapter": candidate_model_ref' in accept_cell

    for name, build in (
        ("02", generator.build_02_data), ("03", generator.build_03_sft), ("04", generator.build_04_dpo),
        ("05", generator.build_05_grpo), ("06", generator.build_06_qat_export),
        ("07", generator.build_07_collect_and_evaluate),
    ):
        for cell in build().cells:
            assert "REPLACE_WITH_" not in cell.source, name


def test_notebook_02_streams_the_public_sources_into_the_corpus():
    generator = load_generator()
    config_cell = code_cell_containing(generator.build_02_data(), "PUBLIC_SOURCES = {")
    assert "from qwen3_8_27b_code.public_sources import" in config_cell
    assert config_cell.index('sys.path.insert(0, str(REPO_DIR / "src"))') < config_cell.index("from qwen3_8_27b_code.public_sources")
    for name in ("SOURCE_OPEN_SWE: 3_000", "SOURCE_OPEN_CODE_INSTRUCT: 2_000", "SOURCE_OPEN_CODE_REASONING: 1_200"):
        assert name in config_cell, name
    assert "PUBLIC_TOKEN_BUDGET = 6_000" in config_cell
    load_cell = code_cell_containing(generator.build_02_data(), "raw_dataset = Dataset.from_list(demo_rows)")
    assert "collect_public_rows(" in load_cell
    assert "count=count_tokens, token=hf_token" in load_cell
    # One Dataset from plain rows, so Arrow infers one schema across sources.
    # Every row carries every column before the table is built, or the
    # first row (a bootstrap row, no lane) would decide the columns.
    assert "raw_dataset = Dataset.from_list(unify_columns(rows))" in load_cell
    assert "concatenate_datasets" not in load_cell
    # The pinned source commits and counts travel with the published corpus,
    # and the report is this run's: the loading cell resets it first.
    assert load_cell.index("public_report = None") < load_cell.index("if DEMO_MODE:")
    publish_cell = code_cell_containing(generator.build_02_data(), "dataset_dict.push_to_hub(")
    assert 'path_in_repo="public_sources.json"' in publish_cell
    assert "if public_report:" in publish_cell
    assert publish_cell.index("dataset_dict.push_to_hub(") < publish_cell.index('path_in_repo="public_sources.json"')
    # The windowed rows fit the training window with the rendered overhead.
    sft_config = code_cell_containing(generator.build_03_sft(), "LEARNING_RATE = 5e-5")
    assert "MAX_SEQ_LENGTH = 8_192" in sft_config


def test_notebook_01_writes_the_hidden_verifier_only_when_it_runs():
    generator = load_generator()
    notebook = generator.build_01_baseline()
    task_cell = code_cell_containing(notebook, "def make_demo_task")
    assert "hidden.write_text(" not in task_cell
    assert "hidden_test_source=hidden_source" in task_cell
    episode_cell = code_cell_containing(notebook, "task.hidden_test_command,")
    assert episode_cell.index("hidden_path.write_text(task.hidden_test_source)") < episode_cell.index(
        "task.hidden_test_command,"
    )
    assert "hidden_path.unlink(missing_ok=True)" in episode_cell
