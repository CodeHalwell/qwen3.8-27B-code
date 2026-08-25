# Colab notebook suite

These notebooks turn the project plan into a deliberately gated execution path
for a Google Colab **G4** runtime (RTX PRO 6000 Blackwell Server Edition,
96 GB). They use trainable safetensors from `unsloth/Qwen3.8-27B`; the GGUF
repository is an inference and quantisation baseline, not a training source.

The whole suite optimises one objective: agentic coding through the native
six-tool schema. Every training stage — SFT, DPO and the GRPO pilot — consumes
only execution-verified software-engineering data; there is no general-chat
replay slice, and drift on non-coding chat is an accepted trade rather than a
gate (see the [specialisation policy](../docs/data-strategy.md#specialisation-policy)).

## Run order

| Notebook | Purpose | Required gate before continuing |
| --- | --- | --- |
| [00 · Colab preflight](00_colab_preflight.ipynb) | Pin the day-zero stack, authenticate, load BF16, validate the native Qwen3.8 template and measure memory | BF16 load, XML tool template and bounded generation all pass |
| [01 · Tool-calling baseline](01_tool_calling_baseline.ipynb) | Run the upstream model through the exact six-tool surface and price a small evaluation gate | Manually classify every pilot failure and record episode cost |
| [02 · Prepare SFT data](02_prepare_sft_data.ipynb) | Validate, render, measure, split and optionally publish native-schema trajectories | 100–300 replayable native trajectories; frozen repository-family split |
| [03 · SFT LoRA](03_sft_lora.ipynb) | Train a language-only BF16 LoRA with assistant-only loss | Held-out coding and tool metrics beat the frozen baseline |
| [04 · DPO preferences](04_dpo_preferences.ipynb) | Apply a small verifier-backed preference stage | DPO beats the accepted SFT adapter without coding, tool-protocol or preserved-thinking regressions |
| [05 · Agentic GRPO](05_agentic_grpo.ipynb) | Validate a stateful coding environment and reward tests; trainer integration is compatibility-gated | Reward fixtures pass; then resolve the recorded Unsloth/TRL blocker before any policy update |
| [06 · QAT and export](06_qat_and_export.ipynb) | Create separate QAT/TorchAO and standard GGUF experiments; prepare Dynamic calibration data | Quantised artifacts pass the frozen long-horizon gate against one BF16 reference |

## Colab setup

1. Select the **G4** GPU runtime.
2. Add a write-capable `HF_TOKEN` in Colab Secrets and grant the notebook access.
3. Open notebook 00 and run the install cell once.
4. Restart the runtime when instructed, then rerun the notebook from the top.
   The pinned install marker makes the repeated install cell a cheap no-op.
5. Record immutable Hugging Face commit revisions before any non-demo run.

The install cell keeps a detailed log at
`/content/qwen38_pip_install.log` and prints its final 120 lines if a phase
fails. The reviewed core matrix is Transformers 5.3.0, TRL 0.22.2, Datasets
4.3.0 and PEFT 0.19.0, plus TorchAO/xFormers selected from the preinstalled
Colab PyTorch minor. This matches the current Unsloth dependency bounds and
the adjacent official Qwen3.5 27B example.

Every expensive or externally persistent operation is off by default behind a
`RUN_*`, `PUSH_*`, `SAVE_*` or `BUILD_*` flag. The included fixture rows prove
plumbing only; they are explicitly not training data for a capability run.

## Design choices inherited from the Unsloth examples

- Unsloth is imported before Transformers/TRL where model patching is needed.
- LoRA starts at rank 16 with attention and MLP projections, Unsloth gradient
  checkpointing, BF16 compute and an 8-bit optimizer.
- SFT trains assistant spans only and inspects a collated label batch before
  optimization. Native rows carry both the tool objects and a canonical schema
  fingerprint; Arrow-inserted null struct fields are removed before structural
  validation and rendering.
- The GRPO notebook unit-tests the environment and reward surface first. Its
  stateful trainer requires TRL's `environment_factory` (introduced after the
  maximum TRL version currently declared by the pinned Unsloth Zoo), so policy
  updates remain disabled until that compatibility gate is resolved.
- QAT uses Unsloth's `qat_scheme="int4"`, performs TorchAO's post-training
  `QATConfig(step="convert")`, and saves through `save_pretrained_torchao`.
- GGUF exports remain a separate post-training branch.

The current Unsloth catalog has adjacent Qwen3/Qwen3.5 SFT, GRPO and QAT
examples, but no dedicated Qwen3.8-27B fine-tuning notebook was available when
this suite was generated. Notebook 00 is therefore a real compatibility gate,
not ceremony.

## Validation status

The repository generator validates notebook structure and parses every Python
cell. The contract tests also execute the Arrow-round-tripped demo path for
notebooks 02 and 03, the search fallback, split construction, and reward guard:

```bash
uv run --group dev python scripts/build_notebooks.py
uv run --group dev pytest -q
```

The model-loading, CUDA, training and export paths cannot be executed on the
local development machine. Their first runtime validation must happen on the
target Colab G4 in notebook order. Do not enable a real training flag until the
preceding dry-run cells and gate have passed.
