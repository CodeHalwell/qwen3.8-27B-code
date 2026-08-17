# References

Sources in this document are expected to change. Pin code and model revisions
in experiments even when a documentation URL is unversioned. Last checked:
**2026-08-17**.

## Model and inference

- [Official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) —
  architecture, reasoning modes, generation settings, context and upstream
  benchmark results.
- [Official Qwen3.8-27B configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) —
  model type, layer layout, vocabulary, dimensions and native context.
- [Unsloth trainable Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B) —
  safetensors source for LoRA fine-tuning.
- [Unsloth Qwen3.8-27B GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) —
  published inference quants and chat template; not a training source.
- [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8) — Dynamic
  V3.0 preview, memory guidance, tool/developer-role support, thinking modes,
  llama.cpp usage and NVFP4 information.
- [Unsloth Qwen3.8-27B NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) —
  Blackwell-specific deployment candidate.

## Training and RL

- [Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) —
  generic LoRA/QLoRA memory planning.
- [Unsloth memory-efficient RL](https://unsloth.ai/docs/basics/memory-efficient-rl) —
  standby and inference/training memory sharing concepts to compatibility-test.
- [Hugging Face TRL](https://huggingface.co/docs/trl) — SFT, DPO, GRPO and
  dataset-format APIs. Pin a tested TRL version rather than following `main`.
- [TRL 0.29 release](https://github.com/huggingface/trl/releases/tag/v0.29.0) —
  introduction of `environment_factory`; currently newer than the maximum TRL
  version declared by the pinned Unsloth Zoo package.
- [NVIDIA NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) — resource-server and
  multi-environment patterns for executable RL.

## Agent and evaluation harnesses

- [Harbor](https://github.com/harbor-framework/harbor) — adopted task,
  sandbox-backend, verifier and trial/job framework.
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) — simple
  bash-only control agent; useful for comparison but not native tool-call SFT
  traces.
- [SWE-bench evaluation harness](https://github.com/SWE-bench/SWE-bench) —
  official Docker/remote evaluation path for SWE-bench results.

## Quantisation

- [Unsloth QAT](https://unsloth.ai/docs/basics/quantization-aware-training-qat) —
  TorchAO fake quantisation, QAT + LoRA and currently documented 4-bit-oriented
  schemes.
- [Unsloth Dynamic 2.0 GGUF](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) —
  selective quantisation and evaluation methodology.
- [Dynamic GGUF on Aider Polyglot](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot) —
  coding-oriented evidence and limitations of naive ultra-low-bit quants.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF conversion,
  quantisation and portable runtime. Pin the converter/runtime commit together.

## Hardware

- [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) —
  official 96 GB memory, bandwidth, precision and power specifications.

## Candidate data

- [Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3)
- [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)

Each dataset still requires revision pinning, licence review, replay where
possible and contamination analysis.

## Evaluation projects

- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench)

Repository-agent benchmarks such as SWE-bench and terminal environments should
be added only with a reproducible harness and an explicit contamination policy.

## Local examples

The `references/` directory contains notebook exports. They are research
references, not production scripts.

| File | Useful content | Required adaptation |
| --- | --- | --- |
| `qwen_3_5_27b_a100(80gb).py` | BF16 27B load, LoRA, chat rendering, assistant-only loss, merge/GGUF patterns | Replace model/revision, validate Qwen3.8 loader and target modules, add configs/eval/tracking |
| `nemo_gym_multi_environment.py` | Resource servers and multi-environment GRPO structure | Replace small reasoning environments with sandboxed repositories and coding rewards |
| `qwen3_5_(4b)_vision_grpo.py` | GRPO/GSPO and reward-function examples | Text/tool environment, no vision tuning, execution rewards |
| `notebook24f5f9a990.ipynb` | Length analysis, reward validation and GRPO diagnostics | Extract concepts into tests/scripts; do not depend on notebook state |
| `qwen3_5_moe.py` | Qwen/Unsloth API patterns | MoE-specific settings do not apply to dense 27B target |
| `glm_flash_a100(80gb).py` | General large-model SFT workflow | Different architecture and template; use only generic operational ideas |
| `qwen3_(4b)_instruct_qat.py` | Current TorchAO dependency matching, fake-quant verification, QAT conversion and `save_pretrained_torchao` | Keep Qwen3.8 day-zero pins, freeze vision, validate 27B memory and treat TorchAO as separate from GGUF |
| `functiongemma_(270m).py` | Full tool-call/message examples, per-row tool schemas, tool-name normalization and response-only mask inspection | Retain Qwen3.8's native XML template and the fixed six-tool surface; do not translate FunctionGemma syntax |
| `gpt_oss_(20b)_500k_context_fine_tuning.py` | Long-context smoke-test and memory/runtime accounting patterns | `unsloth_tiled_mlp` and MXFP4 settings are model-specific; do not assume they work for Qwen3.8 |

Notebook shell magics, mutable dependency installs and Colab assumptions must
not be copied into the final training package.
