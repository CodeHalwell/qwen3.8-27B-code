# Model and hardware

## Model identity

Use three distinct artifact classes:

| Purpose | Artifact | Notes |
| --- | --- | --- |
| Upstream reference | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) | Official post-trained checkpoint and model card |
| Fine-tuning source | [`unsloth/Qwen3.8-27B`](https://huggingface.co/unsloth/Qwen3.8-27B) | Trainable safetensors; roughly 56 GB in BF16 |
| Inference baseline | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | Existing Dynamic V3.0 preview GGUFs; do not use as the training source |

Every training record should contain the exact Hub revision, not only a model
name. The upstream files may change while support for this recently released
architecture settles.

## Published coding baseline

The official model card currently reports the following Qwen3.8-27B results:

| Benchmark | Published score |
| --- | ---: |
| Terminal Bench 2.1 (Terminus) | 73.0 |
| SWE-bench Pro | 61.7 |
| NL2Repo-Bench | 42.3 |
| DeepSWE 1.1 | 42.2 |
| QwenSWEBench | 79.0 |
| LiveCodeBench v6 | 90.3 |

These numbers establish that the upstream model is already a strong coding
model and that regression risk is real. They are not substitutes for this
project's baseline: several use a Claude Code harness, long context and
benchmark-specific settings, while QwenSWEBench is an in-house benchmark.
Reproduce results in the project's own harness before attributing a change to
fine-tuning.

## Relevant architecture facts

The official configuration identifies a multimodal
`Qwen3_5ForConditionalGeneration` model with a `qwen3_5` model type. The text
stack has:

- 64 layers and a 5,120-wide hidden state;
- a 248,320-token vocabulary;
- a repeating hybrid layout of three linear-attention layers followed by one
  full-attention layer;
- untied embeddings;
- one multi-token-prediction layer; and
- a native 262,144-token context limit.

The model also includes a vision tower. “Pure coding model” should mean a
behavioural specialisation, not removing the vision tensors. For initial work:

- use only text/tool examples;
- freeze all vision parameters;
- apply LoRA only to discovered language-model linear modules; and
- preserve the original processor, tokenizer, chat template and architecture
  in saved artifacts.

Do not copy a Qwen3.5 target-module list without checking the loaded Qwen3.8
module names. The hybrid layout means a standard transformer suffix list is
wrong here: only 16 of the 64 layers carry `self_attn.{q,k,v,o}_proj`, while
the other 48 are Gated DeltaNet layers whose projections are named
`linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and
`out_proj`. A list matching a bare `in_proj` matches none of them and leaves
three of every four layers without an adapter on their attention path.

The reviewed target set is therefore:

| Module group | Names | Layers |
| --- | --- | ---: |
| Full attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 16 |
| Gated DeltaNet | `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj` | 48 |
| Feed forward | `gate_proj`, `up_proj`, `down_proj` | 64 |

PEFT matches `target_modules` by name suffix, so the multi-token-prediction
head's own `q_proj`/`o_proj` are adapted unless they are explicitly frozen.
Exclude the vision tower, the `mtp.` prefix and `lm_head`, and fail the run if
discovery returns anything other than the reviewed set.

## Native conversation behaviour

Qwen3.8-27B supports hybrid thinking, developer messages, tool calling,
`reasoning_effort` (`xhigh`, `medium`, `low`) and preserved thinking. The
official default is thinking mode with `xhigh` effort.

Training and inference must use the tokenizer's native chat template. The
harness should pass declared tools to the template and retain structured
`developer`, `user`, `assistant` and `tool` roles. Hand-built prompt strings
are a compatibility risk.

The corpus should not contain only `xhigh` traces. Use:

- `low` for routine inspection and mechanical edits;
- `medium` for most debugging and implementation tasks; and
- `xhigh` for ambiguous, architectural or long-horizon work.

Preserved thinking can improve continuity but consumes context. Measure both
enabled and disabled modes in agent evaluation rather than baking one choice
into every task.

## Hardware assumption

The target is a Google Colab G4 runtime using the
[NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/):

- 96 GB GDDR7 ECC;
- 1,597 GB/s stated memory bandwidth;
- BF16/FP16, FP8 and FP4 Tensor Core support; and
- up to 600 W configurable power.

The 96 GB figure is the vendor's decimal capacity. PyTorch reports total
memory in binary GiB, so a nominal 96 GB card may appear as roughly 89.4 GiB.
The notebook preflight therefore requires at least 85 GiB of total memory:
low enough to accept the intended G4 card, but high enough to reject a nominal
48 GB card (roughly 44.7 GiB). This is a total-capacity check, not a check of
currently free memory.

Confirm the installed card before implementation:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Colab sessions and local storage are reclaimable. The notebook must write run
manifests, accepted datasets and checkpoints to Hugging Face Hub or Drive and
must support resuming after a clean runtime restart.

If the card is the 48 GB RTX 6000 Ada, BF16 LoRA is no longer the default.
Use 4-bit QLoRA, shorter sequences and possibly CPU offload; treat the rest of
this document's BF16-LoRA expectations as invalid.

## Single-GPU feasibility

| Workload | 96 GB Blackwell assessment | Starting point |
| --- | --- | --- |
| Full-parameter BF16 fine-tuning | Not feasible | Out of scope |
| BF16 LoRA SFT | Feasible with profiling | Batch 1, 4K then 8K, gradient checkpointing |
| 4-bit QLoRA SFT | Comfortable fallback | Use if context or QAT pressure requires it |
| DPO with adapters | Feasible but reference-model memory must be controlled | Adapter/ref-sharing or reference-free variant; 4K first |
| Short-horizon GRPO/GSPO | Feasible sequentially | Small groups and generations; use memory-efficient/standby mode if supported |
| Simultaneous long-context rollout and training | Poor fit | Alternate rollout collection and policy updates |
| 262K-context training | Not a first-phase target | Prove 4K/8K, then profile 16K and beyond |
| BF16 inference | Fits, with reduced room for KV cache | Golden evaluation only |
| NVFP4 or Dynamic GGUF inference | Strong fit | Preferred deployment experiments |

Unsloth's generic requirements place 27B LoRA at about 64 GB minimum and QLoRA
at about 22 GB. Those are planning numbers, not guarantees for Qwen3.8's large
vocabulary, hybrid architecture or multi-turn RL. Record measured peak
allocated and reserved VRAM for every smoke run.

## Inference memory guidance

The [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8) publishes the
following total-memory guidance for the 27B model:

| Quantisation | Approximate total memory |
| --- | ---: |
| 2-bit | 11–13 GB |
| 3-bit | 13–16 GB |
| 4-bit | 17–19 GB |
| 6-bit | 24 GB |
| 8-bit | 31 GB |
| BF16 | 56 GB |

These figures describe inference fit, not training requirements. Long-context
agent serving also needs KV-cache capacity, runtime workspaces and room for
concurrent requests. Quantising weights can therefore matter even when BF16
weights fit in 96 GB.

Blackwell makes
[`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
a serious deployment candidate. The current Unsloth guide also describes FP8
KV-cache calibration. Compare this branch against GGUF rather than assuming
that the smallest file is the fastest or most accurate server format.

## Context policy

Use these limits as a curriculum, subject to measured VRAM:

| Phase | Training sequence | Episode budget |
| --- | ---: | ---: |
| Pipeline smoke test | 4,096 | 2–4 tool calls |
| Initial SFT | 8,192 | 2–10 tool calls |
| Extended SFT/RL | 16,384 if profiling passes | 5–20 tool calls |
| Long-horizon evaluation | 32K–262K inference-only bands | 20–50+ tool calls |

Do not pad every example to the maximum. Bucket by token length and inspect the
full prompt-plus-completion distribution before selecting a limit. Truncating
from the right can remove final tool results or the successful patch, turning
valid examples into harmful supervision.

YaRN extension toward one million tokens is outside the first release. The
official model card warns that static YaRN can hurt shorter-context behaviour,
so enable it only in a separate evaluation configuration.

## Preflight checks

Before the first expensive run, capture:

1. GPU name, VRAM, driver, CUDA and PyTorch versions.
2. Exact model, tokenizer and processor revisions.
3. The loader class that the pinned Unsloth release supports for this
   multimodal architecture.
4. Trainable parameter names and count, proving that vision parameters are
   frozen.
5. One native tool-call round trip through the chat template.
6. A forward/backward pass at 4K with peak VRAM.
7. Save, reload and merge of a tiny LoRA checkpoint.
8. Inference parity before and after that no-op/smoke merge.

Failure of any preflight blocks a full SFT run.
