# Qwen3.8-27B coding-specialisation project

This directory is the design and execution guide for specialising
`Qwen3.8-27B` for tool-using, long-horizon software-engineering work on a
single Google Colab G4 runtime backed by an NVIDIA RTX PRO 6000 Blackwell
Server Edition.

The recommended path is:

```text
Qwen3.8-27B BF16
        |
        v
LoRA SFT on execution-verified coding trajectories
        |
        v
Preference training on chosen/rejected trajectories
        |
        v
Constrained multi-turn GRPO/GSPO
        |
        v
Merged BF16 reference checkpoint
        |-----------------------------|
        v                             v
NVFP4 / INT4 QAT experiments    Dynamic GGUF 4/3/2-bit
                                      |
                                      v
                              1-bit research only
```

Do not fine-tune the GGUF repository. Use the trainable safetensors from
[`unsloth/Qwen3.8-27B`](https://huggingface.co/unsloth/Qwen3.8-27B), merge the
resulting adapter into a BF16 checkpoint, and create deployment formats from
that checkpoint. The existing
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
is an inference baseline and a reference for supported quant types.

## Documentation map

| Document | Purpose |
| --- | --- |
| [Colab notebooks](../notebooks/README.md) | Executable preflight, baseline, data, SFT, DPO, agentic GRPO and quantisation suite |
| [Minimum path](minimum-path.md) | The deliberately small route to the first useful baseline, including what is adopted, built and deferred |
| [Model and hardware](model-and-hardware.md) | Model identity, architecture, VRAM feasibility, context constraints and the GPU assumption |
| [Agentic harness](agentic-harness.md) | Tool protocol, sandbox, episode lifecycle and trajectory format |
| [Data strategy](data-strategy.md) | SFT, preference and RL datasets; validation, mixing and contamination controls |
| [Training plan](training-plan.md) | Baseline, SFT, preference optimisation and agentic RL runbook |
| [Quantisation](quantisation.md) | QAT, NVFP4 and Dynamic 4/3/2/1-bit experiment branches |
| [Evaluation](evaluation.md) | Benchmarks, agent metrics, quantisation comparisons and stage gates |
| [Roadmap](roadmap.md) | Ordered implementation milestones, current executable slice, future layout and risks |
| [References](references.md) | Official sources and notes on the examples already in this repository |

## Decisions already made

1. **Specialise behaviour, not architecture.** Train text/tool behaviour while
   freezing the vision stack. Keep the original architecture and tokenizer so
   that merging and export remain compatible.
2. **Use LoRA first.** Full-parameter training is out of scope for one 96 GB
   GPU. BF16 LoRA is the preferred starting point; QLoRA is a fallback if the
   measured context or QAT configuration does not fit.
3. **Adopt the environment; own the model interface.** Use Harbor for task,
   sandbox and verifier orchestration, while keeping a thin project-owned
   Qwen-native tool adapter. Baselines, native-trace collection, RL and
   evaluation share that adapter and schema.
4. **Optimise verified outcomes.** Tests, compilation and regression results
   dominate rewards. Style and verbosity are secondary.
5. **Earn longer horizons.** Begin with short, deterministic episodes and
   increase the tool-call and context budgets only after reliability gates pass.
6. **Keep a BF16 golden checkpoint.** Every quantised artifact is compared to
   the same merged BF16 model.
7. **Separate QAT from Dynamic GGUF.** Current Unsloth/TorchAO QAT is a 4-bit
   deployment path. Dynamic 1/2/3-bit GGUF is a separate post-training
   quantisation path.
8. **Target 3- or 4-bit first.** Two-bit is experimental. One-bit for this
   dense 27B model is a research question, not a release target.

## Definition of success

The project succeeds when the tuned model improves held-out repository-task
completion over the stock model, retains general coding quality, produces
valid native tool calls, recovers from tool failures and preserves most of
those gains after the selected deployment quantisation.

Training loss alone is not a success criterion.

## Scope and assumptions

- Hardware is assumed to be a **Google Colab G4 runtime using the RTX PRO 6000
  Blackwell Server Edition with 96 GB GDDR7 ECC**.
- The primary workload is text-only software engineering. Vision weights are
  preserved but frozen.
- The native context limit is 262,144 tokens, but initial training will use
  4K and 8K sequences. Native maximum context does not imply that training at
  that length is practical on one GPU.
- Public/untrusted tasks run inside disposable, resource-limited sandboxes. The
  first reviewed-task pilot may use the explicitly insecure `trusted-dev`
  fallback described in [Minimum path](minimum-path.md).
- Colab storage and sessions are treated as ephemeral. Checkpoints, manifests
  and irreplaceable traces must be persisted to Hugging Face Hub or Drive.
- Source and benchmark licences must be reviewed before publishing data or
  model artifacts.

The model and its ecosystem are very new. Version pins, model-card facts and
runtime support should be rechecked before each major training or export run.
These documents were last reviewed on **2026-08-17**.
