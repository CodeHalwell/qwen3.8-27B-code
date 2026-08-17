# Quantisation strategy

## Artifact flow

Quantisation begins only after a capability checkpoint passes evaluation.

```text
base BF16 + accepted LoRA adapter
                 |
                 v
        merged BF16 golden model
          |                    |
          v                    v
  TorchAO QAT branch     GGUF PTQ branch
  INT4 / FP8-INT4        Q8 -> Q6 -> Q4 -> Q3 -> Q2
          |                              |
          v                              v
 vLLM/Unsloth/etc.              llama.cpp-compatible runtime
```

Never overwrite the merged BF16 golden checkpoint. Every lower-precision model
must be reproducible from it plus a pinned conversion recipe.

## QAT and Dynamic GGUF are different

[Unsloth QAT](https://unsloth.ai/docs/basics/quantization-aware-training-qat)
currently uses TorchAO fake-quantisation operations during high-precision
training and documents schemes including `int4`, `int8-int4`, `fp8-int4` and
`fp8-fp8`. It supports LoRA and exports through TorchAO-oriented formats.

This is not a documented 1/2/3-bit GGUF QAT pipeline. Do not assume that
TorchAO's learned scales or accuracy recovery survive a later conversion to
llama.cpp GGUF.

[Unsloth Dynamic GGUF](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot)
is post-training quantisation with selective precision: important tensors stay
at higher precision while other tensors use lower-bit formats, guided by
calibration/importance data. This is the branch for 3- and 2-bit experiments.

## Branch A: Blackwell-native deployment

Evaluate the published Qwen3.8 NVFP4 model before custom QAT. It establishes
what a supported Blackwell-specific quant can achieve without further work.

Then evaluate a short QAT-LoRA recovery pass from the accepted capability
checkpoint:

1. Load the merged BF16 model using pinned Unsloth and TorchAO versions.
2. Apply the chosen supported QAT scheme to language layers only.
3. Train briefly on a balanced subset of verified code/tool trajectories.
4. Convert with the matching TorchAO configuration.
5. Serve in a runtime that explicitly supports the output architecture and
   format.
6. Compare against merged BF16, published NVFP4 and normal INT4 PTQ.

Start with `int4` or `fp8-int4`. QAT adds training complexity and is accepted
only if it measurably improves repository-agent success over ordinary PTQ.

## Branch B: Dynamic GGUF

Produce a ladder from the same BF16 model:

- BF16/F16 conversion sanity artifact;
- Q8 and Q6 near-reference artifacts;
- Dynamic 4-bit candidate;
- Dynamic 3-bit primary aggressive candidate; and
- Dynamic 2-bit experimental candidate.

Current Qwen3.8 guidance identifies `UD-Q4_K_XL` and `UD-Q3_K_XL` artifacts for
the 27B model. Pin the exact Unsloth/llama.cpp branch and quant type used by each
experiment; Dynamic V3.0 is currently described as a preview.

The importance-matrix corpus should represent deployment traffic:

- source code in target languages;
- diffs and patches;
- nested tool-call JSON;
- compiler and test output;
- repository trees and configuration files;
- successful and failed recovery sequences;
- `low`, `medium` and `xhigh` reasoning modes; and
- short and long context bands.

Do not calibrate only on Wikipedia or generic chat if the target is coding
agency.

## Format targets

The current Unsloth Qwen3.8 guide gives these approximate inference-memory
bands:

| Target | Approximate memory | Project position |
| --- | ---: | --- |
| BF16 | 56 GB | Golden reference, not compact deployment |
| Q8 | 31 GB | Near-reference diagnostic |
| Q6 | 24 GB | High-quality compact diagnostic |
| Dynamic 4-bit | 17–19 GB | Safe production candidate |
| Dynamic 3-bit | 13–16 GB | Primary aggressive candidate |
| Dynamic 2-bit | 11–13 GB | Experimental; strict agent evaluation |
| Dynamic 1-bit | No current 27B target documented | Research only |

Unsloth's new Qwen3.8 1-bit types are currently presented for the 2.4T model,
not the dense 27B model. Large MoE results do not establish that a dense 27B
coding agent will tolerate the same compression.

## Why long-horizon evaluation is stricter

Small per-turn degradation compounds. If the probability of making a sound
decision is `p` on each of 50 dependent turns, the rough all-correct probability
is `p^50`:

- `0.99^50` is about 61%;
- `0.98^50` is about 36%; and
- `0.97^50` is about 22%.

This is not a literal independence model of agent behaviour, but it explains
why perplexity or a short coding benchmark is insufficient for selecting a
2-bit agent model.

## Quantisation evaluation matrix

Run each candidate with the same harness, prompts, budgets and multiple seeds.
Measure:

- KL divergence and token agreement against BF16 on coding/tool calibration;
- exact tool name and JSON parse rate;
- argument correctness for nested objects;
- single-turn pass@1;
- repository episode success;
- regression and recovery rates;
- turns, tokens and time per successful task;
- looping or repeated-call rate;
- context-band degradation; and
- throughput, latency and peak VRAM in the target runtime.

Promote a quant only when the end-to-end success delta is within the threshold
defined in [Evaluation](evaluation.md). File size alone is not a gate.

## Recommended decision order

1. Merged BF16 reference.
2. Published/custom NVFP4 on the Blackwell server.
3. Dynamic 4-bit GGUF.
4. Dynamic 3-bit GGUF.
5. Dynamic 2-bit only if 3-bit passes.
6. One-bit only in an isolated research branch with its own llama.cpp work,
   calibration study and stop criteria.

The likely production choice is NVFP4/INT4 for Blackwell-native serving or
Dynamic 3/4-bit for portable llama.cpp deployment. The smallest artifact is
not automatically the best use of a 96 GB GPU; retained agent reliability and
KV-cache capacity are more valuable.

## Release manifest

Every quantised artifact must include:

- parent BF16 hash and adapter lineage;
- converter and runtime commits;
- quant type and effective bits per weight where known;
- calibration dataset revision and token count;
- tokenizer/chat-template hash;
- evaluation report;
- supported runtime and launch settings; and
- known limitations, especially reasoning and context mode.
