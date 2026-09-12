# Training plan

## Strategy

Use staged capability development. Each stage starts from the best checkpoint
that passed the previous evaluation gate, and each has a frozen control run.

```text
baseline -> SFT smoke -> main SFT -> preference optimisation -> agentic RL
```

Every stage optimises a single objective: agentic coding through the native
six-tool schema. The corpus carries no general-capability replay slice (see
the [specialisation policy](data-strategy.md#specialisation-policy)); drift on
non-coding chat is an accepted trade, while coding, tool-protocol and
harness-safety regressions remain hard stop conditions. Concentrated coding
gradients are the mechanism for that trade — do not add unlearning objectives
to force it further, because deleting general knowledge does not free capacity
and measurably damages the coding behaviour these stages are gated on.

Do not begin online RL until rewards are unit-tested and SFT already produces
valid tool calls. RL is a poor way to repair a broken template or malformed
dataset.

Before implementing this full sequence, complete the deliberately reduced
[Minimum path to Experiment 1](minimum-path.md).

## What decides the outcome

The stock model already posts strong agentic-coding scores (see the
[published baseline](model-and-hardware.md#published-coding-baseline)), so the
question is not whether this pipeline can train, it is whether it can move a
model that good without breaking it. Three things decide that, in this order,
and none of them is a hyperparameter:

1. **Task supply at real-repository scale.** The bootstrap corpus is 204
   scripted trajectories over twelve toy fixture families, 233K tokens in
   total. It proves plumbing. A main SFT that changes the behaviour of a 27B
   model needs verified trajectories from real repositories in the thousands,
   and tens of millions of assistant tokens, before the rank or the learning
   rate matter. The teacher route (notebook 08) and the collector are the
   machinery; resolvable real-repository tasks with executable tests are the
   missing input, and they are the next item of work.
2. **Sequence length policy.** Real trajectories run to 8K-32K tokens once
   tool output is in the transcript, and right truncation at 4K or 8K cuts
   the patch and the verification off exactly the rows that carry the most
   signal. Raise the window to 16K first, which the
   [logit memory](model-and-hardware.md#logit-memory) arithmetic makes
   conditional on the chunked loss. For trajectories that still do not fit,
   slice at assistant-turn boundaries into examples whose context is the
   compacted preceding history, with loss on that one turn only, rather than
   truncating from the right; accept the multiplied prefix cost for those
   rows alone. The long-horizon band then stops being excluded from training
   by construction.
3. **An external gate.** The held-out suite is eight families. It detects
   regression and protocol damage; it cannot support a claim about coding
   ability in general. Before any checkpoint is called an improvement, score
   a contamination-aware external slice (SWE-bench Verified through the
   official harness, or Terminal-Bench) against a served checkpoint, as
   [Evaluation](evaluation.md#3-repository-tasks) describes.

Everything else in this plan is in service of those three.

## Colab execution contract

The notebook runs on an ephemeral Google Colab G4 session. Every training stage
must therefore:

- print and persist the GPU, runtime, package, model and dataset revisions;
- push checkpoints to Hugging Face Hub or copy them to Drive at each durable
  save point;
- verify the remote artifact before considering a local checkpoint expendable;
- accept `resume_from_checkpoint` after a clean runtime restart;
- keep Trackio logs and a local machine-readable manifest in durable storage;
  and
- avoid running untrusted task code in the same process that holds Hub tokens.

Notebook cells should be idempotent: rerunning setup, data validation or resume
cells must not silently create a different experiment.

## Hardware lanes

| Lane | Hardware | Runs | Does not run |
| --- | --- | --- | --- |
| Capability | Colab G4, RTX PRO 6000 Blackwell, 96 GB, BF16 LoRA | Every stage below, every gate, every number that is compared | Nothing is excluded |
| Plumbing | Kaggle T4 x2, 2 x 16 GB, 4-bit QLoRA, 1,024-2,048 tokens | Notebook 00's template and masking checks, adapter save and reload, tool-parser and harness fixtures, `DEMO_MODE` smoke steps | Any run whose loss, memory, speed or held-out result will be quoted |

The plumbing lane follows Unsloth's own Qwen3.8-27B notebook
(`references/qwen3_8_27b_kaggle_t4x2.py`), and its rules are not negotiable
there: 4-bit load, no `dtype` or `fp16` flags (the DeltaNet path produces
NaN gradients in pure float16 and a T4 has no bfloat16), `device_map` left at
Unsloth's sequential default, device batch 1 with gradient accumulation,
rank 8, and no merge on the kernel. The measured budget and the reasons are
in [Model and hardware](model-and-hardware.md#second-lane-kaggle-t4-x2). A
row that renders, masks and trains for two steps there has proved the data
path; it has proved nothing about the model, and the gates stay on the G4.

## Stage 0: upstream baseline

Run the stock trainable checkpoint and at least one published GGUF in the same
harness. Capture:

- single-turn code benchmarks;
- held-out repository completion;
- valid tool-call and argument rates;
- recovery after injected tool failures;
- tests passed and regressed;
- tokens, turns and wall time; and
- success and reasoning tokens per turn at `low`, `medium` and `xhigh`
  reasoning effort, the effort ladder of [Thinking budget](thinking-budget.md).

Give each rung of the effort ladder its own per-turn generation cap.
Reasoning tokens count against `max_new_tokens`, and a turn cut off inside
its think block returns no action, so one cap sized for `medium` turns the
`xhigh` rung into a measurement of truncation rather than of effort. Measure
the p95 reasoning length per effort on a few tasks first, set each cap above
it, and size the episode's context budget so that ten such turns fit.
Notebook 07's fixed 2,048-token cap and 16,384-token context are `medium`
settings only. Note also that `medium` renders no instruction at all, so
that rung measures the model's uninstructed behaviour.

Freeze this result and the exact harness version. It is the comparison point
for every later claim.

## Stage 1: pipeline smoke SFT

Use a small, audited dataset to prove the complete path:

1. Load the pinned BF16 Unsloth checkpoint, recording the loader class and
   whether it returned a processor (preflight item 3 in
   [Model and hardware](model-and-hardware.md#preflight-checks)).
2. Freeze vision parameters.
3. Discover and attach LoRA to language linear modules.
4. Render native multi-turn tool conversations.
5. Verify assistant-only masking.
6. Train a short run at 4,096 tokens.
7. Save and reload the adapter.
8. Merge into BF16.
9. Run tool protocol and held-out smoke evaluations.

This stage is allowed to overfit a tiny sample for plumbing validation, but its
checkpoint is not a model candidate.

## Stage 2: main SFT

Starting configuration:

| Setting | Initial value | Notes |
| --- | --- | --- |
| Base model | `unsloth/Qwen3.8-27B` | Pin Hub revision |
| Loader | Preflight result; `FastModel` in Unsloth's own Qwen3.8 notebook | Returns a processor, so tokenise text by keyword and reach the inner tokenizer for bare strings |
| Precision | BF16 LoRA | The DeltaNet recurrent state is float32 by configuration; never force float16 anywhere in the stack (NaN gradients). Fall back to 4-bit QLoRA after measured OOM only, noting that the 4-bit build keeps `lm_head` and the DeltaNet `in_proj_qkv/a/b` in 16-bit |
| LoRA rank / alpha | 16 / 32 | Escalate to 32 / 64 as the specialisation lever once the 16 / 32 baseline passes its gate; change one axis per run |
| LoRA dropout | 0 | Unsloth-optimised default |
| Target | Language all-linear after module discovery | Includes the Gated DeltaNet `in_proj_qkv/z/a/b` and `out_proj`; exclude vision, MTP and `lm_head` |
| Sequence length | 8,192 | 4,096 smoke; 16,384 only once the chunked loss is confirmed active, because full logits at 16K are about 38 GiB on their own ([logit memory](model-and-hardware.md#logit-memory)) |
| Device batch | 1 | Single GPU |
| Gradient accumulation | Tune to token budget | Report tokens/update, not only examples/update |
| Checkpointing | Unsloth gradient checkpointing | Record peak VRAM |
| Kernels | Fused linear-attention kernels where the pinned stack supports them | The reviewed install omits `flash-linear-attention` and `causal_conv1d`, which the adjacent Qwen3.5 example installs; record the active DeltaNet path and tokens/s either way |
| Optimizer | 8-bit AdamW initially | Verify support with pinned stack |
| Learning rate | Begin near `2e-5` for the main run | Unsloth's own notebook uses `2e-4` for a 30-step demo and says to drop to `2e-5` for long runs; sweep `2e-5`, `5e-5` and `1e-4` at smoke scale, gated on held-out success, before committing the main run |
| Training length | Token-budgeted, at most roughly one pass initially | Stop on held-out regression |
| Loss | Assistant tokens only | Includes assistant tool calls |
| Tracking | Trackio plus machine-readable run manifest | Required for comparable experiments |

The exact loader (`FastLanguageModel` versus the current multimodal loader) and
target module names are preflight results, not constants to copy from an older
notebook. Unsloth's Qwen3.8 notebook uses `FastModel`; if `FastLanguageModel`
refuses the checkpoint in preflight, that is the one-line switch to make in
the generator, and the contract test that counts load cells changes with it.

Evaluate frequently enough to catch protocol and coding regression, but do not
run the full repository suite every few steps. Use a small sentinel set during
training and the complete gate at candidate checkpoints.

## Stage 3: preference optimisation

Build high-confidence chosen/rejected pairs from the same task state. DPO is
the default first experiment because it is operationally simpler than online
multi-turn RL.

Reference handling decides what the stage optimises, before it is a memory
question. TRL's `ref_model=None` does not mean "no reference": it evaluates the
policy with its adapters disabled. Loading the accepted SFT *adapter* and
relying on that default therefore makes the base model the reference, so the KL
term pulls back toward pre-SFT behaviour. Start preference training from the
merged SFT checkpoint with a fresh adapter, so disabling it yields the accepted
SFT policy. Note also that `5e-7` is a full-fine-tuning rate; a rank-16 adapter
needs roughly 10-100x that, swept alongside beta.

Single-GPU constraints then make reference memory important. Test, in order:

1. shared-base/reference adapter techniques supported by the pinned TRL/PEFT
   versions;
2. reference-free or precomputed-reference-log-probability modes when
   theoretically appropriate; and
3. QLoRA for the preference stage if BF16 memory is insufficient.

Begin at 4K sequences with a small beta sweep rather than assuming an optimal
KL strength. The winning model must improve execution outcomes without
collapsing exploration, reasoning depth or general coding ability.

The mixture may include the reasoning-length pairs the collector writes
(same task, same action, both verified, the shorter think block preferred)
as a minority next to the execution-derived pairs. The gate then reads the
thinking check and the medium horizon band together: shorter thinking that
costs the multi-file tasks is a regression, not a win.

Render every pair at the effort its continuations were generated at. The
template injects an instruction for `low` and `xhigh` and nothing for
`medium`, so a `low` pair rendered at `medium` has lost the instruction its
reasoning was written under. Notebook 04 renders all pairs at `medium`
today, which is consistent only while collection runs at a single effort;
the pair rows need to carry the effort label before the mixture spans the
ladder.

## Stage 4: agentic GRPO/GSPO

Use online RL only for tasks with executable rewards. Begin with 2–4 samples
per prompt, short output budgets and 2–10-tool-call episodes.

The current core Colab environment intentionally pins TRL 0.22.2 because the
pinned Unsloth Zoo declares `trl<=0.24.0` and `datasets<4.4.0`. TRL's stateful
`environment_factory` arrived in 0.29.0 and therefore cannot be installed in
the same reviewed environment today. Notebook 05 validates environment and
reward contracts but must not run policy updates until one of these paths is
proven separately:

1. an Unsloth release with compatible TRL environment support;
2. a separate rollout process using Harbor, NeMo Gym or TRL/OpenEnv while the
   compatible Unsloth environment performs updates; or
3. a custom, unit-tested rollout adapter whose policy/version skew is recorded.

Upstream Unsloth PR #8810 raises the TRL cap to 1.10.0 and was still open on
2026-09-12. Waiting on it is not the only route. Unsloth's own notebooks pin
TRL past the declared cap with `--no-deps` as a matter of course: the Kaggle
Qwen3.8 notebook does it for 0.22.2, and the GRPO reference in
`references/notebook24f5f9a990.ipynb` runs TRL 1.9.2 against Unsloth's git
head. That is acceptable here on one condition: it happens in a separate,
frozen environment for notebook 05 whose full `pip freeze`, git revisions and
compatibility-probe output are committed next to the run manifest, so the
environment is reproducible by construction rather than by the resolver's
blessing. A bare `--no-deps` in the reviewed core matrix, with nothing
recorded, is still not a reproducible run. Do not enable training until the
probe passes in that environment.

Two memory facts from the same references shape the first GRPO run. GRPO's
chunked log-softmax materialises `rows x 248,320` logits plus a float32 copy
on whichever card holds `lm_head`, so `num_generations` and
`max_completion_length` are the two knobs that decide whether a step fits,
and a multi-GPU rollout needs the hidden states co-located with the head
before the matmul. On one 96 GB card neither blocks the group sizes below,
but record peak reserved memory per step from the first run.

The reward is a named vector before it is a scalar:

| Component | Signal | Direction |
| --- | --- | ---: |
| Hidden correctness | Hidden tests passed | Strong positive |
| Visible correctness | Required visible tests passed | Positive |
| Build validity | Patch applies, imports/compiles | Positive |
| Regression | Previously passing tests fail | Strong negative |
| Tool protocol | Valid name and JSON schema | Small positive / invalid negative |
| Scope | Unrelated files or excessive churn | Negative |
| Efficiency | Fewer reasoning tokens and redundant calls, paid only to correct samples (`thinking.length_rewards`) | Small positive |
| Safety | Sandbox escape or prohibited action | Terminal negative |

Correctness must dominate efficiency. Otherwise the policy may learn to stop
early, avoid tests or make tiny but incomplete patches. The brevity term is
group-relative and clipped to zero for incorrect samples, and its weight
(0.1) sits well below the 0.6 gap between a hidden pass and a hidden fail,
so it reorders correct samples among themselves and nothing else; notebook
05 pins those properties with fixtures.

Example scalarisation for early experiments:

```text
reward =
    0.55 * hidden_test_fraction
  + 0.25 * visible_test_fraction
  + 0.10 * build_validity
  + 0.05 * valid_tool_protocol
  + 0.05 * bounded_efficiency
  - regression_penalty
  - scope_penalty
  - safety_penalty
```

This is a hypothesis, not a permanent formula. Unit-test each component with
known good, partial, adversarial and infrastructure-failure trajectories.

## One-GPU rollout schedule

Long multi-turn rollouts and BF16 adapter updates compete for the same VRAM.
Use an alternating schedule:

1. Load the current policy in inference mode.
2. Generate a bounded batch of trajectories into CPU/disk storage.
3. Finalise rewards and discard infrastructure failures.
4. Release or reconfigure inference allocations.
5. Run LoRA policy updates.
6. Evaluate and checkpoint.
7. Refresh the rollout policy and repeat.

Unsloth's memory-efficient RL/standby facilities may reduce reloading cost, but
must pass a Qwen3.8 compatibility smoke test. A separately quantised rollout
model changes the behaviour policy and introduces off-policy mismatch; record
that explicitly rather than treating it as equivalent to the BF16 policy.

If throughput becomes the limiting factor, the highest-value hardware addition
is a separate rollout GPU or temporary inference service. It is not required
for the first experiments.

## Curriculum

Advance in this order:

1. One correct tool call.
2. Inspect then answer without editing.
3. Inspect, make one edit and run one test.
4. Recover from a failed test or malformed assumption.
5. Multi-file implementation with regression tests.
6. Longer debugging involving repeated observation and replanning.
7. Long-context repository work and context compaction.

Increase one axis at a time: task difficulty, tool-call budget, output length or
context length. Changing all four makes regressions difficult to diagnose.

## Monitoring

Trackio runs should include:

- dataset, model and code revisions;
- LoRA configuration and trainable parameter count;
- tokens per update and length-bucket distribution;
- loss, learning rate, gradient norm and throughput;
- a NaN/inf check on loss and gradient norm over the first steps, which is
  where a float16 path or a broken DeltaNet kernel shows;
- the active DeltaNet kernel path and loss implementation;
- peak allocated/reserved VRAM;
- validation tool-call parse rate;
- sentinel repository success;
- mean reward by component;
- reward variance and fraction of zero-standard-deviation groups;
- completion length, reasoning tokens per turn, thinking-overrun rate, tool
  calls and timeout rate; and
- checkpoint artifact hashes.

Checkpoint locally and to durable storage. Verify a checkpoint can be loaded
before deleting any previous copy.

## Stop conditions

Stop or roll back when any of the following persists across evaluation noise:

- held-out repository success falls below the previous accepted checkpoint;
- malformed tool calls rise materially;
- general code benchmark performance regresses beyond the allowed delta;
- reward rises while hidden-test success does not;
- output length or patch size grows without more successful tasks;
- reasoning tokens per turn rise past the thinking-budget ceiling, or the
  medium horizon band drops while the short band holds;
- the model learns repeated calls, test suppression or another reward exploit;
- training becomes numerically unstable; or
- dataset/reward contamination is discovered.

Drift on non-coding chat is not on this list by design: the specialisation
policy accepts it. Stop only on the coding, tool-protocol, safety and
stability signals above.

The best checkpoint may precede the final training step.
