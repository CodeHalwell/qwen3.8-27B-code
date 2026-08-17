# Training plan

## Strategy

Use staged capability development. Each stage starts from the best checkpoint
that passed the previous evaluation gate, and each has a frozen control run.

```text
baseline -> SFT smoke -> main SFT -> preference optimisation -> agentic RL
```

Do not begin online RL until rewards are unit-tested and SFT already produces
valid tool calls. RL is a poor way to repair a broken template or malformed
dataset.

Before implementing this full sequence, complete the deliberately reduced
[Minimum path to Experiment 1](minimum-path.md).

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

## Stage 0: upstream baseline

Run the stock trainable checkpoint and at least one published GGUF in the same
harness. Capture:

- single-turn code benchmarks;
- held-out repository completion;
- valid tool-call and argument rates;
- recovery after injected tool failures;
- tests passed and regressed;
- tokens, turns and wall time; and
- behaviour at `low`, `medium` and `xhigh` reasoning effort.

Freeze this result and the exact harness version. It is the comparison point
for every later claim.

## Stage 1: pipeline smoke SFT

Use a small, audited dataset to prove the complete path:

1. Load the pinned BF16 Unsloth checkpoint.
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
| Precision | BF16 LoRA | Fall back to 4-bit QLoRA after measured OOM only |
| LoRA rank / alpha | 16 / 32 | Compare rank 8 or 32 only after baseline run |
| LoRA dropout | 0 | Unsloth-optimised default |
| Target | Language all-linear after module discovery | Explicitly exclude vision |
| Sequence length | 8,192 | 4,096 smoke; profile 16,384 separately |
| Device batch | 1 | Single GPU |
| Gradient accumulation | Tune to token budget | Report tokens/update, not only examples/update |
| Checkpointing | Unsloth gradient checkpointing | Record peak VRAM |
| Optimizer | 8-bit AdamW initially | Verify support with pinned stack |
| Learning rate | Begin near `2e-5` for the main run | `2e-4` is only a short-LoRA experiment candidate |
| Training length | Token-budgeted, at most roughly one pass initially | Stop on held-out regression |
| Loss | Assistant tokens only | Includes assistant tool calls |
| Tracking | Trackio plus machine-readable run manifest | Required for comparable experiments |

The exact loader (`FastLanguageModel` versus the current multimodal loader) and
target module names are preflight results, not constants to copy from an older
notebook.

Evaluate frequently enough to catch protocol and coding regression, but do not
run the full repository suite every few steps. Use a small sentinel set during
training and the complete gate at candidate checkpoints.

## Stage 3: preference optimisation

Build high-confidence chosen/rejected pairs from the same task state. DPO is
the default first experiment because it is operationally simpler than online
multi-turn RL.

Single-GPU constraints make reference handling important. Test, in order:

1. shared-base/reference adapter techniques supported by the pinned TRL/PEFT
   versions;
2. reference-free or precomputed-reference-log-probability modes when
   theoretically appropriate; and
3. QLoRA for the preference stage if BF16 memory is insufficient.

Begin at 4K sequences with a small beta sweep rather than assuming an optimal
KL strength. The winning model must improve execution outcomes without
collapsing exploration, reasoning depth or general coding ability.

## Stage 4: agentic GRPO/GSPO

Use online RL only for tasks with executable rewards. Begin with 2–4 samples
per prompt, short output budgets and 2–10-tool-call episodes.

The reward is a named vector before it is a scalar:

| Component | Signal | Direction |
| --- | --- | ---: |
| Hidden correctness | Hidden tests passed | Strong positive |
| Visible correctness | Required visible tests passed | Positive |
| Build validity | Patch applies, imports/compiles | Positive |
| Regression | Previously passing tests fail | Strong negative |
| Tool protocol | Valid name and JSON schema | Small positive / invalid negative |
| Scope | Unrelated files or excessive churn | Negative |
| Efficiency | Success with fewer redundant calls/tokens | Small positive |
| Safety | Sandbox escape or prohibited action | Terminal negative |

Correctness must dominate efficiency. Otherwise the policy may learn to stop
early, avoid tests or make tiny but incomplete patches.

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
- peak allocated/reserved VRAM;
- validation tool-call parse rate;
- sentinel repository success;
- mean reward by component;
- reward variance and fraction of zero-standard-deviation groups;
- completion length, tool calls and timeout rate; and
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
- the model learns repeated calls, test suppression or another reward exploit;
- training becomes numerically unstable; or
- dataset/reward contamination is discovered.

The best checkpoint may precede the final training step.
