# Minimum path to Experiment 1

## Outcome

The first useful result is a small, reproducible baseline—not a complete
training platform. Experiment 1 answers four questions:

1. Does the upstream BF16 model use the chosen native tool schema reliably?
2. Does the adopted task/sandbox stack work from a Colab G4 notebook?
3. What are the time, token and Colab-compute costs of one repository episode?
4. Which failure classes are common enough to justify SFT data collection?

Until those answers exist, do not build the four-dataset pipeline, online RL,
custom sandbox service, multi-language suite or full quantisation matrix.

## Provisional decisions

These defaults unblock the experiment. Change them explicitly rather than
allowing them to remain open-ended.

| Decision | Experiment 1 default |
| --- | --- |
| Hardware | Google Colab G4, RTX PRO 6000 Blackwell Server Edition, 96 GB |
| Language/build system | Python with pytest; one environment profile |
| Model | `unsloth/Qwen3.8-27B` BF16, pinned revision |
| Agent interface | Project-owned Qwen-native six-tool schema |
| Task/environment runner | Harbor with an available isolated backend |
| Control agent | mini-SWE-agent where useful; bash-only traces are not SFT data |
| Task count | 12 trusted tasks from at least 3 held-out repository families |
| Attempts | Three fixed seeds per task because generation is sampled |
| Episode budget | 10 tool calls, 8-minute timeout, fixed token ceiling |
| Reasoning | `medium` for the first comparison; one small effort-mode probe |
| Durable storage | Hugging Face Hub or Drive after every irreplaceable output |
| Reasoning retention | Private experiment traces only; keep operational events separable and do not publish raw reasoning by default |

The primary Blackwell deployment candidate is NVFP4. Dynamic GGUF remains the
portable low-bit research branch. Neither choice blocks the first BF16
baseline.

## Adopt versus build

Adopt existing infrastructure where it does not define model behaviour:

- [Harbor](https://github.com/harbor-framework/harbor) for task definitions,
  environment backends, verification and trial/job execution;
- the [SWE-bench harness](https://github.com/SWE-bench/SWE-bench) when an
  official SWE-bench score is needed; and
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) only when multi-turn RL
  environment integration becomes the next proven bottleneck.

Build only the layer that is part of the intended model distribution:

- Qwen chat-template and model-server adapter;
- the six native tool schemas and response parser;
- conversion from the adopted runner's events to the project trajectory JSON;
- budget/termination accounting; and
- a compact result summariser.

[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) is a valuable
control because its loop is simple and mature, but its default interface is
bash-only and intentionally does not use model-native tool calling. Its traces
therefore cannot be used as native tool-call demonstrations without generating
new actions and observations through the target adapter.

This choice avoids writing a sandbox platform, benchmark registry and hidden
test runner while preserving control over the prompts and tools that affect
the trained policy.

## Colab boundary

The Colab notebook owns model loading, generation and result collection. Do not
treat the hosted notebook process as a security boundary for arbitrary public
repository code.

For Experiment 1:

- prefer a Harbor remote or otherwise isolated environment backend;
- if that is not yet available, run only reviewed, trusted repositories in
  disposable directories and label the mode `trusted-dev`;
- keep Hugging Face tokens out of the task environment;
- persist results before the runtime can be reclaimed; and
- print GPU, driver, package and model revisions at notebook start.

Secure execution of untrusted code is required before scaling public trace
generation or RL. It is not required to measure the first trusted-task
baseline.

## Minimum task contract

Each of the 12 tasks needs only:

- immutable repository revision;
- issue/request text;
- setup command or prebuilt environment;
- baseline-health command;
- verifier command whose implementation is not model-visible;
- time, token, tool-call and patch-size budgets; and
- train/evaluation split metadata fixed before any generation.

Use small tasks with setup and tests that normally finish quickly. Do not begin
with full SWE-bench or Terminal-Bench runs.

## Minimum implementation checklist

### Notebook preflight

- Detect the G4 GPU and total VRAM, accounting for decimal GB versus binary
  GiB reporting.
- Install/pin only the dependencies required for inference and the adapter.
- Authenticate durable storage without exposing the token to task sandboxes.
- Load the BF16 model and record peak VRAM.
- Validate one native tool call through the actual tokenizer template.

### Thin agent adapter

- Implement `list_files`, `read_file`, `search`, `apply_patch`, `run_tests`
  and restricted `shell` schemas.
- Append structured tool results to a linear message history.
- Enforce per-episode budgets and typed termination reasons.
- Write one JSONL trajectory and one compact result row per attempt.

### Adopted task runner

- Import or author 12 Harbor-format tasks.
- Run the task setup/verifier outside the model-visible workspace.
- Map environment failures to `infrastructure_failure`, never model failure.
- Destroy or abandon the environment after each trial.

### Baseline run

- Run three BF16 attempts per task with fixed seeds at `medium` reasoning
  effort. Demo mode may use one seed for a plumbing-only smoke test.
- Run protocol fixtures separately from repository tasks.
- Inspect all 12 traces manually.
- Measure mean, p50 and p95 episode duration, token use and tool calls.
- Calculate the projected cost of the candidate and release gates.

The published Dynamic 4-bit model is an optional follow-up, not a requirement
for completing Experiment 1. Adding a second runtime before the BF16 path works
would blur infrastructure and model failures.

## Data decision after Experiment 1

Assume **zero percent** of third-party agent traces are directly compatible
with the target tool schema until an audit proves otherwise.

Public agent datasets may contribute in three ways:

1. task/repository seeds for generating a new native-schema trajectory;
2. verified patches or outcomes used to check a regenerated trajectory; or
3. non-agentic code/reasoning examples that do not supervise tool calls.

Do not rename third-party tools or fabricate target-tool observations. The
agentic portion of SFT must be generated or replayed through the target adapter.
For the 4K smoke SFT, collect 100–300 successful native trajectories through a
teacher, human-guided run or verified model attempts. If that corpus cannot be
produced, reduce the smoke run rather than filling the quota with translated
traces.

The 50% agentic token mixture is a later target, not a prerequisite or a
promise that half of public traces will survive conversion.

## Evaluation budget produced by Experiment 1

Use the observed episode time to fill in:

```text
expected GPU hours = tasks * attempts * mean_episode_seconds / 3600
timeout upper bound = tasks * attempts * episode_timeout_seconds / 3600
expected Colab units = expected GPU hours * observed_units_per_hour
```

No candidate gate is approved until its expected and timeout-upper-bound costs
fit the available Colab budget. See [Evaluation](evaluation.md) for the funnel
that prevents every quant or checkpoint receiving the most expensive gate.

## Definition of done

Experiment 1 is complete when:

- the 12-task manifest and split are frozen;
- the Qwen-native adapter version is recorded;
- all protocol fixtures pass or have classified failures;
- all repository attempts have traces and verifier outcomes;
- infrastructure failures are separated from model failures;
- the time/token/compute cost report exists; and
- the next work item is selected from observed failures.

## Explicitly deferred

- Production sandbox service.
- Multiple languages and build systems.
- Automated replay of entire public trace datasets.
- DPO, GRPO/GSPO and NeMo Gym integration.
- Dataset publication pipeline.
- Full benchmark suites and confidence-interval claims.
- Adapter merge and quantisation ladder.
- Refactoring every reference notebook into package code.

Only pull a deferred item forward when Experiment 1 shows that it blocks the
next experiment.
