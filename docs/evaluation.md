# Evaluation and stage gates

## Evaluation philosophy

Evaluate the capability the model will actually deliver: completing repository
tasks through tools without regressions. Static benchmarks are useful guardrails
but are not the primary optimisation target.

Every comparison fixes:

- task and repository revisions;
- harness and tool-schema versions;
- model, adapter and quantisation revisions;
- reasoning mode and generation parameters;
- context and tool budgets;
- sandbox resources; and
- seeds or repeated-trial count.

Report confidence intervals or per-task outcomes, not only a single aggregate.

## Evaluation layers

### 1. Protocol tests

Run quickly on every candidate:

- chat-template round trip;
- developer-role retention;
- single and multiple tool calls;
- nested JSON arguments;
- invalid-call recovery;
- assistant-only label masking;
- thinking preservation on/off; and
- no partial tool call after truncation.

Target: effectively 100% on deterministic protocol fixtures.

### 2. Static coding tests

Use contamination-aware slices of:

- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench);
- [BigCodeBench](https://github.com/bigcode-project/bigcodebench); and
- internal language/build-system suites.

These detect loss of code generation, library use and instruction following.
They do not prove agentic performance.

### 3. Repository tasks

Use held-out repositories and immutable commits. Cover:

- bug fixes;
- feature additions;
- refactors with invariant tests;
- test authoring;
- dependency/configuration repair; and
- multi-file changes.

Where licences and infrastructure permit, add SWE-bench-family and
Terminal-Bench-style tasks. Maintain a private evaluation set because public
benchmarks are increasingly present in training corpora.

### 4. Robustness tasks

Inject realistic faults:

- a test command fails for an understandable dependency reason;
- search output is truncated;
- the first hypothesis is contradicted by source code;
- a patch conflicts;
- one tool returns a typed error; and
- context compaction is triggered.

Measure whether the model diagnoses and recovers rather than merely avoiding
invalid JSON.

### 5. Long-horizon bands

Evaluate by observed episode length rather than calling every repository task
“long horizon”:

| Band | Tool calls | Typical purpose |
| --- | ---: | --- |
| Short | 1–5 | Protocol and simple edit |
| Medium | 6–15 | Debug, edit and verify |
| Long | 16–30 | Multi-file/replanning |
| Extended | 31+ | Context management and sustained agency |

Also report prompt-plus-history token bands. A 30-call episode with tiny tool
outputs differs materially from one carrying large compiler logs.

## Core scorecard

| Metric | Definition | Direction |
| --- | --- | ---: |
| Episode success | All required hidden criteria pass | Higher |
| Visible test fraction | Required visible tests passed | Higher |
| Regression rate | Previously passing tests now fail | Lower |
| Valid tool-call rate | Calls parsed and matched schema | Higher |
| Tool argument success | Valid calls that execute as intended | Higher |
| Recovery rate | Recoverable injected failures eventually resolved | Higher |
| Unsupported success claim | Model claims completion without required verification | Lower |
| Patch precision | Task-relevant changed lines / total changed lines | Higher, interpreted carefully |
| Calls per success | Tool calls divided by successful episodes | Lower after correctness |
| Tokens per success | Model tokens divided by successful episodes | Lower after correctness |
| Time per success | Wall time divided by successful episodes | Lower after correctness |
| Loop rate | Repeated equivalent actions without progress | Lower |

Do not optimise patch precision or efficiency ahead of correctness. Some tasks
genuinely require broad changes.

## Compute budget and evaluation funnel

The full gate is not run on every checkpoint or quant. First measure episode
cost in Experiment 1, then freeze a budget alongside each gate.

```text
expected GPU hours = tasks * attempts * mean_episode_seconds / 3600
timeout upper bound = tasks * attempts * timeout_seconds / 3600
expected Colab units = expected GPU hours * observed_units_per_hour
```

Colab availability and compute-unit rates can vary, so record the observed
burn rate from the actual G4 session instead of hard-coding a monetary price.

Initial budget ceilings are:

| Tier | Work | Maximum scheduled GPU time per candidate |
| --- | --- | ---: |
| Protocol | Deterministic template/tool fixtures | 0.25 hours |
| Sentinel | 12 repository tasks, one deterministic attempt, 8-minute timeout | 1.6 hours |
| Candidate | 24 tasks, three sampled seeds, 12-minute timeout | 14.4 hours |
| Release | 40 tasks, three attempts, 15-minute timeout | 30 hours |

These are timeout ceilings, not expected runtimes. Replace them after the pilot
with measured mean and p95 projections. If a gate does not fit the available
budget, reduce attempts or report paired task outcomes; do not imply precision
the experiment cannot afford.

Use this quantisation funnel:

1. Protocol and cheap static checks on every loadable artifact.
2. Sentinel gate on BF16, Q4/NVFP4, Q3 and any experimental Q2.
3. Candidate gate on BF16 and at most two compact finalists.
4. Release gate on the chosen compact model and its BF16 parent only.
5. Q1 receives no expensive gate until it passes protocol and sentinel checks.

During training, run protocol checks plus a smaller fixed sentinel subset.
Reserve candidate and release gates for checkpoints that have already passed
the cheaper tiers.

## Stage gates

Thresholds below are starting policy and should be frozen before viewing a
candidate's results.

### SFT gate

- Tool protocol fixtures: no regression.
- Held-out repository success: better than upstream or statistically tied with
  a meaningful efficiency/robustness improvement.
- Static coding aggregate: no more than 2% relative regression.
- Regression and unsupported-success-claim rates: no worse than upstream.
- At least one improvement appears in both medium and long episode bands.

### Preference/RL gate

- Repository success exceeds the accepted SFT checkpoint by a predeclared
  margin, initially 3 percentage points absolute on the internal suite.
- Improvement is not explained only by longer outputs or more tool calls.
- Reward rises with hidden-test success.
- No meaningful increase in safety violations, loops or malformed calls.
- At least two independent evaluation runs reproduce the direction.

### Quantisation gate

Relative to merged BF16:

- 4-bit/NVFP4: target no more than 1 percentage point absolute episode-success
  loss and no protocol regression.
- 3-bit: target no more than 2 points absolute loss.
- 2-bit: research acceptance up to 5 points only if the memory/throughput gain
  is operationally valuable.
- 1-bit: no production gate until a coherent, loadable 27B artifact exists;
  it must still beat the upstream non-specialised baseline on the target suite.

Use enough tasks and repeated trials to make these deltas interpretable. For a
small suite, publish the task-level paired outcomes instead of claiming precise
percentage-point significance.

## Quantisation diagnostics

Compare BF16 and each quant on identical contexts. Categorise divergent first
actions and failures:

- different plan but both succeed;
- invalid/mis-selected tool;
- corrupted JSON arguments;
- incorrect code token;
- lost fact from earlier observation;
- repeated action or loop;
- premature completion; and
- reasoning-mode/template incompatibility.

This diagnosis determines whether better calibration, a higher-bit tensor, QAT
or simply a less aggressive target is appropriate.

## Reward validation

Before RL, create unit fixtures for:

- full success;
- partial tests;
- visible success but hidden failure;
- regression of a baseline test;
- invalid tool call;
- timeout after progress;
- unrelated broad edit;
- infrastructure failure; and
- attempted reward hacking such as deleting tests.

Infrastructure errors are excluded from policy updates. A model must not earn
correctness reward by changing tests, suppressing failures or modifying the
verifier.

## Result report

Each candidate report should contain:

1. Reproducibility manifest.
2. Aggregate scorecard with uncertainty.
3. Results by task type, language, horizon and context band.
4. Comparison to upstream and previous accepted checkpoint.
5. Resource use and cost per successful task.
6. Ten representative wins and ten failures with trace links.
7. Contamination/licence statement.
8. Decision: accept, reject or investigate.

Store raw machine-readable results alongside the Markdown summary.
