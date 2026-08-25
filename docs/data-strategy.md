# Data strategy

## Principles

The model should learn the work loop of software engineering: inspect, form a
hypothesis, edit narrowly, run verification, interpret failures and recover.
Static instruction/answer pairs alone do not teach that loop.

All capability data should be execution-verified where practical. Every
training source needs a licence record, provenance, transformation version and
deduplication fingerprint.

## Four separate datasets

Do not reuse one loosely typed table for every training method.

| Dataset | Required content | Consumer |
| --- | --- | --- |
| Demonstrations | Complete messages and successful tool trajectory | SFT |
| Preferences | Shared prompt plus chosen and rejected continuation/trajectory | DPO or related preference optimisation |
| RL tasks | Initial prompt, task manifest and executable environment | GRPO/GSPO |
| Evaluation | Held-out repositories, manifests and hidden verification | Stage gates only |

Repository-level split membership is fixed before generating attempts. No
commit, fork, near-duplicate patch or issue from an evaluation repository may
enter training.

## SFT examples

Use conversational messages rather than a pre-rendered text field as the
canonical form. Render with the pinned tokenizer during preprocessing.

```json
{
  "id": "sft/example-0001",
  "source": "internal-reviewed",
  "repo_family": "project-a",
  "messages": [
    {"role": "developer", "content": "Work in the repository and verify changes."},
    {"role": "user", "content": "Add validation for empty identifiers."},
    {"role": "assistant", "tool_calls": [{"name": "search", "arguments": {"query": "identifier"}}]},
    {"role": "tool", "name": "search", "content": "..."},
    {"role": "assistant", "tool_calls": [{"name": "apply_patch", "arguments": {"patch": "..."}}]},
    {"role": "tool", "name": "apply_patch", "content": "Done"},
    {"role": "assistant", "tool_calls": [{"name": "run_tests", "arguments": {"profile": "unit"}}]},
    {"role": "tool", "name": "run_tests", "content": "42 passed"},
    {"role": "assistant", "content": "Implemented validation and added regression coverage."}
  ],
  "verification": {"all_required_tests_pass": true},
  "reasoning_effort": "medium"
}
```

Train only on assistant tokens. Assistant tool calls are targets; tool outputs,
user prompts and developer instructions are context, not targets.

## Initial SFT mixture

Use token proportions, not example counts, because long trajectories otherwise
dominate unexpectedly.

| Category | Initial token share | Purpose |
| --- | ---: | --- |
| Successful repository-agent trajectories | 50% | Core inspect/edit/test loop |
| Verified code generation and reasoning | 20% | Algorithmic and implementation depth |
| Debugging, test repair and refactoring | 20% | Diagnosis and minimal patches |
| Tool protocol, failures and recovery | 10% | Robust tool calls and replanning |

There is deliberately no general-capability replay slice; see the
[specialisation policy](#specialisation-policy) below.

Treat this as a later starting hypothesis, not a data-acquisition quota. The
50% agentic component must contain target-schema trajectories generated or
replayed through this project's adapter. If insufficient native data exists,
run a smaller experiment or postpone main SFT rather than filling the share
with syntactically translated third-party traces. Adjust the mixture using
held-out task performance, not SFT loss alone.

Balance other important dimensions:

- languages and build systems;
- greenfield implementation versus maintenance;
- test-present versus test-authoring tasks;
- short versus long repository context;
- successful first attempts versus recovery after realistic failures; and
- `low`, `medium` and `xhigh` reasoning effort.

## Specialisation policy

The corpus is 100% software-engineering data. Every training token in every
stage — SFT demonstrations, preference pairs and RL episodes — should push the
policy toward the inspect/edit/test loop through the native six-tool schema.
There is no general-chat, world-knowledge or open-domain slice, and none should
be added as "insurance": drift on non-coding chat is an accepted cost of
specialisation, not a defect, and it is neither replayed against nor gated on.

Two boundaries keep the intent honest:

- **Specialisation is concentrated signal, not deleted knowledge.** Parameters
  are not a budget that general knowledge occupies and coding can reclaim.
  Targeted unlearning objectives — gradient ascent, corrupted labels or random
  targets on general text — damage the shared representations that coding,
  instruction following and tool use sit on, and reliably regress the coding
  gates this project exists to pass. Do not use them. The levers that convert
  capacity into coding skill are more verified coding tokens, adapter rank
  (see the training plan's escalation note), and longer training within the
  stop conditions.
- **The adapter is where the specialisation lives.** LoRA keeps the base
  weights frozen, so the coding specialisation scales with adapter capacity,
  merges into the release checkpoint, and stays reversible during
  development. The general ability that coding itself depends on — reading
  documentation, requirements and error text in natural language — survives
  through the frozen base and through coding data that exercises it. That is
  the only general knowledge this project needs to keep.

Safety behaviour inside the agent harness — refusing destructive commands,
staying in the sandbox, not fabricating verification — is not general-chat
retention and is not covered by this trade. It is trained by the tool
protocol/failure category, enforced by the environment, scored by the RL
safety penalty, and checked at every acceptance gate.

## Public seed sources

Potential starting sources include:

- [NVIDIA Nemotron-SFT-SWE-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v3)
- [NVIDIA Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)
- [NVIDIA OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)

Availability is not permission to mix and publish. For every source, record:

- dataset and revision;
- licence and downstream model restrictions;
- source-model provenance;
- benchmark overlap risk;
- whether tests were actually executed;
- tool schema and chat format; and
- transformations applied locally.

Prefer tasks whose outcomes can be reproduced. Re-execute regenerated
target-schema traces where repository revisions and dependencies are available;
do not assume the original third-party action sequence is reusable.

## Tool-schema compatibility policy

The public sources above were collected under other prompts, tools and
environments. Their agent actions are therefore **not directly trainable by
default**, even if their JSON can be parsed.

Classify each audited source row into one of four lanes:

| Lane | Treatment |
| --- | --- |
| Native-compatible | Use only after proving the tool meaning, observations and environment are equivalent and replay succeeds |
| Regenerable | Retain the task/revision/outcome, discard original actions, and generate a new verified trace with the target adapter |
| Non-agentic | Use code/reasoning content without tool-call supervision |
| Reject | Drop rows that cannot be licensed, resolved, replayed or decontaminated |

Do not rename a third-party `bash` call to `run_tests`, split a shell transcript
into invented semantic calls or fabricate observations. That produces fluent
but false supervision.

The planning assumption is **0% direct survival** until a stratified audit of
at least 100 rows reports otherwise. Record the fraction in each lane and the
replay success rate. Regardless of the audit, 100% of the main SFT mixture's
“repository-agent trajectory” category must use the target tool schema; public
data can supply tasks from which those trajectories are regenerated.

For the first 4K SFT smoke run, target 100–300 successful native trajectories.
They may be teacher-generated, human-guided, model-generated or scripted, but
they must execute in the target harness and pass verification. This is
deliberately much smaller than a production corpus.

The repository ships that smoke corpus: `scripts/generate_sft_corpus.py`
drives the real six-tool harness over pytest-verified fixture repositories
and emits `data/native_sft/trajectories.jsonl` (scripted gold trajectories —
every observation recorded from a real execution, five trajectory shapes
including failure recovery and test authoring, reasoning-effort mix per this
document). `scripts/generate_preference_pairs.py` produces the matching
execution-derived preference pairs. Both artifacts carry quality reports with
real-tokenizer length statistics and are validated by the test suite against
notebook 02's exact row validation. They satisfy the smoke gate only; the
main SFT mixture still requires regenerated trajectories from real
repositories.

## Demonstration filtering

Reject or quarantine examples when:

- the final patch does not apply to the recorded revision;
- required tests fail;
- the trajectory edits files outside task scope without justification;
- tool calls cannot round-trip through the native template;
- logs contain credentials, personal data or hidden-test contents;
- the episode succeeded through test leakage or disabled verification;
- the final result is truncated; or
- tokenisation exceeds the configured bucket without a safe compaction step.

Keep useful recovery behaviour, but do not SFT directly on unsuccessful final
outcomes. Failed attempts are candidates for preference data.

## Preference pairs

A preference record must compare continuations from the same state. Whole
trajectories with unrelated prompts are not valid pairs.

Prioritise execution-derived pairs such as:

- hidden tests pass versus fail;
- minimal correct patch versus broader regression;
- valid tool call versus malformed call;
- failure followed by diagnosis/recovery versus repeated failure;
- verification performed versus unsupported declaration of success; and
- success within budget versus timeout.

Avoid manufacturing rejected responses solely by adding rude wording or bad
formatting. That teaches style preferences, not software engineering.

Store the evidence used to choose the winner and leave close/ambiguous pairs
out of the first DPO run.

## RL task set

An RL row is prompt-only from the model's perspective but includes a resolvable
task manifest for the environment. Begin with tasks that have:

- deterministic setup;
- fast tests;
- at least one non-trivial decision;
- a clear partial-credit signal where possible; and
- enough baseline headroom to improve without being impossible.

Keep a difficulty ladder based on baseline success:

| Band | Baseline success | Use |
| --- | ---: | --- |
| Trivial | Above 90% | Protocol smoke tests, not most RL batches |
| Learnable | 20–80% | Primary preference/RL curriculum |
| Frontier | Below 20% | Later curriculum after easier tasks improve |

If every response in a GRPO group receives the same reward, the update carries
little useful relative signal. Track the fraction of groups with zero reward
standard deviation.

## Length distribution

Measure rendered prompt and completion tokens before training. Publish at least
p50, p90, p95, p99 and maximum for:

- full sequence length;
- assistant target tokens;
- tool-output tokens;
- tool calls per trajectory; and
- patch size.

Use length buckets and explicit truncation tests. Prefer task-aware context
compaction—summarising old observations while preserving decisions and current
state—over slicing raw tokens.

## Decontamination and splits

1. Split by repository family before row-level processing.
2. Normalize and hash task statements, patches and code windows.
3. Detect exact and near-duplicate prompts/patches across all splits.
4. Exclude known benchmark issues, patches and solution discussions from
   training.
5. Hold out repositories and newer commits for temporal generalisation.
6. Keep hidden test implementations outside any model-visible storage.

Any public benchmark score must state the contamination policy and model/data
cutoff. An unexplained score increase is not sufficient evidence of improved
generalisation.

## Dataset quality report

Every versioned dataset release should include:

- row and token counts by category;
- length percentiles;
- language/repository distribution;
- pass/replay rates;
- tool-call parse rate;
- rejection reasons;
- duplicate and contamination counts;
- source/licence table; and
- the exact tokenizer/template revision used for validation.
