# Implementation roadmap

This is the full roadmap, not the prerequisite list for the first run. Complete
the [Minimum path to Experiment 1](minimum-path.md) before expanding Milestones
0–2. It deliberately does not assign calendar estimates until the baseline and
4K BF16-LoRA smoke runs establish actual G4 throughput.

## Decision gate: before implementation

The previous open decisions are now front-loaded. The following provisional
defaults are considered accepted for Experiment 1 unless deliberately changed:

| Decision | Default |
| --- | --- |
| GPU | Google Colab G4, RTX PRO 6000 Blackwell, 96 GB |
| Initial language | Python and pytest |
| Harness | Adopt Harbor; build only the Qwen-native agent adapter |
| Baseline suite | 12 trusted tasks from at least 3 held-out repository families |
| Primary deployment | NVFP4 on Blackwell; Dynamic GGUF as portable secondary branch |
| Raw reasoning | Private experiment retention only; not public by default |
| Data publication | Private until source licences and lineage are complete |

This decision gate blocks Experiment 1. Changes require updating the task
manifest, adapter or evaluation budget before results are generated.

## Milestone 0: reproducible foundation

Deliverables:

- Pin Python, PyTorch, CUDA-facing packages, Transformers, TRL, PEFT, Unsloth,
  TorchAO and the model revision.
- Add configuration loading, run manifests and Trackio integration.
- Add GPU/model preflight and peak-memory reporting.
- Add deterministic seeds and durable checkpoint policy.
- Use the idempotent [Colab preflight and baseline notebooks](../notebooks/README.md).
  Keep the reference notebooks as read-only research material; converting all
  of them is deferred.

Exit criteria:

- Environment can be recreated from lock/config files.
- Model loads and runs a native tool-call prompt.
- Baseline notebook can resume after a fresh Colab runtime.
- One BF16 model artifact and one trace are durably persisted.

## Milestone 1: harness and baseline

Deliverables:

- Versioned tool schemas and thin Qwen/Harbor adapter.
- Adopted Harbor isolated backend, or labelled `trusted-dev` pilot fallback.
- Task manifest and trajectory schema.
- Visible/hidden verifier separation.
- Baseline runner and result report.
- Initial 12-task Python/pytest suite across at least 3 repository families.

Exit criteria:

- Infrastructure failures are distinguishable from model failures.
- Tool protocol fixtures pass.
- Stock BF16 has a reproducible baseline report and measured gate cost. A
  published quant is an optional follow-up after the BF16 path is stable.

## Milestone 2: dataset pipeline

Begin this milestone only after Experiment 1 classifies the baseline failures
and measures evaluation cost.

Deliverables:

- Source registry and licence metadata.
- Normalisation to native Qwen messages.
- Execution replay and rejection pipeline.
- Repository-level split and decontamination checks.
- Token-length and quality report.
- Versioned SFT, preference, RL and evaluation datasets.

Exit criteria:

- Every SFT trajectory passes schema/template checks.
- Every accepted repository demonstration replays successfully or carries an
  explicit reviewed exception.
- No evaluation repository family appears in training.

## Milestone 3: SFT

Deliverables:

- Tiny overfit/plumbing run.
- Main BF16-LoRA run at 8K.
- Rank/learning-rate comparison only if the first run leaves a clear question.
- Accepted adapter and merged BF16 candidate.

Exit criteria:

- The candidate passes the SFT gate in [Evaluation](evaluation.md).
- Vision parameters remained frozen.
- Tool-call validity and held-out coding did not regress.

## Milestone 4: preferences

Deliverables:

- Candidate generation from the accepted SFT policy.
- Execution-derived chosen/rejected pairs.
- DPO configuration sweep with memory profiling.
- Accepted or explicitly rejected preference checkpoint.

Exit criteria:

- Improvement is demonstrated on held-out repository tasks.
- Preference accuracy is correlated with execution evidence.
- The model has not merely become more verbose or conservative.

## Milestone 5: agentic RL

Deliverables:

- Unit-tested reward vector.
- Resolution of the Unsloth `trl<=0.24.0` versus TRL
  `environment_factory>=0.29.0` compatibility boundary.
- Short-horizon GRPO/GSPO smoke run.
- Alternating rollout/update scheduler for one GPU.
- Curriculum through medium and long episodes.
- Reward-hacking audit.

Exit criteria:

- The accepted RL checkpoint passes the RL gate.
- Reward increase tracks hidden correctness.
- Results reproduce across independent evaluation runs.

## Milestone 6: quantisation and deployment

Deliverables:

- Merged BF16 golden model.
- Published NVFP4 baseline and optional QAT recovery candidate.
- Dynamic Q8/Q6/Q4/Q3 ladder.
- Q2 experiment only after Q3 passes.
- Runtime benchmarks and artifact manifests.

Exit criteria:

- At least one compact model passes its quantisation gate.
- Tool template, reasoning settings and target runtime are documented.
- Release artifacts are reproducible from the BF16 parent.

## Milestone 7: one-bit research decision

Proceed only if 2-bit results justify further work.

Research questions:

- Does the current 27B architecture load with an available selective 1-bit
  llama.cpp datatype?
- Which tensors must remain at 8/16-bit for tool protocol and long-horizon
  memory?
- Can a coding-specific importance matrix prevent loops and JSON corruption?
- Does quantisation-aware distillation help where existing 4-bit QAT does not
  directly apply?
- Is the saved memory operationally useful on a 96 GB card relative to the
  loss in task success?

Stop if the artifact is incoherent, repeatedly loops, or cannot beat the stock
model on target repository tasks.

## Repository layout

The current executable slice is intentionally smaller than the end state:

```text
docs/                            # decisions, gates and operating guidance
notebooks/                       # eight generated Colab notebooks
references/                      # read-only upstream examples
scripts/build_notebooks.py       # notebook source of truth and validation
scripts/generate_sft_corpus.py   # scripted bootstrap corpus
scripts/collect_trajectories.py  # rejection sampling from a real policy
scripts/evaluate_agent.py        # held-out scorecard, comparison and gate
src/qwen3_8_27b_code/
  episodes.py                    # the one episode loop, model call injected
  tasks.py                       # held-out families and hidden verification
  collection.py                  # attempt filtering and the corpus report
  evaluation.py                  # scorecard, paired comparison, gate
  policies.py                    # scripted policies incl. reward-hack fixtures
tests/                           # generator, notebook and agent contracts
```

The episode loop is shared rather than copied per notebook. Collection and
evaluation must agree about what counts as success; if they drift, the gate
stops measuring the thing the collector optimises.

The following is the future package layout to extract only after a proven
notebook path needs reusable CLI or harness code. It is not a description of
the repository today. Do not create empty modules to match it; add only the
adapter, manifest and evaluation code required by the current experiment.

```text
configs/
  model/
  sft/
  preference/
  rl/
  qat/
  quant/
  eval/
data/
  manifests/
  schemas/
docs/
notebooks/
  00_colab_preflight.ipynb
  01_tool_calling_baseline.ipynb
  02_prepare_sft_data.ipynb
  03_sft_lora.ipynb
  04_dpo_preferences.ipynb
  05_agentic_grpo.ipynb
  06_qat_and_export.ipynb
scripts/
  build_notebooks.py
  prepare_data.py
  train_sft.py
  train_preference.py
  train_rl.py
  merge_adapter.py
  quantize.py
  evaluate.py
src/qwen3_8_27b_code/
  agent/
  data/
  evaluation/
  rewards/
  training/
  quantization/
tests/
  protocol/
  rewards/
  sandbox/
  data/
```

Generated datasets, model weights and episode sandboxes should not be committed
to Git. Store small manifests and hashes in the repository and large artifacts
in controlled object storage or Hugging Face Hub repositories.

## First three experiments

### Experiment 1: baseline and template audit

Follow [Minimum path](minimum-path.md): run 12 trusted Python repository tasks
with the upstream BF16 model, three fixed seeds each because decoding is
sampled. The published Dynamic 4-bit GGUF is an optional follow-up after the
BF16 path works. The objective is to
validate the adopted runner and native adapter, establish failure categories
and measure gate cost—not produce a headline score.

Milestone 1's baseline runner and Milestone 2's replay/rejection pipeline are
now executable: `scripts/collect_trajectories.py` produces verified rows and
`scripts/evaluate_agent.py` scores a candidate against a frozen baseline. What
remains for those milestones is task supply — real repositories with resolvable
revisions and environments — not the machinery around them.

### Experiment 2: 4K SFT smoke

Use a few hundred highly verified trajectories, rank-16 BF16 LoRA and
assistant-only loss. Prove save/merge/reload and check whether tool-call
validity improves without a static-coding regression.

### Experiment 3: 8K capability SFT

Train the documented token mixture with a frozen validation set and frequent
sentinel evaluations. Only after this checkpoint passes should preference data
generation begin.

## Principal risks

| Risk | Mitigation |
| --- | --- |
| New Qwen3.8 support changes rapidly | Pin revisions; maintain load/template/export preflights |
| Training improves imitation but not outcomes | Require execution-verified data and repository gates |
| Benchmark contamination | Repository-level splits, near-duplicate checks and private evaluation |
| Single-GPU RL is too slow or memory-heavy | DPO first; alternating rollouts; shorter curriculum; optional later rollout GPU |
| Reward hacking | Component unit tests, hidden verification and trace audits |
| Long context hides truncation errors | Length reports, bucketing and template-aware compaction tests |
| Quantisation breaks tools before prose | Tool-schema and long-horizon quant gates, BF16 paired traces |
| One-bit work consumes the project | Isolate behind a 2-bit success gate and explicit stop criteria |
| Vision stack is accidentally trained or removed | Parameter assertions and merge/export parity tests |
| User code escapes the harness | Disposable non-root sandboxes with no network or host credentials |

## Decisions required before main SFT

Experiment 1 is unblocked by the defaults above. Before Milestone 2 or main
SFT, confirm:

1. Which additional languages/build systems have enough native-schema data.
2. Whether the private held-out suite expands beyond the pilot repositories.
3. The model/data publication licence and reasoning-retention policy.
4. The measured Colab budget for candidate and release gates.
5. Whether NVFP4 remains the primary deployment target after baseline runtime
   tests.
