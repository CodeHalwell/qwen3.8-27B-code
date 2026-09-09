# Distillation from a larger open model

## What it means here

A larger open model, the *teacher*, attempts the training tasks through the
exact six-tool harness and deployment tool schema. Every attempt is graded
from outside its workspace, and only verified attempts become data for the
27B student. That is the "Regenerable" lane of the
[data strategy](data-strategy.md) applied to a model rather than a public
trace set: the task is kept, the actions are generated through the target
adapter, and nothing is translated or fabricated. The rows are native-schema,
execution-verified trajectories in the teacher's own words.

Three artifacts come out of one teacher run:

| Artifact | Consumer | Built by |
| --- | --- | --- |
| Verified trajectories, shortest-reasoning first | SFT (notebook 02, then 03) | `qwen3_8_27b_code.collection.collect` with a teacher policy |
| Reasoning-length pairs from teacher attempts that verified but thought more than another | DPO (notebook 04), as a minority | `qwen3_8_27b_code.thinking.build_reasoning_length_pairs` |
| Outcome pairs: teacher verified, student did not, from the same state | DPO (notebook 04) | `qwen3_8_27b_code.distillation.build_outcome_pairs` |

`scripts/collect_from_teacher.py` and notebook 08 produce all three. Neither
needs a GPU: the teacher sits behind an endpoint, and the harness runs on
CPU.

## What it does not mean

**Logit-level distillation is out of scope.** Matching the teacher's token
distribution needs the same tokenizer, and Kimi and GLM do not share
Qwen3.8's. A larger Qwen3.8 would, but a 2.4T-class teacher cannot be
co-hosted with the 27B student on a 96 GB card, and vendor endpoints return
text, not logits. Sequence-level distillation on verified outputs is the
route that fits the hardware and the data policy. Revisit only if a
same-tokenizer teacher small enough to serve next to the student appears.

**The teacher does not decide correctness.** The hidden verifiers do. A
teacher trajectory that ends green because the teacher edited the tests is
rejected exactly as a student's would be (`protected_files_modified`).

## Teacher candidates

| Teacher | How to reach it | Notes |
| --- | --- | --- |
| A larger Qwen3.8 | Hugging Face Inference Providers router, a vendor endpoint, or vLLM if a size fits a spare card | Same chat template and tool syntax as the student; `reasoning_effort` is honoured, so the teacher can be run at the effort the rows will be labelled with |
| Kimi K3 | Moonshot API (`moonshot` preset) or the router | 2.8T-parameter MoE with 104B active per its paper; thinking models return `reasoning_content`, and multi-turn tool use expects earlier reasoning passed back, which the adapter does by default |
| GLM 5.3 / 5.3 Flash | Z.ai API (`zai` preset) or the router; Flash may be self-hostable, check its card | Enable thinking through the request body per the vendor docs (`--extra-body`); the probe reports whether reasoning came back |

The presets in `qwen3_8_27b_code.teachers.PRESETS` carry the base URLs read
from the vendors' documentation on the date in [References](references.md).
Endpoints move; `--base-url` and `--api-key-env` override any preset. Model
ids are whatever the endpoint calls them and are deliberately not hard-coded
anywhere in this repository.

## The adapter

`qwen3_8_27b_code.teachers` speaks the OpenAI-compatible chat-completions
protocol with the standard library only, so the package keeps no runtime
dependencies. Outgoing, the episode loop's messages are folded into the
protocol's shape with tool-call ids threaded through and the developer
message folded into the system message. Incoming, the structured response is
rendered back into Qwen3.8's XML tool-call text, so the loop parses, stores
and grades a teacher turn exactly as a student turn. Truncation
(`finish_reason == "length"`) becomes the loop's `output_truncated` fault,
reasoning tokens are read from the usage block when the endpoint reports
them, and a malformed arguments string becomes a typed `invalid_tool_call`
observation rather than an adapter crash.

Probe before spending anything:

```bash
uv run --group dev python scripts/collect_from_teacher.py \
    --preset moonshot --model <model-id> --probe
```

The probe reports whether the endpoint answered a tool-bearing prompt with a
native tool call, whether its reasoning was visible, and whether it counted
reasoning tokens. A teacher is usable only when the first two hold.

## Reasoning must be visible

A teacher turn without its reasoning is refused by default
(`require_reasoning=True`), and an endpoint that hides reasoning turns every
episode into an infrastructure failure with a clear message. The reason is
not fastidiousness. The chat template renders an assistant turn's reasoning
into its think block; an empty one, labelled `medium`, teaches the student
that medium effort means not thinking. That is a fabricated signal, and the
data strategy forbids fabricated content. Switch the requirement off only for
an endpoint known to omit the field when the reasoning was genuinely empty.

## Labelling effort

The rows carry a `reasoning_effort` label the student is trained to honour.
The teacher's reasoning length is not the student's, and Kimi and GLM have
no effort dial to set. Before choosing `--effort-label`:

1. Run the student's effort ladder from the [thinking budget](thinking-budget.md)
   and note its reasoning tokens per turn at `low`, `medium` and `xhigh`.
2. Run the teacher on a handful of tasks and read the collection report's
   `thinking` section.
3. Label the teacher rows with the student effort whose budget they resemble.
   Teacher rows several times longer than the student's `medium` labelled
   `medium` move the student's budget up, and the thinking gate will refuse
   the checkpoint.

Shortest-reasoning selection applies to teacher attempts as it does to the
student's, so more attempts per task buy shorter verified rows, not only
more rows. A Qwen3.8 teacher can also be run at the target effort directly
with `--reasoning-effort`.

## Outcome pairs

`build_outcome_pairs` takes any mix of attempts and, at each task, pairs
every verified attempt against every failed one at the first assistant turn
where their actions diverge. The verified continuation is chosen. Both
sides were executed and graded, so the pair is execution-derived whichever
policy produced each side, and the evidence names both policies, both
verdicts and the failure's rejection reason.

Supply the student's `attempts.jsonl` (written by notebook 07 and
`scripts/collect_trajectories.py`) with `--student-attempts` to get
teacher-versus-student pairs. Without it the pairs are teacher-versus-teacher
across seeds, and the student collector writes student-versus-student pairs
of its own. The report counts form-matched pairs (a tool call on both sides)
separately from prose rejections, because the [data strategy](data-strategy.md)
requires prose rejections to stay a minority.

## Licences and terms

Open weights and vendor APIs are licensed separately. Check the teacher's
weight licence for downstream-model clauses, and, for a vendor endpoint, its
terms on using output to train other models, before any artifact of a
teacher run is published or a checkpoint trained on it is released. Record
the answer in the dataset's quality report alongside the source/licence table
the data strategy already requires.

## Cost

```text
teacher cost = tasks * attempts * (prompt tokens + completion tokens per episode) * price
```

Prompt tokens grow with every tool observation in the transcript, so an
episode costs several times its first turn. Measure one task before enabling
a sweep, and set `--max-tokens` and the episode budget to the shape of the
tasks rather than the endpoint's maximum.

## Experiment order

1. **Probe** the endpoint; confirm tool calls and visible reasoning.
2. **Price** one task at one attempt; read the `thinking` section and choose
   the effort label.
3. **Collect** with three attempts per task and shortest-reasoning selection;
   persist attempts.
4. **SFT** the student on teacher rows mixed with its own verified rows, and
   run the gate with the thinking check.
5. **DPO** on the outcome pairs (teacher against the pre-SFT student's
   attempts) and the execution-derived pairs, with length pairs a minority.
6. **Iterate**: the post-SFT student's own attempts become the next
   `--student-attempts`, so the outcome pairs track the student's current
   errors rather than its first ones.

Stop when teacher rows fail the thinking gate at the chosen label, when the
teacher's acceptance rate is below the student's on the same tasks (a weaker
teacher on this harness teaches nothing), or when licence terms cannot be
resolved.
