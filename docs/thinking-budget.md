# Thinking budget

## Objective

Qwen3.8-27B is a hybrid-thinking model. On repository tasks the tokens it
spends inside the think block dominate what it generates, so they dominate
latency and cost at deployment. The objective of this document is to cut
reasoning tokens per turn **without losing held-out task success**, and to
make the second half of that sentence something the gate measures rather
than something a training run hopes for.

Everything below is conditioned on verified success. A shorter attempt is
only preferred over a longer one when both passed the same verifier; a
brevity reward is only paid to a correct sample; and a checkpoint that thinks
less but solves fewer tasks fails the existing gate before the thinking check
is even read.

## What is measured

The episode loop (`qwen3_8_27b_code.episodes`) keeps the books, and the
model-backed policy in notebook 07 supplies the counts:

| Quantity | Definition | Where it comes from |
| --- | --- | --- |
| `reasoning_tokens` | Tokens generated up to and including `</think>`, summed over the episode | The policy counts them from the generated ids; a turn cut off before `</think>` is entirely reasoning |
| `reasoning_chars` | Characters inside the think block | Always available; the CPU-side proxy used by scripted policies and the pair builder when tokens are absent |
| `thinking_overrun` | The turn hit the per-turn token cap before closing its think block | The loop, from the truncation fault and the absence of `</think>` |
| `turn_reasoning_tokens` | Per completed assistant turn | The loop, aligned with the assistant messages |

The scorecard (`qwen3_8_27b_code.evaluation`) reports:

- `reasoning_tokens_per_turn` and `reasoning_chars_per_turn`;
- `reasoning_share_of_completion`, the fraction of generated tokens spent
  thinking;
- `completion_tokens_per_turn`;
- `thinking_overrun_rate`, the fraction of episodes that died inside a think
  block; and
- `horizon_bands` and `success_by_horizon`, so a brevity change can be read
  per band.

Thinking is normalised **per assistant turn**, not per success. A candidate
that solves more tasks legitimately takes more turns, and a per-success ratio
against a baseline with no successes is undefined. Per turn is also the budget
the user pays at deployment: one decision at a time.

## The gate

`evaluation.gate()` carries two thinking checks alongside the quality checks
from [Evaluation](evaluation.md):

| Check | Passes when | Notes |
| --- | --- | --- |
| `thinking_budget` | Candidate reasoning tokens per turn ≤ baseline × (1 + growth), default growth 10% | Evaluated only when both policies counted tokens; otherwise reported as *not measured* and passed, so a scripted or older report cannot fail a run it could not measure |
| `thinking_overrun_no_worse` | Overrun rate did not increase | Always measurable: it comes from the termination reason |

The tolerance is a policy setting, frozen before a candidate's results are
seen, like every other threshold:

```bash
uv run --group dev python scripts/evaluate_agent.py compare \
    reports/baseline.json reports/candidate.json --max-reasoning-growth 0.10
```

`--ignore-thinking-budget` drops the check for a comparison where thinking is
deliberately not under test. The comparison also records a `thinking`
section: tokens per turn on each side, the relative change, the reasoning
share, and success by horizon band.

The check is a ceiling, not a target. A candidate that thinks 40% less at
equal success passes with room to spare; the ceiling exists to stop a
checkpoint that "improved" by thinking longer from being promoted as a win.

## The levers, in the order to pull them

### 1. The effort dial (costs nothing)

Qwen3.8 exposes `reasoning_effort` (`low`, `medium`, `xhigh`) through its
chat template. Before any training, run the held-out suite at each effort
and tabulate success against reasoning tokens per turn. That table decides
the deployment default and shows how much of the budget the dial alone
recovers. It also tells you where the model's own judgement is miscalibrated:
tasks that succeed at `low` but are thought about at `xhigh` length are the
cases the training levers below exist for.

The corpus keeps the effort each row was generated at, and the collector
records the effort it ran at, so training preserves the dial rather than
flattening it. Never relabel a row to a different effort; the scripted
bootstrap corpus is labelled `low` and `medium` for exactly this reason.

### 2. Shortest rejection sampling (SFT data)

`collect()` in `qwen3_8_27b_code.collection` attempts each task several
times and keeps only verified attempts. With
`selection="shortest_reasoning"` (the default), the rows kept under the
per-task cap are the verified attempts that reasoned least, and when two
attempts took the same actions the copy that thought less is the one that
survives deduplication. SFT then teaches the shortest path the model has
itself shown to work, in its own words, at the effort it ran at.

The collection report's `thinking` section splits reasoning per turn by
outcome. If the attempts that failed thought far more than the ones that
verified, the policy is spending tokens on the tasks it cannot do, and a
tighter budget costs little. If the reverse holds, brevity is being bought
with correctness and the thinking check above is the thing to watch.

```bash
uv run --group dev python scripts/collect_trajectories.py \
    --policy my_policies:unsloth_policy --suite training --attempts 3 \
    --selection shortest_reasoning --max-rows-per-task 2
```

### 3. Reasoning-length preference pairs (DPO)

`qwen3_8_27b_code.thinking.build_reasoning_length_pairs` turns the verified
attempts at one task into pairs in notebook 04's format. Two attempts are
walked in step while they take identical actions; at each such turn the
continuation that reached the action with less reasoning is *chosen*, the
other *rejected*. Both sides succeeded, both carry a think block, and both
carry the same tool call, so the only thing DPO can learn from the pair is to
think less before the same decision. A pair is emitted only when the longer
side thinks at least 1.5× as much and at least 32 tokens (or 120 characters)
more, so near-identical continuations do not become noise.

The collector writes these pairs next to the corpus
(`data/collected/length_pairs.jsonl`, with a quality report). Notebook 04
loads them through `PREFERENCE_LOCAL_JSONL` like the execution-derived pairs.

Two limits are recorded in every report. For turns after the first, the
rejected turn was generated under its own prefix, whose earlier reasoning and
observation text differ from the chosen prefix the pair renders; the actions
are identical, the token-level context is not. And these pairs teach brevity
only: keep them a minority of the DPO mixture (no more than roughly a third)
next to the execution-derived correctness pairs in `data/preferences`, or the
preference stage learns "shorter" more strongly than it learns "right".

### 4. A correctness-gated brevity term (RL)

`qwen3_8_27b_code.thinking.length_rewards` is the group-relative length
reward of Kimi k1.5's long2short recipe: within one GRPO group the shortest
sample earns `+weight/2`, the longest `-weight/2`, linearly in between, and an
incorrect sample is clipped to at most zero. With the environment's hidden
pass worth 0.8 and the weight at 0.1, a correct sample always outranks an
incorrect one: the term reorders correct samples among themselves and does
nothing else. Notebook 05 carries the same function with fixtures that pin
those properties, and the test suite pins the notebook copy to the package.

It is wired in only when the agentic trainer path opens (see the recorded
Unsloth/TRL blocker in the [training plan](training-plan.md)).

### What not to do

- Do not cap reasoning tokens during training as a substitute for the levers
  above. A hard cap teaches the model to be truncated, and the scorecard
  will show it as `thinking_overrun_rate`.
- Do not penalise length on failed attempts. That rewards giving up, which
  is why `length_rewards` clips incorrect samples at zero and the collector
  only ever selects among verified successes.
- Do not build length pairs from one verified and one failed attempt; the
  execution-derived pairs already cover correctness contrasts and keep the
  two signals separable.
- Do not relabel scripted rows to a lower effort to "teach brevity". The
  effort label is an instruction the model must keep honouring.

## Quantisation

Low-bit quantisation degrades tool protocol and loops before it degrades
prose, and a looping model thinks longer. The quantisation gate runs the same
scorecard against the BF16 parent, so `thinking_overrun_rate` and reasoning
tokens per turn belong in the quant diagnostics of the
[quantisation strategy](quantisation.md): a quant that keeps episode success
but doubles reasoning per turn has not kept the deployment properties this
project is optimising for.

## Long horizon

The expected failure mode of any brevity lever is that it hurts the harder
tasks first: a model taught to think less on three-call fixes may stop
inspecting enough on seven-call multi-file ones. The held-out suite now
carries two multi-file families (`qwen3_8_27b_code.long_horizon`) that land
in the medium band, and the scorecard reports `success_by_horizon`. Read the
medium band before the aggregate: a brevity change that holds the aggregate
but drops the medium band has been paid for with exactly the capability
this project exists to build.

## Experiment order

1. **Effort ladder on the stock model.** Held-out suite at `low`, `medium`
   and `xhigh`; record success, reasoning tokens per turn and overrun rate
   per effort. Choose the deployment effort and the baseline report for the
   gate.
2. **Collection with shortest-reasoning selection**, then SFT, then the gate
   with the thinking check at the frozen tolerance.
3. **DPO** on the execution-derived pairs plus a minority of length pairs
   from the collection; gate again, reading the medium band.
4. **RL brevity term** once the trainer path is available, with the fixtures
   in notebook 05 as the contract.

Stop and roll back when `thinking_overrun_rate` rises, when the medium band
drops while the short band holds, or when the thinking check passes only
because success fell.

## References

- Kimi Team, *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — the
  long2short methods: shortest rejection sampling, long-versus-short DPO and
  the group-relative length reward used here.
- Aggarwal and Welleck, *L1: Controlling How Long a Reasoning Model Thinks
  with Reinforcement Learning* — length-controlled policy optimisation, the
  RL-side counterpart of Qwen3.8's native effort dial.
