# Qwen3.8-27B code specialist

This project explores supervised fine-tuning, preference optimisation,
agentic reinforcement learning and low-bit deployment of Qwen3.8-27B for
long-horizon software-engineering work.

Training is intended to run in a Google Colab notebook on a G4 runtime with an
NVIDIA RTX PRO 6000 Blackwell Server Edition GPU and 96 GB VRAM.

Start with:

- [Colab notebook suite](notebooks/README.md)
- [Documentation index](docs/README.md)
- [Minimum path to the first baseline experiment](docs/minimum-path.md)
- [Implementation roadmap](docs/roadmap.md)

The immediate objective is not to build the full training platform. It is to
run a small, reproducible upstream baseline through the intended native tool
schema, measure episode cost and use the result to decide what infrastructure
and data work is justified next.

The repository contains the planning documents, upstream notebook references
and a restartable Colab suite. Start at notebook 00; all training, upload and
large export switches are disabled until their preceding gate passes.

Regenerate and validate the notebooks locally with:

```bash
uv run --group dev python scripts/build_notebooks.py
uv run --group dev pytest -q
```

Score a checkpoint against the frozen baseline on held-out tasks, and collect
verified trajectories from a real policy, with:

```bash
uv run --group dev python scripts/evaluate_agent.py run --policy gold --out reports/candidate.json
uv run --group dev python scripts/evaluate_agent.py compare reports/baseline.json reports/candidate.json
uv run --group dev python scripts/collect_trajectories.py --policy gold --attempts 3
```

The `gold` policy is a scripted stand-in that exercises the whole path on CPU.
Supply a model-backed policy as `module:attribute`, or use notebook 07. The
comparison exits non-zero when the gate fails.

The execution-verified bootstrap training data in `data/` (native-schema SFT
trajectories and preference pairs, with quality reports) regenerates with:

```bash
uv run --group dev python scripts/generate_sft_corpus.py
uv run --group dev python scripts/generate_preference_pairs.py
uv run --group dev --with "transformers==5.3.0" --with jinja2 \
    python scripts/validate_dataset_rendering.py
```
