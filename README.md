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
