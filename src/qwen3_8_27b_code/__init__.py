"""Qwen3.8-27B coding-specialisation experiments.

The GPU-bound workflow lives in the generated Colab notebooks. This package
holds the CPU-side tooling: the six-tool schema and repository harness
(importable twins of the notebook cells, drift-guarded by tests), and the
generators that bootstrap the execution-verified SFT and preference corpora
in ``data/``.
"""

__version__ = "0.1.0"
