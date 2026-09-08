"""Qwen3.8-27B coding-specialisation experiments.

The GPU-bound workflow lives in the generated Colab notebooks. This package
holds the CPU-side tooling: the six-tool schema and repository harness
(importable twins of the notebook cells, drift-guarded by tests), the
generators that bootstrap the execution-verified SFT and preference corpora
in ``data/``, the shared episode loop with its collector and held-out gate,
the multi-file long-horizon task families, and the thinking budget
(``thinking``): shortest-correct selection, reasoning-length preference pairs
and the correctness-gated brevity reward.
"""

__version__ = "0.1.0"
