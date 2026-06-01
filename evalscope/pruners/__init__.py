"""
evalscope.pruners
~~~~~~~~~~~~~~~~~
Dataset-level benchmark pruners for efficient go/no-go evaluation.

Pruners select the smallest sample subset that still gives a reliable
capability signal, operating on pre-computed review files rather than
requiring live inference.

Available pruners
-----------------
disagreement / disagreement_pruner
    Inter-model disagreement pruner for LiveCodeBench v5 and AA-LCR.
    Selects samples where models disagree most (maximally discriminative).

mmmu_encoder / mmmu
    MMMU image-encoder probe pruner for forward-looking multimodal assessment.
    Selects samples that stress the image encoder specifically.

Usage
-----
Programmatic::

    from evalscope.pruners import build_pruner

    pruner = build_pruner('disagreement', target_n=75, score_key='pass')
    selected = pruner.select(reviews)   # reviews = list of dicts from review jsonl

CLI::

    python -m evalscope.pruners.cli \\
        --benchmark lcb \\
        --reviews-dir Evals/Part\\ 1/reviews \\
        --target 75 \\
        --out selected_lcb.json
"""

from .base import Pruner, build_pruner, get_pruner, register_pruner, PRUNER_REGISTRY
from .disagreement import DisagreementPruner
from .mmmu import MMMUEncoderPruner

__all__ = [
    'Pruner',
    'build_pruner',
    'get_pruner',
    'register_pruner',
    'PRUNER_REGISTRY',
    'DisagreementPruner',
    'MMMUEncoderPruner',
]
