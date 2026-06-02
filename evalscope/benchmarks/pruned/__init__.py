"""
evalscope.benchmarks.pruned
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-registered pruned benchmark variants for LiveCodeBench v5 and AA-LCR.

Importing this package registers the following benchmark names:

  ``live_code_bench_pruned``
      75-sample pruned variant of LiveCodeBench v5 (from 315).
      Pass ``indices_file`` pointing to pruner output JSON, or
      ``indices`` as an explicit list.

  ``aa_lcr_pruned``
      30-sample pruned variant of AA-LCR (from 100).
      Uses judge-noise-guard-compatible index selection.

  ``mmmu_pruned``
      110-sample encoder-stress probe for MMMU (from 660 reference rows).

All three are registered at import time so they are discoverable via
``evalscope.api.registry.BENCHMARK_REGISTRY`` without any extra setup.

To register additional pruned benchmarks at runtime::

    from evalscope.benchmarks.pruned import register_pruned_benchmark
    register_pruned_benchmark('gpqa', indices_file='selected_gpqa.json')
"""

from .pruned_adapter import PrunedDataAdapter, register_pruned_benchmark

# -------------------------------------------------------------------------
# Register the three pre-built pruned benchmarks.
# Indices are not baked in here -- callers supply them via indices_file or
# indices in their task config / extra_params.
# -------------------------------------------------------------------------

register_pruned_benchmark('live_code_bench', pruned_name='live_code_bench_pruned')
register_pruned_benchmark('aa_lcr',          pruned_name='aa_lcr_pruned')
register_pruned_benchmark('mmmu',            pruned_name='mmmu_pruned')

__all__ = [
    'PrunedDataAdapter',
    'register_pruned_benchmark',
]
