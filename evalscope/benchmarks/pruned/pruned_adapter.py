"""
evalscope.benchmarks.pruned.pruned_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Universal pruned benchmark adapter.

This module provides a single ``PrunedDataAdapter`` that wraps *any* existing
evalscope benchmark and restricts evaluation to a pre-selected subset of sample
indices.  The subset is produced by the disagreement-based pruners in
``evalscope.pruners`` and stored as a JSON file.

Design
------
Rather than hard-coding pruned variants of individual benchmarks, we register a
*factory function* ``register_pruned_benchmark`` that creates a new
``BenchmarkMeta`` / adapter pair for any base benchmark at import time.  Callers
can then refer to the pruned variant by name (e.g. ``live_code_bench_pruned``,
``aa_lcr_pruned``) in their evalscope task config.

Usage
-----
In your evalscope task config::

    datasets:
      - name: live_code_bench_pruned
        dataset_args:
          live_code_bench_pruned:
            indices_file: /path/to/selected_lcb.json
            # or pass indices directly:
            # indices: [4, 5, 6, 7, ...]

Or programmatically::

    from evalscope.benchmarks.pruned import register_pruned_benchmark
    register_pruned_benchmark('aa_lcr', indices=[1, 5, 10, ...])

The pruned adapter is a transparent wrapper: it delegates all data loading,
prompt construction, and scoring to the base adapter, and simply skips samples
whose index is not in the selected set.

Defensibility
-------------
The adapter is index-agnostic -- it applies whatever index list you hand it.
The *quality* of those indices comes from the pruner (see ``evalscope.pruners``),
not from this adapter.  A fourth model evaluated against the same index list will
get a fair evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Dataset, MemoryDataset, Sample
from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.utils.logger import get_logger

logger = get_logger()


class PrunedDataAdapter(DefaultDataAdapter):
    """A transparent wrapper around any DefaultDataAdapter that restricts
    evaluation to a pre-selected subset of sample indices.

    Args:
        indices:      Explicit list of integer indices to keep.
        indices_file: Path to a JSON file produced by ``evalscope.pruners.cli``
                      (contains a ``selected_indices`` key).
        **kwargs:     Forwarded to the base adapter constructor.

    Either ``indices`` or ``indices_file`` must be provided; if both are given,
    ``indices`` takes precedence.
    """

    def __init__(
        self,
        indices: Optional[List[int]] = None,
        indices_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._selected: Set[int] = set()

        # Resolve indices
        if indices is not None:
            self._selected = set(int(i) for i in indices)
        elif indices_file is not None:
            path = Path(indices_file)
            if not path.exists():
                raise FileNotFoundError(
                    f'PrunedDataAdapter: indices_file not found: {indices_file}'
                )
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            raw = data.get('selected_indices', [])
            # Support both int indices (LCB/AA-LCR) and "Subject__idx" strings (MMMU)
            for item in raw:
                if isinstance(item, int):
                    self._selected.add(item)
                elif isinstance(item, str) and '__' in item:
                    # MMMU format: "Accounting__2" -- use local index
                    try:
                        self._selected.add(int(item.split('__')[1]))
                    except (IndexError, ValueError):
                        pass
                else:
                    try:
                        self._selected.add(int(item))
                    except (TypeError, ValueError):
                        pass
        else:
            raise ValueError(
                'PrunedDataAdapter requires either `indices` or `indices_file`.'
            )

        if not self._selected:
            raise ValueError('PrunedDataAdapter: selected index set is empty.')

        logger.info(
            f'PrunedDataAdapter: will evaluate {len(self._selected)} '
            f'of the available samples.'
        )

    def load(self) -> Dataset:
        """Load the full dataset then filter to selected indices."""
        full_dataset = super().load()
        pruned_samples = [
            s for s in full_dataset
            if s.id is not None and int(s.id) in self._selected
        ]

        if not pruned_samples:
            # Fallback: try matching by position if id-based match fails
            logger.warning(
                'PrunedDataAdapter: id-based filtering returned 0 samples. '
                'Falling back to position-based filtering.'
            )
            pruned_samples = [
                s for i, s in enumerate(full_dataset)
                if i in self._selected
            ]

        logger.info(
            f'PrunedDataAdapter: {len(full_dataset)} → {len(pruned_samples)} samples '
            f'({100.0 * len(pruned_samples) / max(len(full_dataset), 1):.1f}% retained)'
        )

        return MemoryDataset(
            samples=pruned_samples,
            name=full_dataset.name,
            location=getattr(full_dataset, 'location', None),
        )


def register_pruned_benchmark(
    base_name: str,
    indices: Optional[List[int]] = None,
    indices_file: Optional[str] = None,
    pruned_name: Optional[str] = None,
) -> str:
    """Register a pruned variant of an existing benchmark.

    Args:
        base_name:    Name of the registered base benchmark (e.g. ``'live_code_bench'``).
        indices:      Explicit index list (optional).
        indices_file: Path to pruner output JSON (optional).
        pruned_name:  Name to register the pruned variant under.
                      Defaults to ``f'{base_name}_pruned'``.

    Returns:
        The registered name of the pruned benchmark.

    Raises:
        ValueError: If the base benchmark is not registered.
    """
    if base_name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Base benchmark '{base_name}' is not registered. "
            f'Available: {sorted(BENCHMARK_REGISTRY.keys())}'
        )

    pruned_name = pruned_name or f'{base_name}_pruned'

    # Skip if already registered (idempotent)
    if pruned_name in BENCHMARK_REGISTRY:
        logger.debug(f"Pruned benchmark '{pruned_name}' already registered.")
        return pruned_name

    base_meta = BENCHMARK_REGISTRY[base_name]

    # Build pruned BenchmarkMeta inheriting everything from base
    pruned_meta = BenchmarkMeta(
        name=pruned_name,
        pretty_name=f'{base_meta.pretty_name} (Pruned)',
        tags=base_meta.tags,
        description=(
            f'Pruned variant of {base_meta.pretty_name}. '
            f'Evaluates only the most discriminative samples selected by '
            f'the disagreement-based pruner (evalscope.pruners). '
            f'See the base benchmark for full documentation.\n\n'
            + (base_meta.description or '')
        ),
        dataset_id=base_meta.dataset_id,
        subset_list=base_meta.subset_list,
        metric_list=base_meta.metric_list,
        few_shot_num=base_meta.few_shot_num,
        train_split=base_meta.train_split,
        eval_split=base_meta.eval_split,
        prompt_template=base_meta.prompt_template,
        extra_params={
            **(base_meta.extra_params or {}),
            'indices': {
                'type': 'list[int] | null',
                'description': 'Explicit list of sample indices to evaluate.',
                'value': indices,
            },
            'indices_file': {
                'type': 'str | null',
                'description': 'Path to pruner output JSON with selected_indices key.',
                'value': indices_file,
            },
        },
    )

    # Create a subclass of the base adapter that mixes in PrunedDataAdapter behaviour
    base_adapter_cls = base_meta.data_adapter

    class _PrunedAdapter(PrunedDataAdapter, base_adapter_cls):  # type: ignore[valid-type]
        """Auto-generated pruned adapter for ``{base_name}``."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Pull our params from extra_params / kwargs
            ep = kwargs.get('benchmark_meta', pruned_meta).extra_params or {}
            _indices = ep.get('indices', {}).get('value') or kwargs.pop('indices', indices)
            _indices_file = ep.get('indices_file', {}).get('value') or kwargs.pop('indices_file', indices_file)
            PrunedDataAdapter.__init__(
                self,
                indices=_indices,
                indices_file=_indices_file,
            )
            base_adapter_cls.__init__(self, *args, **kwargs)

    _PrunedAdapter.__name__ = f'Pruned{base_adapter_cls.__name__}'
    _PrunedAdapter.__qualname__ = _PrunedAdapter.__name__

    # Register using the decorator pattern evalscope expects
    register_benchmark(pruned_meta)(_PrunedAdapter)

    logger.info(f"Registered pruned benchmark: '{pruned_name}'")
    return pruned_name
