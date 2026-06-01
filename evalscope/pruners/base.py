"""
evalscope.pruners.base
~~~~~~~~~~~~~~~~~~~~~~
Base class and registry helpers for dataset-level pruners.

Pruners operate on *datasets* (collections of pre-computed review rows) rather
than on individual model responses.  The contract mirrors evalscope's Filter
abstraction:

  * Subclass ``Pruner`` and implement ``select()``.
  * Decorate with ``@register_pruner('my_name')`` so the registry can resolve it
    by name (same pattern as ``@register_filter``).
  * Call ``build_pruner(name, **kwargs)`` to instantiate from config.

The ``PRUNER_REGISTRY`` follows the same ``Registry`` base used by every other
evalscope registry (filters, metrics, benchmarks …) so it integrates cleanly
with the rest of the framework.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union

from evalscope.api.registry import Registry

# Registry

PRUNER_REGISTRY: Registry = Registry('Pruner')


def register_pruner(name: Union[str, List[str]]):
    """Decorator that registers a Pruner class under one or more names.

    Usage::

        @register_pruner('disagreement')
        class DisagreementPruner(Pruner):
            ...
    """
    return PRUNER_REGISTRY.register(name)


def get_pruner(name: str) -> type:
    """Retrieve a registered Pruner class by name."""
    return PRUNER_REGISTRY.lookup(name)


def build_pruner(name: str, **kwargs: Any) -> 'Pruner':
    """Instantiate a registered pruner by name.

    Args:
        name:    Registry key (e.g. ``'disagreement'``).
        **kwargs: Constructor arguments forwarded to the pruner class.

    Returns:
        Instantiated :class:`Pruner`.
    """
    cls = get_pruner(name)
    return cls(**kwargs)


# Abstract base

class Pruner(abc.ABC):
    """Abstract base class for benchmark pruners.

    A Pruner receives a list of *review rows* (dicts parsed from the
    ``reviews/<model>.jsonl`` files) and returns the subset of *indices*
    that should be retained for evaluation.

    Subclasses must implement :meth:`select`.
    """

    @abc.abstractmethod
    def select(self, reviews: List[Dict[str, Any]]) -> List[int]:
        """Return the list of sample indices to keep.

        Args:
            reviews: List of review dicts.  Each dict must contain at least
                     ``'index'`` (int) and ``'sample_score'`` (dict).  The
                     score key inside ``sample_score['score']['value']`` is
                     benchmark-specific (``'pass'`` for LCB, ``'acc'`` for
                     AA-LCR).

        Returns:
            Sorted list of ``index`` values to retain.
        """
        ...

    def __call__(self, reviews: List[Dict[str, Any]]) -> List[int]:
        """Allow pruner instances to be called directly."""
        return self.select(reviews)

    def summary(self, original: int, selected: int) -> str:
        """Human-readable compression summary."""
        pct = 100.0 * selected / original if original else 0.0
        return (
            f'{self.__class__.__name__}: {original} → {selected} samples '
            f'({pct:.1f}% retained)'
        )
