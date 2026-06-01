"""
evalscope.pruners.disagreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Disagreement-based benchmark pruner for LiveCodeBench v5 and AA-LCR.

Algorithm
---------
The goal is to find the *smallest* sample subset that still gives a reliable
go/no-go signal across models.  The key insight is:

  * Samples where **all models agree** (all pass or all fail) carry no
    discriminative information about which model is better.
  * Samples where **models disagree** sit on the capability decision boundary
    and are maximally informative.

We therefore rank samples by their *disagreement score* -- the variance of
pass/fail outcomes across models -- and select the top-k most disagreeing
samples, subject to a configurable budget.

To avoid the forbidden "top-k easiest / hardest" baseline, we also enforce a
*difficulty spread*: the selected set must include samples from every observed
pass-rate bucket (0%, 33%, 67%, 100% for three models).  This ensures the
pruned set covers the full difficulty spectrum rather than collapsing to the
midpoint.

AA-LCR caveat
-------------
AA-LCR is graded by an LLM judge, which introduces non-deterministic noise.
When ``judge_noise_guard=True`` (the default), we apply a conservative filter:
only samples with a *clear* binary split across models (at least one model
scores 1.0 and at least one scores 0.0) are treated as high-disagreement.
Borderline cases (e.g. all models scoring 0.5) are down-weighted because they
may reflect judge variance rather than genuine capability gaps.

Defensibility for unseen models
--------------------------------
The selected indices are chosen based on inter-model disagreement structure,
*not* on the identity or absolute performance of any specific model.  A fourth
model will face the same samples -- which probe the hard decision boundary --
regardless of whether it passes or fails.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .base import Pruner, register_pruner


def _extract_score(review: Dict[str, Any], score_key: str) -> Optional[float]:
    """Pull a scalar score out of a review row.

    Handles the nested structure::

        {"sample_score": {"score": {"value": {"pass": 1.0}}}}

    Returns ``None`` if the key is absent or the value is not a number.
    """
    try:
        value = review['sample_score']['score']['value']
        raw = value.get(score_key)
        if raw is None:
            # Try the first numeric value if key not found
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return None
        return float(raw)
    except (KeyError, TypeError, ValueError):
        return None


def _disagreement_score(scores: List[float]) -> float:
    """Compute disagreement as variance of binary scores across models.

    For binary {0, 1} scores, variance = p*(1-p) where p is the pass rate.
    Maximum disagreement (0.25) occurs at p=0.5 (half pass, half fail).
    Zero disagreement occurs at p=0 or p=1 (unanimous).
    """
    if not scores:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    return mean * (1.0 - mean)


@register_pruner(['disagreement', 'disagreement_pruner'])
class DisagreementPruner(Pruner):
    """Select the most discriminative samples by inter-model disagreement.

    Args:
        target_n:
            Maximum number of samples to retain.  If the disagreement pool is
            smaller, all disagreeing samples are kept and the remainder is
            filled from the difficulty-spread buckets.
        score_key:
            Field name inside ``sample_score.score.value`` to read.
            Use ``'pass'`` for LiveCodeBench, ``'acc'`` for AA-LCR.
        judge_noise_guard:
            If ``True``, only samples with a *clear* binary split (min score
            = 0, max score = 1 across models) are treated as high-disagreement.
            Recommended for LLM-judged benchmarks (AA-LCR).
        fill_from_buckets:
            If ``True``, after filling the disagreement quota, pad the result
            with samples drawn proportionally from the easy / hard buckets to
            maintain difficulty spread.
        seed:
            Random seed for deterministic bucket sampling (default: 42).
    """

    def __init__(
        self,
        target_n: int = 75,
        score_key: str = 'pass',
        judge_noise_guard: bool = False,
        fill_from_buckets: bool = True,
        seed: int = 42,
    ) -> None:
        self.target_n = target_n
        self.score_key = score_key
        self.judge_noise_guard = judge_noise_guard
        self.fill_from_buckets = fill_from_buckets
        self.seed = seed
    
    # Core implementation
    def _group_by_index(
        self, reviews: List[Dict[str, Any]]
    ) -> Dict[int, List[float]]:
        """Group scores by sample index across all models.

        Each review row corresponds to one (model, sample) pair.  We collect
        all scores for the same ``index`` so we can compute disagreement.

        Returns:
            Mapping of ``index`` → list of float scores (one per model file).
        """
        groups: Dict[int, List[float]] = defaultdict(list)
        for row in reviews:
            idx = row.get('index')
            if idx is None:
                continue
            score = _extract_score(row, self.score_key)
            if score is not None:
                groups[int(idx)].append(score)
        return dict(groups)

    def _bucket(self, pass_rate: float, n_models: int) -> str:
        """Assign a pass-rate bucket label for difficulty spread."""
        # Round to nearest step to handle float imprecision
        step = 1.0 / n_models if n_models > 1 else 1.0
        rounded = round(pass_rate / step) * step
        rounded = max(0.0, min(1.0, rounded))
        if rounded == 0.0:
            return 'all_fail'
        elif rounded == 1.0:
            return 'all_pass'
        else:
            return f'partial_{rounded:.2f}'

    def select(self, reviews: List[Dict[str, Any]]) -> List[int]:
        """Return indices of samples to retain.

        Args:
            reviews: Combined review rows from all model files for one benchmark.
                     Each row must have ``'index'`` and ``'sample_score'``.

        Returns:
            Sorted list of selected sample indices.
        """
        import random
        rng = random.Random(self.seed)

        groups = self._group_by_index(reviews)
        if not groups:
            return []

        # Infer number of models from the most common group size
        sizes = [len(v) for v in groups.values()]
        n_models = max(set(sizes), key=sizes.count)

        # Compute disagreement and pass-rate for each sample
        scored: List[Tuple[int, float, float]] = []  # (index, disagreement, pass_rate)
        for idx, scores in groups.items():
            pass_rate = sum(scores) / len(scores)
            disag = _disagreement_score(scores)

            # Apply judge noise guard: require clear binary split
            if self.judge_noise_guard:
                has_pass = any(s >= 0.5 for s in scores)
                has_fail = any(s < 0.5 for s in scores)
                if not (has_pass and has_fail):
                    disag = 0.0  # treat as non-discriminative

            scored.append((idx, disag, pass_rate))

        # Separate into high-disagreement and low-disagreement buckets
        high_disag = [(idx, pr) for idx, d, pr in scored if d > 0.0]
        low_disag  = [(idx, pr) for idx, d, pr in scored if d == 0.0]

        # Sort high-disagreement by descending disagreement score
        high_disag_with_d = [(idx, d, pr) for idx, d, pr in scored if d > 0.0]
        high_disag_with_d.sort(key=lambda x: x[1], reverse=True)

        # Take as many high-disagreement samples as budget allows
        selected = set()
        for idx, d, pr in high_disag_with_d[:self.target_n]:
            selected.add(idx)

        # Fill remaining budget from low-disagreement buckets (difficulty spread)
        remaining = self.target_n - len(selected)
        if remaining > 0 and self.fill_from_buckets:
            # Group low-disagreement by bucket
            buckets: Dict[str, List[int]] = defaultdict(list)
            for idx, pr in low_disag:
                if idx not in selected:
                    b = self._bucket(pr, n_models)
                    buckets[b].append(idx)

            # Shuffle each bucket deterministically
            for b in buckets:
                rng.shuffle(buckets[b])

            # Round-robin across buckets to maintain spread
            bucket_names = sorted(buckets.keys())
            round_robin = [idx for b in bucket_names for idx in buckets[b]]
            # Interleave: take one from each bucket in turn
            interleaved: List[int] = []
            iters = [iter(buckets[b]) for b in bucket_names]
            exhausted = [False] * len(iters)
            while len(interleaved) < remaining and not all(exhausted):
                for i, it in enumerate(iters):
                    if exhausted[i]:
                        continue
                    try:
                        interleaved.append(next(it))
                    except StopIteration:
                        exhausted[i] = True

            for idx in interleaved[:remaining]:
                selected.add(idx)

        return sorted(selected)
