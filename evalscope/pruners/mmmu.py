"""
evalscope.pruners.mmmu
~~~~~~~~~~~~~~~~~~~~~~
MMMU probe-set pruner for image-encoder capability assessment (Part B).

Design goal
-----------
The customer may extend into multimodal next quarter.  We need a cheap probe
that surfaces image-encoder degradation specifically -- not generic VLM
capability gaps -- using only the standard OpenAI-compatible chat API.

What stresses an image encoder?
--------------------------------
1. The answer is unreachable without the image.
2. The image carries dense low-level information: charts, diagrams, equations,
   spatial layouts -- not natural photos.
3. STEM subjects (Math, Physics, Engineering, Electronics, Chemistry) penalize
   encoder errors more than humanities subjects.
4. Multiple images per sample stress cross-reference ability.

Probe selection strategy
------------------------
Rank all 22 MMMU subjects by encoder sensitivity and allocate sample budget
proportionally.  Ensure all 6 MMMU disciplines have coverage.
Target: 200 samples from the full 12K dataset (~1.7%).

MMMU indexing note
------------------
Each MMMU subject file uses its own 0-based index (0-29 for the 660-row
reference set, 0-~550 for the full 12K dataset).  Indices are therefore only
unique within a subject.  This pruner returns globally unique keys as
"SubjectName__local_index" strings via ``selected_keys`` attribute, and
returns sequential integers from ``select()`` for base-class compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Pruner, register_pruner


# Encoder sensitivity scores per MMMU subject (0=text-heavy, 1=image-critical)
_ENCODER_SENSITIVITY: Dict[str, float] = {
    'Math':                                 0.95,
    'Electronics':                          0.93,
    'Energy_and_Power':                     0.90,
    'Architecture_and_Engineering':         0.88,
    'Chemistry':                            0.87,
    'Materials':                            0.85,
    'Computer_Science':                     0.83,
    'Physics':                              0.82,
    'Biology':                              0.78,
    'Geography':                            0.72,
    'Diagnostics_and_Laboratory_Medicine':  0.70,
    'Basic_Medical_Science':                0.68,
    'Clinical_Medicine':                    0.65,
    'Economics':                            0.62,
    'Finance':                              0.60,
    'Accounting':                           0.58,
    'Agriculture':                          0.50,
    'Manage':                               0.45,
    'Marketing':                            0.42,
    'Design':                               0.40,
    'Art_Theory':                           0.35,
    'Art':                                  0.30,
    'History':                              0.28,
    'Literature':                           0.25,
}

_DISCIPLINES: Dict[str, List[str]] = {
    'Art_and_Design':    ['Art', 'Design', 'Art_Theory'],
    'Business':          ['Accounting', 'Economics', 'Finance', 'Manage', 'Marketing'],
    'Health_Medicine':   ['Basic_Medical_Science', 'Clinical_Medicine',
                          'Diagnostics_and_Laboratory_Medicine'],
    'Humanities_Social': ['History', 'Literature', 'Geography'],
    'Science':           ['Biology', 'Chemistry', 'Math', 'Physics'],
    'Tech_Engineering':  ['Agriculture', 'Architecture_and_Engineering',
                          'Computer_Science', 'Electronics',
                          'Energy_and_Power', 'Materials'],
}


@register_pruner(['mmmu_encoder', 'mmmu'])
class MMMUEncoderPruner(Pruner):
    """Select a MMMU probe set that stresses the image encoder specifically.

    Args:
        total_budget:        Target probe size across all subjects. Default 200.
        min_per_discipline:  Minimum samples per MMMU discipline (6 total).
        sensitivity_threshold: Subjects below this score get minimum allocation only.
        seed:                Random seed for deterministic sampling.
    """

    def __init__(
        self,
        total_budget: int = 200,
        min_per_discipline: int = 5,
        sensitivity_threshold: float = 0.60,
        seed: int = 42,
    ) -> None:
        self.total_budget = total_budget
        self.min_per_discipline = min_per_discipline
        self.sensitivity_threshold = sensitivity_threshold
        self.seed = seed
        self._selected_keys: List[str] = []

    def _allocate_budget(self, subjects: List[str]) -> Dict[str, int]:
        """Compute per-subject sample budget proportional to sensitivity."""
        high = {s: _ENCODER_SENSITIVITY.get(s, 0.5)
                for s in subjects
                if _ENCODER_SENSITIVITY.get(s, 0.5) >= self.sensitivity_threshold}
        low = {s for s in subjects if s not in high}

        reserved = len(low) * self.min_per_discipline
        remaining = max(0, self.total_budget - reserved)

        alloc: Dict[str, int] = {}
        for s in low:
            alloc[s] = self.min_per_discipline

        total_weight = sum(high.values()) or 1.0
        items = sorted(high.items(), key=lambda x: x[1], reverse=True)
        distributed = 0
        for i, (s, w) in enumerate(items):
            if i == len(items) - 1:
                alloc[s] = max(self.min_per_discipline, remaining - distributed)
            else:
                n = max(self.min_per_discipline, round(remaining * w / total_weight))
                alloc[s] = n
                distributed += n

        return alloc

    def _infer_subject(self, row: Dict[str, Any], filename_hint: str = '') -> str:
        """Infer subject from row metadata or filename hint."""
        subject = (
            row.get('metadata', {}).get('subset_key')
            or row.get('subset_key')
        )
        if subject:
            return str(subject)
        # Try to infer from filename hint (e.g. "mmmu_Accounting")
        if filename_hint:
            parts = filename_hint.replace('mmmu_', '').replace('.jsonl', '')
            if parts:
                return parts
        return 'Unknown'

    def select(self, reviews: List[Dict[str, Any]]) -> List[int]:
        """Select probe indices from MMMU review rows.

        MMMU uses per-subject local indices (0-based within each subject file).
        This method groups rows by subject, allocates budget, prioritises
        multi-image samples, and stores globally unique keys as
        ``self.selected_keys`` in the format ``"SubjectName__local_index"``.

        Returns sequential integers (0, 1, 2, ...) for base-class compatibility.
        Use ``self.selected_keys`` for the actual subject+index pairs.
        """
        import random
        rng = random.Random(self.seed)

        # Group rows by subject
        subject_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in reviews:
            subject = self._infer_subject(row)
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(row)

        subjects = list(subject_groups.keys())
        alloc = self._allocate_budget(subjects)

        def image_count(row: Dict[str, Any]) -> int:
            meta = row.get('metadata', {})
            ic = meta.get('image_count', 1)
            if ic == 1:
                inp = row.get('input', '')
                if isinstance(inp, str):
                    ic = max(1, inp.count('<image'))
            return ic

        selected_keys: List[str] = []
        for subject, rows in subject_groups.items():
            n = alloc.get(subject, self.min_per_discipline)
            if not rows:
                continue

            multi = [r for r in rows if image_count(r) > 1]
            single = [r for r in rows if image_count(r) <= 1]

            rng.shuffle(multi)
            rng.shuffle(single)

            pool = multi + single
            chosen = pool[:n]

            for r in chosen:
                idx = r.get('index')
                if idx is not None:
                    selected_keys.append(f"{subject}__{idx}")

        # Deduplicate and sort
        self._selected_keys = sorted(set(selected_keys))
        return list(range(len(self._selected_keys)))

    @property
    def selected_keys(self) -> List[str]:
        """Globally unique subject__index keys from the last select() call."""
        return self._selected_keys

    def summary(self, original: int, selected: int) -> str:
        n_subjects = len(set(k.split('__')[0] for k in self._selected_keys))
        return (
            f'{self.__class__.__name__}: {selected} samples selected '
            f'across {n_subjects} subjects '
            f'(from {original} reference rows)'
        )
