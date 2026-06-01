"""
evalscope.pruners.cli
~~~~~~~~~~~~~~~~~~~~~
Command-line interface for running benchmark pruners.

Usage examples
--------------
Prune LiveCodeBench (315 → ~75 samples)::

    python -m evalscope.pruners.cli \\
        --benchmark lcb \\
        --reviews-dir "Evals/Part 1/reviews" \\
        --target 75 \\
        --out selected_lcb.json

Prune AA-LCR (100 → ~30 samples, with judge noise guard)::

    python -m evalscope.pruners.cli \\
        --benchmark aa_lcr \\
        --reviews-dir "Evals/Part 1/reviews" \\
        --target 30 \\
        --judge-noise-guard \\
        --out selected_aa_lcr.json

Prune MMMU (660 reference rows → ~60 encoder-stress samples)::

    python -m evalscope.pruners.cli \\
        --benchmark mmmu \\
        --reviews-dir "Evals/MMMU/reviews" \\
        --target 60 \\
        --out selected_mmmu.json

Output format
-------------
A JSON file with::

    {
        "benchmark": "lcb",
        "pruner": "disagreement",
        "original_n": 315,
        "selected_n": 74,
        "retention_pct": 23.5,
        "selected_indices": [0, 3, 7, ...]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.pruners import build_pruner

# Benchmark configs
BENCHMARK_CONFIGS = {
    'lcb': {
        'pruner':       'disagreement',
        'score_key':    'pass',
        'file_pattern': 'live_code_bench_v5',
        'default_target': 75,
        'judge_noise_guard': False,
    },
    'aa_lcr': {
        'pruner':       'disagreement',
        'score_key':    'acc',
        'file_pattern': 'aa_lcr',
        'default_target': 30,
        'judge_noise_guard': True,   # LLM judge -- enable noise guard by default
    },
    'mmmu': {
        'pruner':       'mmmu_encoder',
        'score_key':    'acc',
        'file_pattern': '',           # All files in reviews dir
        'default_target': 60,
        'judge_noise_guard': False,
    },
}

# Helpers
def _load_reviews(reviews_dir: str, file_pattern: str) -> List[Dict[str, Any]]:
    base = Path(reviews_dir)
    if not base.exists():
        raise FileNotFoundError(f'Reviews directory not found: {reviews_dir}')

    rows: List[Dict[str, Any]] = []
    for jsonl_file in sorted(base.rglob('*.jsonl')):
        if file_pattern and file_pattern not in jsonl_file.name:
            continue
        stem = jsonl_file.stem
        subject_hint = stem.replace('mmmu_', '') if stem.startswith('mmmu_') else ''
        with open(jsonl_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        if subject_hint and not row.get('subset_key'):
                            if 'metadata' not in row or row['metadata'] is None:
                                row['metadata'] = {}
                            if not row['metadata'].get('subset_key'):
                                row['subset_key'] = subject_hint
                        rows.append(row)
                    except json.JSONDecodeError:
                        pass

    if not rows:
        raise ValueError(
            f'No review rows found in {reviews_dir!r} '
            f'(pattern={file_pattern!r}). '
            f'Check the path and file names.'
        )
    return rows


def _run_pruner(
    benchmark: str,
    reviews_dir: str,
    target: Optional[int],
    judge_noise_guard: bool,
    out: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    cfg = BENCHMARK_CONFIGS[benchmark]
    pruner_name = cfg['pruner']
    score_key   = cfg['score_key']
    file_pattern = cfg['file_pattern']
    default_target = cfg['default_target']
    noise_guard = judge_noise_guard or cfg['judge_noise_guard']

    target_n = target if target is not None else default_target

    # Load review rows
    if verbose:
        print(f'Loading reviews from: {reviews_dir}', file=sys.stderr)
    reviews = _load_reviews(reviews_dir, file_pattern)
    if verbose:
        print(f'Loaded {len(reviews)} review rows', file=sys.stderr)

    # Build and run pruner
    kwargs: Dict[str, Any] = {'target_n': target_n}
    if pruner_name == 'disagreement':
        kwargs['score_key'] = score_key
        kwargs['judge_noise_guard'] = noise_guard
    elif pruner_name == 'mmmu_encoder':
        kwargs = {'total_budget': target_n}

    pruner = build_pruner(pruner_name, **kwargs)
    selected = pruner.select(reviews)

    # MMMU has per-subject local indices -- use subject__index keys
    from evalscope.pruners.mmmu import MMMUEncoderPruner
    is_mmmu = isinstance(pruner, MMMUEncoderPruner)

    if is_mmmu:
        unique_pairs = set()
        for r in reviews:
            idx = r.get('index')
            subj = (r.get('metadata', {}) or {}).get('subset_key') or r.get('subset_key') or 'Unknown'
            if idx is not None:
                unique_pairs.add(f"{subj}__{idx}")
        original_n = len(unique_pairs)
        selected_n = len(pruner.selected_keys)
        retention_pct = round(100.0 * selected_n / original_n, 1) if original_n else 0.0
        selected_output = pruner.selected_keys
    else:
        unique_indices = set(r['index'] for r in reviews if 'index' in r)
        original_n = len(unique_indices)
        selected_n = len(selected)
        retention_pct = round(100.0 * selected_n / original_n, 1) if original_n else 0.0
        selected_output = selected

    result = {
        'benchmark':        benchmark,
        'pruner':           pruner_name,
        'reviews_dir':      str(reviews_dir),
        'original_n':       original_n,
        'selected_n':       selected_n,
        'retention_pct':    retention_pct,
        'selected_indices': selected_output,
    }

    if verbose:
        print(pruner.summary(original_n, selected_n), file=sys.stderr)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f'Saved to: {out}', file=sys.stderr)
    else:
        print(json.dumps(result, indent=2))

    return result

# CLI entrypoint
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m evalscope.pruners.cli',
        description='Prune benchmark review files to the most discriminative sample subset.',
    )
    parser.add_argument(
        '--benchmark', required=True,
        choices=list(BENCHMARK_CONFIGS.keys()),
        help='Benchmark to prune: lcb, aa_lcr, or mmmu.',
    )
    parser.add_argument(
        '--reviews-dir', required=True,
        help='Path to the reviews directory (contains .jsonl files).',
    )
    parser.add_argument(
        '--target', type=int, default=None,
        help='Target number of samples to retain.',
    )
    parser.add_argument(
        '--judge-noise-guard', action='store_true',
        help='Only select samples with a clear binary split across models '
             '(recommended for LLM-judged benchmarks like AA-LCR).',
    )
    parser.add_argument(
        '--out', default=None,
        help='Output JSON file path. If omitted, prints to stdout.',
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print progress to stderr.',
    )

    args = parser.parse_args(argv)
    _run_pruner(
        benchmark=args.benchmark,
        reviews_dir=args.reviews_dir,
        target=args.target,
        judge_noise_guard=args.judge_noise_guard,
        out=args.out,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
