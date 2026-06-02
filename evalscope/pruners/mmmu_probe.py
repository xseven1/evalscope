"""
evalscope.pruners.mmmu_probe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Working implementation of the MMMU image-encoder ablation probe.

This module provides ``MMUAblationProbe`` -- a runnable probe that measures
image-encoder contribution by submitting each selected MMMU question twice:

  1. **Real image**: the original question with its actual image(s).
  2. **Blank image**: the same question with a 1×1 white JPEG replacing all images.

The accuracy *drop* (real − blank) isolates the encoder's contribution.  A
functioning encoder shows a large drop; a degraded encoder shows almost none
because the model was already guessing from text.

Requirements
------------
- ``datasets`` (HuggingFace): ``pip install datasets``
- ``openai``: ``pip install openai``
- ``Pillow``: ``pip install Pillow``
- A HuggingFace token with access to ``MMMU/MMMU``
- An OpenAI-compatible API key / base URL

Usage
-----
Run the probe from the command line::

    python -m evalscope.pruners.mmmu_probe \\
        --indices-file selected_mmmu.json \\
        --hf-token YOUR_HF_TOKEN \\
        --api-key YOUR_API_KEY \\
        --model glm-4.5v-fp8 \\
        --out mmmu_probe_results.json

Or programmatically::

    from evalscope.pruners.mmmu_probe import MMUAblationProbe

    probe = MMUAblationProbe(
        api_key="sk-...",
        model="glm-4.5v-fp8",
        hf_token="hf_...",
    )
    results = probe.run(selected_indices=["Accounting__2", "Math__15", ...])
    print(f"Encoder delta: {results['mean_accuracy_delta']:.3f}")
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
# Constants
BLANK_JPEG_B64: str = (
    # 1×1 white pixel JPEG, base64-encoded
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U'
    'HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN'
    'DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy'
    'MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAA'
    'AAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/'
    'aAAwDAQACEQMRAD8AJQAB/9k='
)

MMMU_HF_DATASET = 'MMMU/MMMU'

MCQ_PROMPT = (
    'Answer the following multiple choice question. '
    'The last line of your response should be of the following format: '
    "\"ANSWER: $LETTER\" (without quotes) where LETTER is one of A,B,C,D. "
    'Think step by step before answering.\n\n{question}'
)
# Utilities

def _blank_image_url() -> Dict[str, Any]:
    """Return an OpenAI image_url content block with a 1×1 white JPEG."""
    return {
        'type': 'image_url',
        'image_url': {
            'url': f'data:image/jpeg;base64,{BLANK_JPEG_B64}',
            'detail': 'low',
        },
    }


def _pil_to_b64(pil_image: Any, fmt: str = 'JPEG') -> str:
    """Convert a PIL Image to base64-encoded bytes."""
    buf = io.BytesIO()
    if pil_image.mode in ('RGBA', 'P'):
        pil_image = pil_image.convert('RGB')
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _image_content_block(pil_image: Any) -> Dict[str, Any]:
    """Return an OpenAI image_url content block from a PIL Image."""
    b64 = _pil_to_b64(pil_image)
    return {
        'type': 'image_url',
        'image_url': {
            'url': f'data:image/jpeg;base64,{b64}',
            'detail': 'high',
        },
    }


def _extract_answer(text: str) -> Optional[str]:
    """Extract the MCQ answer letter from model output."""
    import re
    match = re.search(r'ANSWER:\s*([A-D])', text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _is_correct(prediction: Optional[str], target: str) -> float:
    """Return 1.0 if prediction matches target, else 0.0."""
    if prediction is None:
        return 0.0
    return 1.0 if prediction.strip().upper() == target.strip().upper() else 0.0

# Main probe class
class MMUAblationProbe:
    """Run the image-ablation probe on selected MMMU samples.

    For each selected sample:
    - Submits the question with the real image(s) → records accuracy
    - Submits the same question with blank 1×1 images → records accuracy
    - Computes delta = real_acc - blank_acc

    A large positive delta indicates the encoder is contributing meaningfully.
    A delta near zero indicates the model was guessing from text alone
    (encoder degradation or text-leakage).

    Args:
        api_key:      OpenAI-compatible API key.
        model:        Model identifier (e.g. ``'glm-4.5v-fp8'``).
        api_base:     Base URL for the API (default: OpenAI).
        hf_token:     HuggingFace token for MMMU dataset access.
        max_retries:  Retries on API error (default: 3).
        retry_delay:  Seconds between retries (default: 2).
        verbose:      Print progress to stderr.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        hf_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError('openai package required: pip install openai')

        self.model = model
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.hf_token = hf_token

        kwargs: Dict[str, Any] = {'api_key': api_key}
        if api_base:
            kwargs['base_url'] = api_base
        self.client = OpenAI(**kwargs)

    # ------------------------------------------------------------------
    # HuggingFace dataset loading
    # ------------------------------------------------------------------

    def _load_hf_sample(self, subject: str, local_idx: int) -> Optional[Dict[str, Any]]:
        """Load a single MMMU sample from HuggingFace by subject and local index."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError('datasets package required: pip install datasets')

        hf_kwargs: Dict[str, Any] = {'path': MMMU_HF_DATASET, 'name': subject, 'split': 'validation'}
        if self.hf_token:
            hf_kwargs['token'] = self.hf_token

        try:
            ds = load_dataset(**hf_kwargs)
            if local_idx >= len(ds):
                if self.verbose:
                    print(f'  Warning: index {local_idx} out of range for {subject} (len={len(ds)})', file=sys.stderr)
                return None
            return dict(ds[local_idx])
        except Exception as e:
            if self.verbose:
                print(f'  Warning: could not load {subject}[{local_idx}]: {e}', file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _call(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Call the chat API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ''
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    if self.verbose:
                        print(f'  API error after {self.max_retries} attempts: {e}', file=sys.stderr)
                    return None
        return None

    def _build_messages(
        self,
        question: str,
        image_blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build OpenAI messages with interleaved image blocks."""
        content: List[Dict[str, Any]] = []
        # Add images first, then text (standard VLM convention)
        content.extend(image_blocks)
        content.append({'type': 'text', 'text': MCQ_PROMPT.format(question=question)})
        return [{'role': 'user', 'content': content}]

    # ------------------------------------------------------------------
    # Per-sample probe
    # ------------------------------------------------------------------

    def _probe_sample(
        self, subject: str, local_idx: int
    ) -> Optional[Dict[str, Any]]:
        """Run ablation probe on one sample. Returns result dict or None on failure."""
        record = self._load_hf_sample(subject, local_idx)
        if record is None:
            return None

        question = record.get('question', '')
        target = record.get('answer', '')
        options = record.get('options', [])

        # Format options into question
        if options:
            opts_str = '\n'.join(f'{chr(65+i)}. {o}' for i, o in enumerate(options))
            full_question = f'{question}\n\n{opts_str}'
        else:
            full_question = question

        # Collect real images (image_1 ... image_7 keys)
        real_image_blocks: List[Dict[str, Any]] = []
        for i in range(1, 8):
            img = record.get(f'image_{i}')
            if img is not None:
                try:
                    real_image_blocks.append(_image_content_block(img))
                except Exception:
                    pass

        if not real_image_blocks:
            # No images -- skip, can't measure encoder contribution
            return None

        n_images = len(real_image_blocks)
        blank_image_blocks = [_blank_image_url() for _ in range(n_images)]

        # Real image call
        real_messages  = self._build_messages(full_question, real_image_blocks)
        real_response  = self._call(real_messages)
        real_pred      = _extract_answer(real_response or '')
        real_correct   = _is_correct(real_pred, target)

        # Blank image call
        blank_messages = self._build_messages(full_question, blank_image_blocks)
        blank_response = self._call(blank_messages)
        blank_pred     = _extract_answer(blank_response or '')
        blank_correct  = _is_correct(blank_pred, target)

        delta = real_correct - blank_correct

        return {
            'subject':        subject,
            'local_index':    local_idx,
            'target':         target,
            'real_pred':      real_pred,
            'blank_pred':     blank_pred,
            'real_correct':   real_correct,
            'blank_correct':  blank_correct,
            'delta':          delta,
            'n_images':       n_images,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        selected_indices: List[str],
        out_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the ablation probe on all selected indices.

        Args:
            selected_indices: List of ``"Subject__local_index"`` strings
                              (as produced by ``MMMUEncoderPruner``).
            out_file:         Optional path to write JSON results.

        Returns:
            Summary dict with per-sample results and aggregate metrics.
        """
        results = []
        skipped = 0

        for key in selected_indices:
            if '__' not in key:
                skipped += 1
                continue
            parts = key.split('__', 1)
            subject   = parts[0]
            try:
                local_idx = int(parts[1])
            except ValueError:
                skipped += 1
                continue

            if self.verbose:
                print(f'Probing {key} ...', file=sys.stderr)

            result = self._probe_sample(subject, local_idx)
            if result is None:
                skipped += 1
                continue
            results.append(result)

            if self.verbose:
                print(
                    f'  real={result["real_correct"]:.0f}  '
                    f'blank={result["blank_correct"]:.0f}  '
                    f'delta={result["delta"]:+.0f}',
                    file=sys.stderr,
                )

        # Aggregate
        n = len(results)
        if n == 0:
            summary = {
                'n_probed': 0,
                'n_skipped': skipped,
                'mean_real_accuracy': None,
                'mean_blank_accuracy': None,
                'mean_accuracy_delta': None,
                'encoder_signal': 'insufficient_data',
                'samples': [],
            }
        else:
            mean_real  = sum(r['real_correct']  for r in results) / n
            mean_blank = sum(r['blank_correct'] for r in results) / n
            mean_delta = mean_real - mean_blank

            # Encoder signal interpretation
            if mean_delta >= 0.2:
                signal = 'strong'       # encoder clearly contributing
            elif mean_delta >= 0.05:
                signal = 'moderate'
            else:
                signal = 'weak'         # model guessing from text; encoder may be degraded

            summary = {
                'n_probed':            n,
                'n_skipped':           skipped,
                'mean_real_accuracy':  round(mean_real,  4),
                'mean_blank_accuracy': round(mean_blank, 4),
                'mean_accuracy_delta': round(mean_delta, 4),
                'encoder_signal':      signal,
                'samples':             results,
            }

        if out_file:
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            if self.verbose:
                print(f'Results written to: {out_file}', file=sys.stderr)

        return summary

# CLI
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog='python -m evalscope.pruners.mmmu_probe',
        description='Run the MMMU image-encoder ablation probe.',
    )
    parser.add_argument('--indices-file', required=True,
                        help='JSON file from mmmu pruner (contains selected_indices).')
    parser.add_argument('--api-key', required=True,
                        help='OpenAI-compatible API key.')
    parser.add_argument('--model', required=True,
                        help='Model name to probe (e.g. glm-4.5v-fp8).')
    parser.add_argument('--api-base', default=None,
                        help='API base URL (default: OpenAI).')
    parser.add_argument('--hf-token', default=None,
                        help='HuggingFace token for MMMU dataset access.')
    parser.add_argument('--out', default=None,
                        help='Output JSON file path.')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args(argv)

    # Load indices
    with open(args.indices_file, encoding='utf-8') as f:
        data = json.load(f)
    selected = data.get('selected_indices', [])

    if not selected:
        print('No selected_indices found in indices file.', file=sys.stderr)
        sys.exit(1)

    probe = MMUAblationProbe(
        api_key=args.api_key,
        model=args.model,
        api_base=args.api_base,
        hf_token=args.hf_token,
        verbose=args.verbose,
    )

    summary = probe.run(selected_indices=selected, out_file=args.out)

    if not args.out:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Encoder signal: {summary['encoder_signal']} "
              f"(delta={summary['mean_accuracy_delta']})")


if __name__ == '__main__':
    main()
