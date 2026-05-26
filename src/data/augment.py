"""Sequence augmentation utilities: sliding windows + reverse complement."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from ..core.sequence import reverse_complement


@dataclass(slots=True)
class AugmentationConfig:
    """Configuration for sliding-window data augmentation."""

    window_sizes: tuple[int, ...] = (300, 600, 1200)
    stride_ratio: float = 0.5  # stride = window_size * stride_ratio
    keep_full_sequence: bool = True
    add_reverse_complement: bool = True
    min_length: int = 150
    max_chunks_per_sequence: int = 60


def _hash(s: str) -> str:
    return hashlib.md5(s.encode("ascii", errors="ignore")).hexdigest()


def sliding_chunks(
    sequence: str, *, window: int, stride: int, max_chunks: int
) -> list[str]:
    """Slide a fixed window across the sequence with configurable stride."""
    if len(sequence) <= window:
        return [sequence]
    out: list[str] = []
    for start in range(0, len(sequence) - window + 1, stride):
        out.append(sequence[start : start + window])
        if len(out) >= max_chunks:
            break
    last_start = len(sequence) - window
    if (last_start - (last_start % stride)) != last_start and len(out) < max_chunks:
        out.append(sequence[last_start : last_start + window])
    return out


def expand_sequence(sequence: str, cfg: AugmentationConfig) -> list[str]:
    """Expand a single sequence into multiple training samples."""
    seq = sequence.upper()
    if len(seq) < cfg.min_length:
        return []

    samples: list[str] = []
    if cfg.keep_full_sequence:
        samples.append(seq)

    for window in cfg.window_sizes:
        if len(seq) <= window:
            continue
        stride = max(1, int(window * cfg.stride_ratio))
        samples.extend(
            sliding_chunks(
                seq, window=window, stride=stride, max_chunks=cfg.max_chunks_per_sequence
            )
        )

    if cfg.add_reverse_complement:
        rc_samples = [reverse_complement(s) for s in list(samples)]
        samples.extend(rc_samples)

    # Deduplicate by hash to prevent identical fragments
    seen: set[str] = set()
    unique: list[str] = []
    for s in samples:
        if len(s) < cfg.min_length:
            continue
        h = _hash(s)
        if h in seen:
            continue
        seen.add(h)
        unique.append(s)
    return unique


def expand_records(
    records: Iterable[tuple[str, str]],
    cfg: AugmentationConfig,
) -> list[tuple[str, str, str]]:
    """Expand (label, sequence) pairs into (label, parent_id, fragment) triples."""
    out: list[tuple[str, str, str]] = []
    for parent_id, (label, sequence) in enumerate(records):
        for fragment in expand_sequence(sequence, cfg):
            out.append((label, f"p{parent_id}", fragment))
    return out
