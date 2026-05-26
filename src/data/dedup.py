"""Deduplication: exact + near-duplicate detection via k-mer Jaccard sets."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd

from ..core.sequence import get_kmers


def _seq_signature(sequence: str, k: int = 8, sketch_size: int = 64) -> frozenset[str]:
    """Return a small signature set of the smallest k-mer hashes (MinHash-lite)."""
    if len(sequence) < k:
        return frozenset()
    kmers = set(get_kmers(sequence, k))
    if not kmers:
        return frozenset()
    hashed = sorted(
        kmers, key=lambda km: hashlib.md5(km.encode("ascii")).hexdigest()
    )
    return frozenset(hashed[:sketch_size])


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def deduplicate(
    df: pd.DataFrame,
    *,
    sequence_col: str = "sequence",
    label_col: str = "disease",
    k: int = 8,
    sketch_size: int = 64,
    similarity_threshold: float = 0.95,
) -> pd.DataFrame:
    """Drop exact duplicates and group near-duplicates within each label.

    Group-aware: only considers two sequences as duplicates if they share the
    same label. This avoids accidentally merging same-gene fragments labelled
    differently.
    """
    df = df.drop_duplicates(subset=[sequence_col]).reset_index(drop=True)

    keep_indices: list[int] = []
    by_label: dict[str, list[tuple[int, frozenset, int]]] = defaultdict(list)

    for idx, row in df.iterrows():
        sig = _seq_signature(row[sequence_col], k=k, sketch_size=sketch_size)
        is_near_dup = False
        for kept_idx, kept_sig, kept_len in by_label[row[label_col]]:
            if jaccard(sig, kept_sig) >= similarity_threshold:
                # Keep the longer of the two
                if len(row[sequence_col]) > kept_len:
                    by_label[row[label_col]].remove((kept_idx, kept_sig, kept_len))
                    keep_indices.remove(kept_idx)
                    by_label[row[label_col]].append((idx, sig, len(row[sequence_col])))
                    keep_indices.append(idx)
                is_near_dup = True
                break
        if not is_near_dup:
            by_label[row[label_col]].append((idx, sig, len(row[sequence_col])))
            keep_indices.append(idx)

    return df.loc[sorted(keep_indices)].reset_index(drop=True)
