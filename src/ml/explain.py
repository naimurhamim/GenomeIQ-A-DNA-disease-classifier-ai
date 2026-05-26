"""Saliency-style explanation: per-position contribution to the prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.sequence import get_kmers
from .classifier import load_models, trained_kmer_size


@dataclass(slots=True)
class SaliencyResult:
    """Per-base saliency scores for a sequence."""

    sequence: str
    scores: list[float]  # one entry per base, in [0, 1]
    top_regions: list[dict]
    kmer_size: int

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "scores": [round(s, 4) for s in self.scores],
            "top_regions": self.top_regions,
            "kmer_size": self.kmer_size,
        }


def _class_index(label: str) -> int:
    bundle = load_models()
    classes = list(bundle.label_encoder.classes_)
    return classes.index(label)


def _feature_importance_for_class(target_class_idx: int) -> dict[str, float]:
    """Return TF-IDF token -> importance for the target class.

    Uses the RandomForest's feature_importances_ as a proxy weighted by
    how strongly the class contains those features. Works whether the
    classifier is a raw VotingClassifier or wrapped in CalibratedClassifierCV.
    """
    bundle = load_models()
    feature_names = bundle.vectorizer.get_feature_names_out()

    rf = _find_random_forest(bundle.classifier)

    if rf is None:
        # fall back to uniform importance
        return {name: 1.0 for name in feature_names}

    importances = rf.feature_importances_
    return {name: float(importances[i]) for i, name in enumerate(feature_names)}


def _find_random_forest(model):
    """Locate a RandomForestClassifier inside an arbitrary estimator wrapping."""
    if hasattr(model, "feature_importances_"):
        return model

    # CalibratedClassifierCV(prefit) -> .estimator
    inner = getattr(model, "estimator", None)
    if inner is not None and inner is not model:
        found = _find_random_forest(inner)
        if found is not None:
            return found

    # CalibratedClassifierCV (cv) -> .calibrated_classifiers_[i].estimator
    for cc in getattr(model, "calibrated_classifiers_", []) or []:
        inner = getattr(cc, "estimator", None)
        if inner is not None:
            found = _find_random_forest(inner)
            if found is not None:
                return found

    # VotingClassifier / Pipeline-like
    named = getattr(model, "named_estimators_", None)
    if named:
        for est in named.values():
            found = _find_random_forest(est)
            if found is not None:
                return found

    return None


def saliency(
    sequence: str,
    *,
    target_class: str | None = None,
    k: int | None = None,
) -> SaliencyResult:
    """Compute per-base saliency.

    Each base receives a score that aggregates the contribution of every
    overlapping vocabulary k-mer-token (n-gram of single k-mers) it appears in.
    """
    if k is None:
        k = trained_kmer_size()
    bundle = load_models()
    feature_names = bundle.vectorizer.get_feature_names_out()
    feature_set = set(feature_names)

    seq_lower = sequence.lower()
    base_scores = np.zeros(len(seq_lower), dtype=np.float64)

    if target_class is None:
        # Use predicted class
        from .classifier import predict_sequence

        target_class = predict_sequence(sequence).predicted_class

    target_idx = _class_index(target_class)
    importance = _feature_importance_for_class(target_idx)

    # Sliding window must match vectorizer ngram_range.
    ngram_low, ngram_high = bundle.vectorizer.ngram_range
    kmers = get_kmers(seq_lower, k)
    if not kmers:
        return SaliencyResult(
            sequence=sequence, scores=[], top_regions=[], kmer_size=k
        )

    # For each n-gram window of k-mers, look it up in vocabulary.
    for n in range(ngram_low, ngram_high + 1):
        for i in range(len(kmers) - n + 1):
            token = " ".join(kmers[i : i + n])
            if token not in feature_set:
                continue
            weight = importance.get(token, 0.0)
            if weight <= 0:
                continue
            start = i
            end = i + n + k - 1  # span covered by these n consecutive k-mers
            base_scores[start:end] += weight

    if base_scores.max() > 0:
        base_scores = base_scores / base_scores.max()

    # Find top regions by sliding-window mean.
    top_regions = _top_regions(base_scores.tolist(), window=max(k * 4, 24))

    return SaliencyResult(
        sequence=sequence,
        scores=base_scores.tolist(),
        top_regions=top_regions,
        kmer_size=k,
    )


def _top_regions(scores: list[float], *, window: int = 30, top_n: int = 5) -> list[dict]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    if len(arr) <= window:
        return [
            {
                "start": 0,
                "end": len(arr),
                "mean_score": round(float(arr.mean()), 4),
            }
        ]
    cumulative = np.cumsum(np.insert(arr, 0, 0.0))
    means = (cumulative[window:] - cumulative[:-window]) / window

    regions: list[dict] = []
    used = np.zeros(len(means), dtype=bool)
    order = np.argsort(means)[::-1]
    for idx in order:
        if used[idx]:
            continue
        start = int(idx)
        end = min(int(idx + window), len(arr))
        regions.append(
            {"start": start, "end": end, "mean_score": round(float(means[idx]), 4)}
        )
        # Suppress overlapping windows
        lo = max(0, start - window // 2)
        hi = min(len(used), start + window // 2)
        used[lo:hi] = True
        if len(regions) >= top_n:
            break
    return regions
