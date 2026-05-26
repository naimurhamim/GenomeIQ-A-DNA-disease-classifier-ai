"""Out-of-distribution detection using cosine similarity to training corpus."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from ..core.config import (
    DISEASE_DATASET_CSV,
    MODELS_DIR,
    OOD_STATS_PATH,
)
from ..core.sequence import kmers_as_text
from .classifier import load_models, trained_kmer_size


_OOD_META_PATH: Path = MODELS_DIR / "ood_stats.meta.json"


@dataclass(slots=True)
class OODResult:
    """Outcome of OOD detection for a sequence."""

    is_in_distribution: bool
    risk: str  # "low" | "medium" | "high"
    novelty_score: float  # 0..1, higher means more novel
    nearest_similarity: float
    threshold_low: float
    threshold_high: float
    message: str

    def to_dict(self) -> dict:
        return {
            "is_in_distribution": self.is_in_distribution,
            "risk": self.risk,
            "novelty_score": round(self.novelty_score, 4),
            "nearest_similarity": round(self.nearest_similarity, 4),
            "threshold_low": round(self.threshold_low, 4),
            "threshold_high": round(self.threshold_high, 4),
            "message": self.message,
        }


def build_ood_stats(*, force: bool = False) -> None:
    """Compute training-set similarity distribution to derive OOD thresholds."""
    if OOD_STATS_PATH.exists() and _OOD_META_PATH.exists() and not force:
        return

    if not DISEASE_DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DISEASE_DATASET_CSV}")

    bundle = load_models()
    df = pd.read_csv(DISEASE_DATASET_CSV)
    texts = [kmers_as_text(seq, trained_kmer_size()) for seq in df["sequence"].astype(str)]
    matrix = bundle.vectorizer.transform(texts)
    matrix = normalize(matrix, norm="l2", axis=1)

    # Pairwise nearest-neighbour similarity (excluding self) — sample if large
    n = matrix.shape[0]
    sample_size = min(n, 400)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sims = (matrix[sample_idx] @ matrix.T).toarray()
    # Mask self-similarity
    for i, idx in enumerate(sample_idx):
        sims[i, idx] = -1.0
    nearest = sims.max(axis=1)

    stats = {
        "nearest_similarity_mean": float(nearest.mean()),
        "nearest_similarity_std": float(nearest.std()),
        "nearest_similarity_p05": float(np.percentile(nearest, 5)),
        "nearest_similarity_p25": float(np.percentile(nearest, 25)),
    }

    OOD_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(OOD_STATS_PATH, matrix)
    _OOD_META_PATH.write_text(json.dumps(stats), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_ood() -> tuple[sparse.csr_matrix, dict]:
    if not OOD_STATS_PATH.exists() or not _OOD_META_PATH.exists():
        build_ood_stats(force=True)
    matrix = sparse.load_npz(OOD_STATS_PATH).tocsr()
    stats = json.loads(_OOD_META_PATH.read_text(encoding="utf-8"))
    return matrix, stats


def detect(sequence: str) -> OODResult:
    """Compute the OOD verdict for a sequence."""
    matrix, stats = _load_ood()
    bundle = load_models()
    q = bundle.vectorizer.transform([kmers_as_text(sequence, trained_kmer_size())])
    q = normalize(q, norm="l2", axis=1)
    sims = (matrix @ q.T).toarray().ravel()
    nearest = float(sims.max()) if sims.size else 0.0

    threshold_low = float(stats["nearest_similarity_p05"])
    threshold_high = float(stats["nearest_similarity_p25"])

    novelty = max(0.0, min(1.0, 1.0 - nearest))
    if nearest >= threshold_high:
        risk, in_dist, msg = (
            "low",
            True,
            "Sequence resembles known training distribution.",
        )
    elif nearest >= threshold_low:
        risk, in_dist, msg = (
            "medium",
            True,
            "Sequence is somewhat novel; treat predictions as suggestive.",
        )
    else:
        risk, in_dist, msg = (
            "high",
            False,
            "Sequence is far from the training distribution. Prediction may be unreliable.",
        )

    return OODResult(
        is_in_distribution=in_dist,
        risk=risk,
        novelty_score=novelty,
        nearest_similarity=nearest,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        message=msg,
    )
