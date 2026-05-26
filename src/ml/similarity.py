"""Nearest-known-sequence similarity search (BLAST-lite)."""
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
    SIMILARITY_INDEX_PATH,
)
from ..core.sequence import kmers_as_text
from .classifier import load_models, trained_kmer_size


_INDEX_META_PATH: Path = MODELS_DIR / "similarity_index.meta.json"


@dataclass(slots=True)
class SimilarityHit:
    """A single nearest-neighbour hit."""

    rank: int
    similarity: float
    disease: str
    sequence_id: str
    preview: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "similarity": round(self.similarity, 4),
            "disease": self.disease,
            "sequence_id": self.sequence_id,
            "preview": self.preview,
        }


def build_similarity_index(*, force: bool = False) -> None:
    """Vectorise the training dataset and persist a normalised sparse matrix."""
    if SIMILARITY_INDEX_PATH.exists() and _INDEX_META_PATH.exists() and not force:
        return

    if not DISEASE_DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DISEASE_DATASET_CSV}")

    bundle = load_models()
    df = pd.read_csv(DISEASE_DATASET_CSV)
    texts = [kmers_as_text(seq, trained_kmer_size()) for seq in df["sequence"].astype(str)]
    matrix = bundle.vectorizer.transform(texts)
    matrix = normalize(matrix, norm="l2", axis=1)

    SIMILARITY_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(SIMILARITY_INDEX_PATH, matrix)

    meta = {
        "ids": df["id"].astype(str).tolist(),
        "diseases": df["disease"].astype(str).tolist(),
        "previews": [str(s)[:80] for s in df["sequence"]],
    }
    _INDEX_META_PATH.write_text(json.dumps(meta), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_index() -> tuple[sparse.csr_matrix, dict]:
    if not SIMILARITY_INDEX_PATH.exists() or not _INDEX_META_PATH.exists():
        build_similarity_index(force=True)
    matrix = sparse.load_npz(SIMILARITY_INDEX_PATH).tocsr()
    meta = json.loads(_INDEX_META_PATH.read_text(encoding="utf-8"))
    return matrix, meta


def find_similar(sequence: str, *, top_k: int = 5) -> list[SimilarityHit]:
    """Return the top-K most similar training sequences (cosine similarity)."""
    matrix, meta = _load_index()
    bundle = load_models()
    q = bundle.vectorizer.transform([kmers_as_text(sequence, trained_kmer_size())])
    q = normalize(q, norm="l2", axis=1)

    sims = (matrix @ q.T).toarray().ravel()
    if sims.size == 0:
        return []
    order = np.argsort(sims)[::-1][:top_k]
    hits: list[SimilarityHit] = []
    for rank, idx in enumerate(order, start=1):
        hits.append(
            SimilarityHit(
                rank=rank,
                similarity=float(sims[idx]),
                disease=meta["diseases"][idx],
                sequence_id=meta["ids"][idx],
                preview=meta["previews"][idx],
            )
        )
    return hits
