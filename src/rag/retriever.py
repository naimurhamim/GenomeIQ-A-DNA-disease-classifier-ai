"""FAISS-based retriever over the curated knowledge base."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.config import MODELS_DIR
from .knowledge_base import Fact, all_facts


_INDEX_PATH = MODELS_DIR / "rag_index.faiss"
_META_PATH = MODELS_DIR / "rag_index_meta.npz"
_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Lazy globals
_EMBEDDER = None
_FAISS = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME)
    return _EMBEDDER


def _faiss():
    global _FAISS
    if _FAISS is None:
        import faiss  # type: ignore

        _FAISS = faiss
    return _FAISS


@dataclass(slots=True)
class RetrievedFact:
    fact: Fact
    score: float

    def to_dict(self) -> dict:
        return {
            "fact_id": self.fact.fact_id,
            "topic": self.fact.topic,
            "category": self.fact.category,
            "title": self.fact.title,
            "content": self.fact.content,
            "source": self.fact.source,
            "score": round(self.score, 4),
        }


def _build_index() -> tuple[object, list[Fact]]:
    facts = all_facts()
    embedder = _embedder()
    texts = [f"{f.title}. {f.content}" for f in facts]
    vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    faiss = _faiss()
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_PATH))
    np.savez(
        _META_PATH,
        ids=np.array([f.fact_id for f in facts]),
    )
    return index, facts


@lru_cache(maxsize=1)
def _index_and_facts() -> tuple[object, list[Fact]]:
    facts = all_facts()
    if _INDEX_PATH.exists():
        try:
            index = _faiss().read_index(str(_INDEX_PATH))
            return index, facts
        except Exception:
            pass
    return _build_index()


def warmup() -> None:
    """Pre-build the index at startup."""
    _index_and_facts()


def search(query: str, *, top_k: int = 5, min_score: float = 0.15) -> list[RetrievedFact]:
    """Return the top-k retrieved facts for a query."""
    if not query.strip():
        return []
    index, facts = _index_and_facts()
    embedder = _embedder()
    q = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    scores, indices = index.search(q, top_k)
    out: list[RetrievedFact] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(facts):
            continue
        if float(score) < min_score:
            continue
        out.append(RetrievedFact(fact=facts[idx], score=float(score)))
    return out
