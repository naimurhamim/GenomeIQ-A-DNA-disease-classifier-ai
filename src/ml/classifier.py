"""Wrapper around the trained TF-IDF + voting-classifier pipeline."""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..core.config import (
    DISEASE_CLASSIFIER_PATH,
    DISEASE_LABEL_ENCODER_PATH,
    DISEASE_VECTORIZER_PATH,
    KMER_SIZE,
    MODELS_DIR,
)
from ..core.sequence import kmers_as_text


_TRAIN_META_PATH: Path = MODELS_DIR / "train_meta.json"


def _train_meta() -> dict:
    if _TRAIN_META_PATH.exists():
        try:
            return json.loads(_TRAIN_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def trained_kmer_size() -> int:
    return int(_train_meta().get("kmer_size", KMER_SIZE))


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


@dataclass(slots=True)
class ModelBundle:
    """Container for the trained pipeline."""

    classifier: object
    vectorizer: object
    label_encoder: object

    @property
    def classes(self) -> list[str]:
        return list(self.label_encoder.classes_)


@lru_cache(maxsize=1)
def load_models() -> ModelBundle:
    """Lazily load (and cache) the trained model artifacts."""
    return ModelBundle(
        classifier=_load_pickle(DISEASE_CLASSIFIER_PATH),
        vectorizer=_load_pickle(DISEASE_VECTORIZER_PATH),
        label_encoder=_load_pickle(DISEASE_LABEL_ENCODER_PATH),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PredictionResult:
    """Output of a single sequence prediction."""

    predicted_class: str
    confidence: float
    entropy: float
    probabilities: dict[str, float]
    top_kmers: list[dict]  # [{"kmer": str, "score": float}]

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "top_kmers": self.top_kmers,
        }


def _softmax_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector (in nats normalised to [0,1])."""
    p = np.clip(probs, 1e-12, 1.0)
    h = float(-(p * np.log(p)).sum())
    h_max = math.log(len(probs)) if len(probs) > 1 else 1.0
    return h / h_max if h_max > 0 else 0.0


def vectorize(sequence: str) -> np.ndarray:
    """Vectorise a cleaned DNA sequence using the trained vectorizer."""
    bundle = load_models()
    return bundle.vectorizer.transform([kmers_as_text(sequence, trained_kmer_size())])


def predict_sequence(
    sequence: str,
    *,
    top_k_kmers: int = 10,
) -> PredictionResult:
    """Run the trained pipeline and return a structured prediction result."""
    bundle = load_models()
    X = bundle.vectorizer.transform([kmers_as_text(sequence, trained_kmer_size())])
    probs = bundle.classifier.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    classes = bundle.classes
    predicted_class = bundle.label_encoder.inverse_transform([pred_idx])[0]

    probabilities = {cls: float(probs[i]) for i, cls in enumerate(classes)}
    confidence = float(probs[pred_idx])
    entropy = _softmax_entropy(probs)

    top_kmers = _top_contributing_tokens(X, k=top_k_kmers)

    return PredictionResult(
        predicted_class=str(predicted_class),
        confidence=confidence,
        entropy=entropy,
        probabilities=probabilities,
        top_kmers=top_kmers,
    )


def _top_contributing_tokens(X, *, k: int = 10) -> list[dict]:
    """Return the k tokens with the highest TF-IDF weight in the sequence."""
    bundle = load_models()
    feature_names = bundle.vectorizer.get_feature_names_out()
    row = X.toarray().ravel()
    if row.size == 0:
        return []
    top_idx = np.argsort(row)[::-1][:k]
    return [
        {"kmer": str(feature_names[i]), "score": float(row[i])}
        for i in top_idx
        if row[i] > 0
    ]
