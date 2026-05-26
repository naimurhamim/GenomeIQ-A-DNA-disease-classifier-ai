"""Central configuration paths and constants."""
from __future__ import annotations

from pathlib import Path

# --- Project paths ----------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# --- Model artifacts --------------------------------------------------------
DISEASE_CLASSIFIER_PATH: Path = MODELS_DIR / "disease_classifier.pkl"
DISEASE_VECTORIZER_PATH: Path = MODELS_DIR / "disease_vectorizer.pkl"
DISEASE_LABEL_ENCODER_PATH: Path = MODELS_DIR / "disease_label_encoder.pkl"

# Reference embedding index (built lazily, cached on disk)
SIMILARITY_INDEX_PATH: Path = MODELS_DIR / "similarity_index.npz"
OOD_STATS_PATH: Path = MODELS_DIR / "ood_stats.npz"

# DNABERT-2 transformer artifacts
DNABERT2_MODEL_DIR: Path = MODELS_DIR / "dnabert2_base"
DNABERT2_HEAD_PATH: Path = MODELS_DIR / "dnabert2_head.pkl"
DNABERT2_EMBEDDINGS_PATH: Path = MODELS_DIR / "dnabert2_embeddings.npz"
DNABERT2_LABEL_ENCODER_PATH: Path = MODELS_DIR / "dnabert2_label_encoder.pkl"

# --- Source dataset ---------------------------------------------------------
DISEASE_DATASET_CSV: Path = RAW_DIR / "disease_sequences" / "disease_dna_dataset.csv"

# --- Sequence processing constants ----------------------------------------
KMER_SIZE: int = 6
NGRAM_RANGE: tuple[int, int] = (4, 4)
MIN_SEQUENCE_LENGTH: int = 20
MAX_SEQUENCE_LENGTH: int = 200_000
VALID_NUCLEOTIDES: frozenset[str] = frozenset("ATGC")
EXTENDED_NUCLEOTIDES: frozenset[str] = frozenset("ATGCN")

# --- API metadata -----------------------------------------------------------
API_TITLE: str = "DNA Disease Classifier API"
API_VERSION: str = "0.2.0"
API_DESCRIPTION: str = (
    "AI-powered DNA disease classifier with explainability, "
    "out-of-distribution detection, similarity search, and mutation analysis."
)
