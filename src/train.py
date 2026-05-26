"""Train the disease classifier on the augmented, deduplicated dataset.

Run:
    python -m src.data.build_dataset
    python -m src.train

The training script consumes processed/train.parquet and processed/val.parquet
produced by build_dataset.py. The ``train_on_dataframes`` helper is also reused
by ``src.eval`` for cross-validation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from .core.config import (
    DISEASE_CLASSIFIER_PATH,
    DISEASE_LABEL_ENCODER_PATH,
    DISEASE_VECTORIZER_PATH,
    KMER_SIZE,
    MODELS_DIR,
    NGRAM_RANGE,
    PROCESSED_DIR,
)
from .core.sequence import kmers_as_text


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data.build_dataset` first."
        )
    return pd.read_parquet(path)


def build_vectorizer(
    *, ngram_range: tuple[int, int] = NGRAM_RANGE
) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )


def build_classifier(*, n_jobs: int = -1) -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=n_jobs,
        random_state=42,
        class_weight="balanced",
    )
    svm = CalibratedClassifierCV(
        LinearSVC(C=1.0, class_weight="balanced", max_iter=4000),
        method="sigmoid",
        cv=3,
    )
    lr = LogisticRegression(
        C=1.0,
        max_iter=4000,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=42,
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lr", lr)],
        voting="soft",
        n_jobs=n_jobs,
        weights=[2, 2, 1],
    )


def train_on_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    k: int = KMER_SIZE,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    verbose: bool = True,
) -> dict:
    """Train + evaluate on already-prepared train/val dataframes.

    Each dataframe must contain ``sequence`` and ``disease`` columns.

    Returns a metrics dictionary plus the fitted artifacts under
    ``"_artifacts"`` (vectorizer, classifier, label encoder).
    """
    t0 = time.time()
    X_train_raw = [kmers_as_text(s, k) for s in train_df["sequence"]]
    X_val_raw = [kmers_as_text(s, k) for s in val_df["sequence"]]

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["disease"])
    y_val = le.transform(val_df["disease"])

    vectorizer = build_vectorizer(ngram_range=ngram_range)
    if verbose:
        print(f"[fit] vectorizing {len(X_train_raw):,} samples (k={k}, ngrams={ngram_range})...")
    Xt_train = vectorizer.fit_transform(X_train_raw)
    Xt_val = vectorizer.transform(X_val_raw)
    if verbose:
        print(f"[fit] vocabulary size: {len(vectorizer.vocabulary_):,}")

    classifier = build_classifier()
    if verbose:
        print("[fit] training ensemble (RF + SVM + LR)...")
    classifier.fit(Xt_train, y_train)

    val_preds = classifier.predict(Xt_val)
    macro_f1 = f1_score(y_val, val_preds, average="macro")
    weighted_f1 = f1_score(y_val, val_preds, average="weighted")
    accuracy = float((val_preds == y_val).mean())
    report = classification_report(
        y_val,
        val_preds,
        target_names=list(le.classes_),
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_val, val_preds).tolist()

    return {
        "training_seconds": round(time.time() - t0, 2),
        "k": k,
        "ngram_range": list(ngram_range),
        "vocab_size": len(vectorizer.vocabulary_),
        "train_size": int(len(X_train_raw)),
        "val_size": int(len(X_val_raw)),
        "classes": list(le.classes_),
        "accuracy": accuracy,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "_artifacts": {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "label_encoder": le,
        },
    }


def train(*, k: int = KMER_SIZE, save: bool = True) -> dict:
    """Public entry point used by the CLI — trains on the saved parquet splits."""
    train_df = _load_split("train")
    val_df = _load_split("val")

    metrics = train_on_dataframes(train_df, val_df, k=k)
    artifacts = metrics.pop("_artifacts")

    metrics_for_log = dict(metrics)
    print(json.dumps(metrics_for_log, indent=2))

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(DISEASE_CLASSIFIER_PATH, "wb") as fh:
            pickle.dump(artifacts["classifier"], fh)
        with open(DISEASE_VECTORIZER_PATH, "wb") as fh:
            pickle.dump(artifacts["vectorizer"], fh)
        with open(DISEASE_LABEL_ENCODER_PATH, "wb") as fh:
            pickle.dump(artifacts["label_encoder"], fh)
        (MODELS_DIR / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        for stale in (
            MODELS_DIR / "similarity_index.npz",
            MODELS_DIR / "similarity_index.meta.json",
            MODELS_DIR / "ood_stats.npz",
            MODELS_DIR / "ood_stats.meta.json",
        ):
            if stale.exists():
                stale.unlink()
        print(f"[save] artifacts written to {MODELS_DIR}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the disease classifier.")
    parser.add_argument("--k", type=int, default=KMER_SIZE)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    train(k=args.k, save=not args.no_save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
