"""Dataset cleaning + processed-feature generation utilities.

Run:
    python -m src.preprocess
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .core.config import DISEASE_DATASET_CSV, KMER_SIZE, MIN_SEQUENCE_LENGTH, PROCESSED_DIR
from .core.sequence import clean_sequence, kmers_as_text


def load_disease_dataset(path: Path = DISEASE_DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].astype(str).map(lambda s: clean_sequence(s, allow_n=False))
    df = df[df["sequence"].str.len() >= MIN_SEQUENCE_LENGTH].reset_index(drop=True)
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    # Merge curated NCBI reference sequences if available.
    try:
        from .core.reference_genes import load_reference_supplement

        ref = load_reference_supplement()
    except Exception:  # noqa: BLE001
        ref = pd.DataFrame()

    if not ref.empty:
        ref["sequence"] = ref["sequence"].astype(str).map(
            lambda s: clean_sequence(s, allow_n=False)
        )
        ref = ref[ref["sequence"].str.len() >= MIN_SEQUENCE_LENGTH]
        ref = ref[["sequence", "disease", "id"]]
        df = pd.concat([df, ref], ignore_index=True)
        df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    return df


def to_kmer_corpus(df: pd.DataFrame, k: int = KMER_SIZE) -> list[str]:
    return [kmers_as_text(seq, k) for seq in df["sequence"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preprocess the disease dataset.")
    parser.add_argument("--out", default=str(PROCESSED_DIR), help="Output directory.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_disease_dataset()
    corpus = to_kmer_corpus(df)
    le = LabelEncoder()
    y = le.fit_transform(df["disease"])

    np.save(out_dir / "kmer_corpus_meta.npy", np.asarray([len(corpus)]))
    pd.Series(corpus).to_csv(out_dir / "kmer_corpus.csv", index=False, header=False)
    np.save(out_dir / "y_labels.npy", y)
    pd.Series(le.classes_).to_csv(out_dir / "label_classes.csv", index=False, header=False)

    print(f"Processed {len(df)} sequences into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
