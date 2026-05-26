"""Build a clean, deduplicated, augmented training dataset.

Run:
    python -m src.data.build_dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import (
    DISEASE_DATASET_CSV,
    MIN_SEQUENCE_LENGTH,
    PROCESSED_DIR,
)
from ..core.sequence import clean_sequence
from .augment import AugmentationConfig, expand_sequence
from .dedup import deduplicate


def build(
    *,
    raw_csv: Path = DISEASE_DATASET_CSV,
    output_dir: Path = PROCESSED_DIR,
    aug: AugmentationConfig | None = None,
    target_per_class: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Load → clean → dedup → augment → balance → train/val split → save."""
    aug = aug or AugmentationConfig()
    rng = np.random.default_rng(seed)

    df = pd.read_csv(raw_csv)
    df["sequence"] = df["sequence"].astype(str).map(
        lambda s: clean_sequence(s, allow_n=False)
    )
    df = df[df["sequence"].str.len() >= MIN_SEQUENCE_LENGTH].reset_index(drop=True)

    raw_count = len(df)
    df = deduplicate(df, similarity_threshold=0.95)
    deduped_count = len(df)

    # Group-aware split BEFORE augmentation: same parent goes to one side only.
    df["parent_id"] = np.arange(len(df))

    train_parents: list[int] = []
    val_parents: list[int] = []
    for cls in sorted(df["disease"].unique()):
        ids = df[df["disease"] == cls]["parent_id"].tolist()
        rng.shuffle(ids)
        cutoff = max(1, int(len(ids) * (1 - test_size)))
        train_parents.extend(ids[:cutoff])
        val_parents.extend(ids[cutoff:])

    train_df = df[df["parent_id"].isin(train_parents)].reset_index(drop=True)
    val_df = df[df["parent_id"].isin(val_parents)].reset_index(drop=True)

    # Augment training set
    train_records: list[dict] = []
    for _, row in train_df.iterrows():
        for frag in expand_sequence(row["sequence"], aug):
            train_records.append(
                {
                    "sequence": frag,
                    "disease": row["disease"],
                    "parent_id": int(row["parent_id"]),
                }
            )
    train_aug = pd.DataFrame(train_records)

    # Augment validation lightly: only sliding (no RC) to keep evaluation realistic
    val_aug_cfg = AugmentationConfig(
        window_sizes=(300, 600),
        stride_ratio=0.75,
        keep_full_sequence=True,
        add_reverse_complement=False,
        min_length=aug.min_length,
        max_chunks_per_sequence=aug.max_chunks_per_sequence,
    )
    val_records: list[dict] = []
    for _, row in val_df.iterrows():
        for frag in expand_sequence(row["sequence"], val_aug_cfg):
            val_records.append(
                {
                    "sequence": frag,
                    "disease": row["disease"],
                    "parent_id": int(row["parent_id"]),
                }
            )
    val_aug = pd.DataFrame(val_records)

    # Class balancing (training only) — undersample to target
    if target_per_class is None:
        target_per_class = int(train_aug["disease"].value_counts().median())
    balanced: list[pd.DataFrame] = []
    for cls in sorted(train_aug["disease"].unique()):
        sub = train_aug[train_aug["disease"] == cls]
        if len(sub) > target_per_class:
            sub = sub.sample(target_per_class, random_state=seed)
        elif len(sub) < target_per_class:
            sub = sub.sample(target_per_class, random_state=seed, replace=True)
        balanced.append(sub)
    train_balanced = pd.concat(balanced).sample(frac=1, random_state=seed).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_balanced.to_parquet(output_dir / "train.parquet", index=False)
    val_aug.to_parquet(output_dir / "val.parquet", index=False)

    summary = {
        "raw_records": int(raw_count),
        "after_dedup": int(deduped_count),
        "train_parents": len(train_parents),
        "val_parents": len(val_parents),
        "train_augmented_total": int(len(train_aug)),
        "train_balanced_total": int(len(train_balanced)),
        "val_augmented_total": int(len(val_aug)),
        "train_class_distribution": train_balanced["disease"].value_counts().to_dict(),
        "val_class_distribution": val_aug["disease"].value_counts().to_dict(),
        "target_per_class": int(target_per_class),
        "augmentation": {
            "window_sizes": list(aug.window_sizes),
            "stride_ratio": aug.stride_ratio,
            "keep_full_sequence": aug.keep_full_sequence,
            "add_reverse_complement": aug.add_reverse_complement,
            "min_length": aug.min_length,
        },
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build training dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--target-per-class", type=int, default=None)
    parser.add_argument("--no-rc", action="store_true", help="Disable reverse-complement aug.")
    args = parser.parse_args(argv)

    aug = AugmentationConfig(add_reverse_complement=not args.no_rc)
    summary = build(
        aug=aug,
        target_per_class=args.target_per_class,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
