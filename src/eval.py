"""Cross-validation + hyperparameter grid search.

Run:
    # Re-build dataset first (only if not already built)
    python -m src.data.build_dataset

    # Quick CV on the current configuration
    python -m src.eval cv

    # Grid search over k and ngram ranges
    python -m src.eval grid

    # Both, save the best config + metrics report
    python -m src.eval grid --save-best
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .core.config import KMER_SIZE, MODELS_DIR, NGRAM_RANGE, PROCESSED_DIR
from .core.sequence import clean_sequence
from .data.augment import AugmentationConfig, expand_sequence
from .data.dedup import deduplicate
from .train import train_on_dataframes


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CVConfig:
    k: int = KMER_SIZE
    ngram_range: tuple[int, int] = NGRAM_RANGE
    n_splits: int = 5
    seed: int = 42
    target_per_class: int | None = None


def _load_clean_dataset() -> pd.DataFrame:
    """Load the raw CSV, clean it and dedup. Each row gets a stable parent_id."""
    from .core.config import DISEASE_DATASET_CSV, MIN_SEQUENCE_LENGTH

    df = pd.read_csv(DISEASE_DATASET_CSV)
    df["sequence"] = df["sequence"].astype(str).map(
        lambda s: clean_sequence(s, allow_n=False)
    )
    df = df[df["sequence"].str.len() >= MIN_SEQUENCE_LENGTH].reset_index(drop=True)
    df = deduplicate(df, similarity_threshold=0.95)
    df["parent_id"] = np.arange(len(df))
    return df


def _augment_split(
    df: pd.DataFrame,
    *,
    aug: AugmentationConfig,
    seed: int,
    balance_to: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _, r in df.iterrows():
        for frag in expand_sequence(r["sequence"], aug):
            rows.append(
                {
                    "sequence": frag,
                    "disease": r["disease"],
                    "parent_id": int(r["parent_id"]),
                }
            )
    out = pd.DataFrame(rows)
    if balance_to is None:
        return out

    balanced: list[pd.DataFrame] = []
    for cls in sorted(out["disease"].unique()):
        sub = out[out["disease"] == cls]
        if len(sub) > balance_to:
            sub = sub.sample(balance_to, random_state=seed)
        elif len(sub) < balance_to:
            sub = sub.sample(balance_to, random_state=seed, replace=True)
        balanced.append(sub)
    return (
        pd.concat(balanced)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def cross_validate(cfg: CVConfig | None = None) -> dict:
    """Run group-aware stratified K-fold cross-validation.

    Splits are made over **parent sequences** so that fragments of the same
    original sequence never appear in both train and validation, preventing
    leakage from the sliding-window augmentation.
    """
    cfg = cfg or CVConfig()
    df = _load_clean_dataset()

    train_aug = AugmentationConfig()
    val_aug = AugmentationConfig(
        window_sizes=(300, 600),
        stride_ratio=0.75,
        keep_full_sequence=True,
        add_reverse_complement=False,
    )

    splitter = StratifiedGroupKFold(
        n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed
    )
    fold_metrics: list[dict] = []
    aggregated_cm = np.zeros((4, 4), dtype=np.int64)
    classes: list[str] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(df["sequence"], df["disease"], groups=df["parent_id"]),
        start=1,
    ):
        print(f"\n[CV] fold {fold}/{cfg.n_splits} — train parents={len(train_idx)} val parents={len(val_idx)}")
        train_parents = df.iloc[train_idx]
        val_parents = df.iloc[val_idx]

        train_df = _augment_split(
            train_parents,
            aug=train_aug,
            seed=cfg.seed + fold,
            balance_to=cfg.target_per_class,
        )
        val_df = _augment_split(
            val_parents, aug=val_aug, seed=cfg.seed + fold, balance_to=None
        )
        if cfg.target_per_class is None:
            cfg = CVConfig(
                k=cfg.k,
                ngram_range=cfg.ngram_range,
                n_splits=cfg.n_splits,
                seed=cfg.seed,
                target_per_class=int(train_df["disease"].value_counts().median()),
            )
            # rebalance on first fold to the median
            train_df = _augment_split(
                train_parents,
                aug=train_aug,
                seed=cfg.seed + fold,
                balance_to=cfg.target_per_class,
            )

        result = train_on_dataframes(
            train_df,
            val_df,
            k=cfg.k,
            ngram_range=cfg.ngram_range,
            verbose=False,
        )
        result.pop("_artifacts", None)
        fold_metrics.append(result)
        cm = np.asarray(result["confusion_matrix"])
        if cm.shape == aggregated_cm.shape:
            aggregated_cm += cm
        classes = result["classes"]
        print(
            f"[CV] fold {fold} — accuracy={result['accuracy']:.3f} "
            f"macro_f1={result['macro_f1']:.3f} weighted_f1={result['weighted_f1']:.3f}"
        )

    macro_f1s = np.array([m["macro_f1"] for m in fold_metrics])
    accs = np.array([m["accuracy"] for m in fold_metrics])
    summary = {
        "config": {
            "k": cfg.k,
            "ngram_range": list(cfg.ngram_range),
            "n_splits": cfg.n_splits,
            "seed": cfg.seed,
            "target_per_class": cfg.target_per_class,
        },
        "fold_metrics": fold_metrics,
        "aggregated_confusion_matrix": aggregated_cm.tolist(),
        "classes": classes,
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "macro_f1_mean": float(macro_f1s.mean()),
        "macro_f1_std": float(macro_f1s.std()),
    }
    return summary


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    *,
    k_values: Iterable[int] = (4, 5, 6, 7),
    ngram_ranges: Iterable[tuple[int, int]] = ((1, 1), (1, 2), (4, 4)),
    n_splits: int = 3,
    seed: int = 42,
) -> dict:
    """Grid-search over (k, ngram_range) using a smaller K-fold CV per cell."""
    results: list[dict] = []
    t0 = time.time()
    for k in k_values:
        for ngram in ngram_ranges:
            print(f"\n=== Grid cell k={k} ngrams={ngram} ===")
            cfg = CVConfig(k=k, ngram_range=ngram, n_splits=n_splits, seed=seed)
            summary = cross_validate(cfg)
            results.append(
                {
                    "k": k,
                    "ngram_range": list(ngram),
                    "macro_f1_mean": summary["macro_f1_mean"],
                    "macro_f1_std": summary["macro_f1_std"],
                    "accuracy_mean": summary["accuracy_mean"],
                    "accuracy_std": summary["accuracy_std"],
                }
            )
    results.sort(key=lambda r: r["macro_f1_mean"], reverse=True)
    return {
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": results,
        "best": results[0] if results else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _save(name: str, payload: dict) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    out.write_text(json.dumps(payload, indent=2, default=float))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-validation and grid search.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cv = sub.add_parser("cv", help="Run K-fold cross-validation.")
    p_cv.add_argument("--k", type=int, default=KMER_SIZE)
    p_cv.add_argument("--n-splits", type=int, default=5)
    p_cv.add_argument("--seed", type=int, default=42)

    p_grid = sub.add_parser("grid", help="Grid search over k and ngram ranges.")
    p_grid.add_argument("--ks", type=int, nargs="+", default=[4, 5, 6, 7])
    p_grid.add_argument(
        "--ngram-ranges",
        type=str,
        nargs="+",
        default=["1,1", "1,2", "4,4"],
        help='Comma-separated ngram low,high pairs e.g. 1,1 1,2 4,4',
    )
    p_grid.add_argument("--n-splits", type=int, default=3)
    p_grid.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.cmd == "cv":
        summary = cross_validate(
            CVConfig(k=args.k, n_splits=args.n_splits, seed=args.seed)
        )
        out = _save(f"cv_report_k{args.k}.json", summary)
        print(json.dumps(
            {kk: vv for kk, vv in summary.items() if kk != "fold_metrics"},
            indent=2,
        ))
        print(f"\n[saved] {out}")
        return 0

    if args.cmd == "grid":
        ngram_ranges = []
        for token in args.ngram_ranges:
            lo, hi = token.split(",")
            ngram_ranges.append((int(lo), int(hi)))
        report = grid_search(
            k_values=tuple(args.ks),
            ngram_ranges=tuple(ngram_ranges),
            n_splits=args.n_splits,
            seed=args.seed,
        )
        out = _save("grid_search_report.json", report)
        print(json.dumps(report, indent=2))
        print(f"\n[saved] {out}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
