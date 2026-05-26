"""Dataset audit utilities — diagnoses leakage, imbalance and duplication."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..core.config import DISEASE_DATASET_CSV
from ..core.sequence import clean_sequence


@dataclass(slots=True)
class ClassAudit:
    name: str
    count: int
    unique_sequences: int
    duplicate_rate: float
    length_min: int
    length_max: int
    length_mean: float
    length_median: float
    length_std: float
    distinct_30char_prefixes: int


def _audit_one(name: str, sequences: pd.Series) -> ClassAudit:
    unique = sequences.nunique()
    lens = sequences.str.len()
    return ClassAudit(
        name=name,
        count=int(len(sequences)),
        unique_sequences=int(unique),
        duplicate_rate=float(1 - unique / max(len(sequences), 1)),
        length_min=int(lens.min()),
        length_max=int(lens.max()),
        length_mean=float(lens.mean()),
        length_median=float(lens.median()),
        length_std=float(lens.std()),
        distinct_30char_prefixes=int(sequences.str[:30].nunique()),
    )


def audit_dataset(path: Path = DISEASE_DATASET_CSV) -> dict:
    df = pd.read_csv(path)
    df["sequence"] = df["sequence"].astype(str).map(lambda s: clean_sequence(s, allow_n=False))
    df = df[df["sequence"].str.len() > 0]

    classes = sorted(df["disease"].unique())
    per_class = [
        asdict(_audit_one(c, df[df["disease"] == c]["sequence"])) for c in classes
    ]

    total = len(df)
    unique_total = df["sequence"].nunique()

    return {
        "total_records": total,
        "total_unique": int(unique_total),
        "global_duplicate_rate": float(1 - unique_total / max(total, 1)),
        "class_distribution": {c: int((df["disease"] == c).sum()) for c in classes},
        "per_class": per_class,
        "warnings": _build_warnings(per_class, total),
    }


def _build_warnings(per_class: list[dict], total: int) -> list[str]:
    warnings: list[str] = []
    counts = [c["count"] for c in per_class]
    if max(counts) >= 2 * min(counts):
        warnings.append(
            "Strong class imbalance — majority class is at least 2x the minority."
        )
    for c in per_class:
        if c["duplicate_rate"] > 0.3:
            warnings.append(
                f"{c['name']}: {c['duplicate_rate']*100:.1f}% duplicate sequences."
            )
        if c["distinct_30char_prefixes"] < 0.5 * c["count"]:
            warnings.append(
                f"{c['name']}: only {c['distinct_30char_prefixes']} distinct prefixes "
                f"in {c['count']} samples — likely many shifted variants of the same gene."
            )
    means = {c["name"]: c["length_mean"] for c in per_class}
    if max(means.values()) > 2 * min(means.values()):
        warnings.append(
            "Severe length disparity across classes — possible length-based leakage."
        )
    return warnings


def main() -> None:
    report = audit_dataset()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
