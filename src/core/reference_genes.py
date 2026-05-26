"""Curated reference gene sequences per disease + NCBI Entrez fetcher.

Used by the training pipeline to fill coverage gaps in the dataset
(e.g. ensure APP, PSEN1, PSEN2, APOE reference fragments are present
for Alzheimer's; INS, GCK for Diabetes; BRCA1/2, TP53 for Cancer).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


# ---------------------------------------------------------------------------
# Canonical reference RefSeq IDs per disease, with a primary mRNA accession.
# Hand-picked from NCBI; conservative, well-known disease genes.
# ---------------------------------------------------------------------------

REFERENCE_GENES: dict[str, list[dict]] = {
    "Cancer": [
        {"gene": "BRCA1", "accession": "NM_007294.4", "description": "Breast cancer 1, DNA repair associated"},
        {"gene": "BRCA2", "accession": "NM_000059.4", "description": "Breast cancer 2, DNA repair associated"},
        {"gene": "TP53",  "accession": "NM_000546.6", "description": "Tumor protein p53"},
        {"gene": "EGFR",  "accession": "NM_005228.5", "description": "Epidermal growth factor receptor"},
        {"gene": "KRAS",  "accession": "NM_004985.5", "description": "KRAS proto-oncogene"},
        {"gene": "PTEN",  "accession": "NM_000314.8", "description": "Phosphatase and tensin homolog"},
        {"gene": "MYC",   "accession": "NM_002467.6", "description": "MYC proto-oncogene"},
    ],
    "Diabetes": [
        {"gene": "INS",   "accession": "NM_000207.3", "description": "Insulin"},
        {"gene": "INSR",  "accession": "NM_000208.4", "description": "Insulin receptor"},
        {"gene": "GCK",   "accession": "NM_000162.5", "description": "Glucokinase"},
        {"gene": "HNF1A", "accession": "NM_000545.8", "description": "HNF1 homeobox A"},
        {"gene": "HNF4A", "accession": "NM_175914.5", "description": "HNF4 alpha"},
        {"gene": "PPARG", "accession": "NM_138712.5", "description": "Peroxisome proliferator activated receptor gamma"},
    ],
    "Alzheimers": [
        {"gene": "APP",   "accession": "NM_000484.4", "description": "Amyloid beta precursor protein"},
        {"gene": "PSEN1", "accession": "NM_000021.4", "description": "Presenilin 1"},
        {"gene": "PSEN2", "accession": "NM_000447.3", "description": "Presenilin 2"},
        {"gene": "APOE",  "accession": "NM_000041.4", "description": "Apolipoprotein E"},
        {"gene": "MAPT",  "accession": "NM_005910.6", "description": "Microtubule associated protein tau"},
        {"gene": "TREM2", "accession": "NM_018965.4", "description": "Triggering receptor expressed on myeloid cells 2"},
    ],
    "Normal": [
        {"gene": "ACTB",  "accession": "NM_001101.5", "description": "Beta-actin"},
        {"gene": "GAPDH", "accession": "NM_002046.7", "description": "Glyceraldehyde-3-phosphate dehydrogenase"},
        {"gene": "B2M",   "accession": "NM_004048.4", "description": "Beta-2-microglobulin"},
        {"gene": "HPRT1", "accession": "NM_000194.3", "description": "Hypoxanthine phosphoribosyltransferase 1"},
        {"gene": "RPL13", "accession": "NM_000977.4", "description": "Ribosomal protein L13"},
    ],
}


REFERENCE_CSV: Path = DATA_DIR / "raw" / "reference_genes.csv"


@dataclass(slots=True)
class FetchResult:
    fetched: int
    skipped: int
    failed: int
    output_path: Path


def fetch_reference_sequences(
    *,
    email: str | None = None,
    output_path: Path = REFERENCE_CSV,
    overwrite: bool = False,
    delay_seconds: float = 0.4,
) -> FetchResult:
    """Fetch curated reference RefSeq mRNAs from NCBI Entrez.

    Requires an email per NCBI policy — pass via *email* arg or NCBI_EMAIL env var.
    Skips fetches that already exist on disk when *overwrite* is False.
    """
    from Bio import Entrez, SeqIO  # imported lazily

    Entrez.email = email or os.getenv("NCBI_EMAIL") or "research@example.com"

    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path)
        existing_keys = set(zip(existing["disease"], existing["accession"]))
    else:
        existing = pd.DataFrame(columns=["sequence", "disease", "id", "gene", "accession", "description"])
        existing_keys = set()

    rows: list[dict] = existing.to_dict("records")
    fetched = 0
    skipped = 0
    failed = 0

    for disease, genes in REFERENCE_GENES.items():
        for entry in genes:
            key = (disease, entry["accession"])
            if key in existing_keys:
                skipped += 1
                continue
            try:
                handle = Entrez.efetch(
                    db="nuccore", id=entry["accession"], rettype="fasta", retmode="text"
                )
                record = SeqIO.read(handle, "fasta")
                handle.close()
                seq = str(record.seq).upper().replace("\n", "")
                rows.append(
                    {
                        "sequence": seq,
                        "disease": disease,
                        "id": entry["accession"],
                        "gene": entry["gene"],
                        "accession": entry["accession"],
                        "description": entry["description"],
                    }
                )
                fetched += 1
                time.sleep(delay_seconds)  # be polite to NCBI
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[ncbi] failed {disease}/{entry['gene']}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return FetchResult(
        fetched=fetched, skipped=skipped, failed=failed, output_path=output_path
    )


def load_reference_supplement() -> pd.DataFrame:
    """Return the curated reference DataFrame, or an empty one if not built yet."""
    if not REFERENCE_CSV.exists():
        return pd.DataFrame(columns=["sequence", "disease", "id"])
    df = pd.read_csv(REFERENCE_CSV)
    return df[["sequence", "disease", "id"]].copy()
