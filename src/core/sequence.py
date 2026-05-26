"""DNA sequence utilities: validation, statistics, k-mers, ORFs, FASTA parsing."""
from __future__ import annotations

import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

from .config import (
    EXTENDED_NUCLEOTIDES,
    KMER_SIZE,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    VALID_NUCLEOTIDES,
)

# ---------------------------------------------------------------------------
# Cleaning + validation
# ---------------------------------------------------------------------------

_NON_DNA_CHARS = re.compile(r"[^ACGTNacgtn]")
_WHITESPACE = re.compile(r"\s+")


def clean_sequence(sequence: str, *, allow_n: bool = True) -> str:
    """Uppercase, strip whitespace, and optionally drop ambiguous bases."""
    if not sequence:
        return ""
    seq = _WHITESPACE.sub("", sequence).upper()
    if allow_n:
        return _NON_DNA_CHARS.sub("", seq)
    return re.sub(r"[^ACGT]", "", seq)


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a DNA sequence."""

    is_valid: bool
    cleaned: str
    length: int
    invalid_chars: list[str]
    message: str | None = None

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "length": self.length,
            "invalid_chars": self.invalid_chars,
            "message": self.message,
        }


def validate_sequence(sequence: str, *, allow_n: bool = True) -> ValidationResult:
    """Validate a DNA sequence; return cleaned form + diagnostics."""
    if sequence is None:
        return ValidationResult(False, "", 0, [], "Sequence is empty.")

    raw = _WHITESPACE.sub("", sequence).upper()
    allowed = EXTENDED_NUCLEOTIDES if allow_n else VALID_NUCLEOTIDES
    invalid = sorted({c for c in raw if c not in allowed})
    cleaned = "".join(c for c in raw if c in allowed)
    length = len(cleaned)

    if length == 0:
        return ValidationResult(False, "", 0, invalid, "No valid DNA bases found.")
    if length < MIN_SEQUENCE_LENGTH:
        return ValidationResult(
            False,
            cleaned,
            length,
            invalid,
            f"Sequence too short. Minimum {MIN_SEQUENCE_LENGTH} bases required.",
        )
    if length > MAX_SEQUENCE_LENGTH:
        return ValidationResult(
            False,
            cleaned,
            length,
            invalid,
            f"Sequence too long. Maximum {MAX_SEQUENCE_LENGTH:,} bases supported.",
        )

    msg = None
    if invalid:
        msg = f"Removed {len(invalid)} invalid character type(s): {', '.join(invalid)}"

    return ValidationResult(True, cleaned, length, invalid, msg)


# ---------------------------------------------------------------------------
# Sequence statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SequenceStats:
    """Quality / composition statistics for a DNA sequence."""

    length: int
    gc_content: float
    at_content: float
    n_content: float
    base_counts: dict[str, int]
    shannon_entropy: float
    complexity_score: float  # 0..1, lower means more repetitive
    is_low_complexity: bool

    def to_dict(self) -> dict:
        return {
            "length": self.length,
            "gc_content": round(self.gc_content, 4),
            "at_content": round(self.at_content, 4),
            "n_content": round(self.n_content, 4),
            "base_counts": self.base_counts,
            "shannon_entropy": round(self.shannon_entropy, 4),
            "complexity_score": round(self.complexity_score, 4),
            "is_low_complexity": self.is_low_complexity,
        }


def compute_stats(sequence: str) -> SequenceStats:
    """Compute composition + complexity statistics."""
    seq = sequence.upper()
    length = len(seq)
    counts = Counter(seq)
    a = counts.get("A", 0)
    t = counts.get("T", 0)
    g = counts.get("G", 0)
    c = counts.get("C", 0)
    n = counts.get("N", 0)
    denom = max(length, 1)

    gc = (g + c) / denom
    at = (a + t) / denom
    n_pct = n / denom

    # Shannon entropy on observed bases
    entropy = 0.0
    for base in "ACGT":
        p = counts.get(base, 0) / denom
        if p > 0:
            entropy -= p * math.log2(p)

    # Complexity: ratio of distinct k-mers to total k-mers (linguistic complexity)
    k = 4
    if length >= k:
        kmers = [seq[i : i + k] for i in range(length - k + 1)]
        complexity = len(set(kmers)) / max(len(kmers), 1)
    else:
        complexity = 0.0

    return SequenceStats(
        length=length,
        gc_content=gc,
        at_content=at,
        n_content=n_pct,
        base_counts={"A": a, "T": t, "G": g, "C": c, "N": n},
        shannon_entropy=entropy,
        complexity_score=complexity,
        is_low_complexity=complexity < 0.5 or entropy < 1.5,
    )


# ---------------------------------------------------------------------------
# K-mer tokenization (must match training pipeline)
# ---------------------------------------------------------------------------

def get_kmers(sequence: str, k: int = KMER_SIZE) -> list[str]:
    """Slide a window of size *k* across the sequence, lowercased."""
    seq = sequence.lower()
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def kmers_as_text(sequence: str, k: int = KMER_SIZE) -> str:
    """Return space-separated k-mers compatible with the trained vectorizer."""
    return " ".join(get_kmers(sequence, k))


# ---------------------------------------------------------------------------
# Reverse complement & ORFs
# ---------------------------------------------------------------------------

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(_COMPLEMENT)[::-1]


@dataclass(slots=True)
class ORF:
    """Open Reading Frame information."""

    start: int  # 0-indexed start within the (forward) sequence
    end: int  # exclusive
    strand: str  # "+" or "-"
    frame: int  # 0, 1, or 2
    length: int  # nucleotides
    protein_length: int  # amino acids (excluding stop)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "frame": self.frame,
            "length": self.length,
            "protein_length": self.protein_length,
        }


def find_orfs(sequence: str, *, min_protein_length: int = 30) -> list[ORF]:
    """Find ORFs on both strands across all reading frames."""
    seq = sequence.upper()
    rev = reverse_complement(seq)
    orfs: list[ORF] = []
    for strand_label, strand_seq in (("+", seq), ("-", rev)):
        n = len(strand_seq)
        for frame in range(3):
            i = frame
            while i < n - 2:
                if strand_seq[i : i + 3] == "ATG":
                    j = i
                    while j < n - 2:
                        codon = strand_seq[j : j + 3]
                        if codon in ("TAA", "TAG", "TGA"):
                            protein_len = (j - i) // 3
                            if protein_len >= min_protein_length:
                                if strand_label == "+":
                                    start, end = i, j + 3
                                else:
                                    start, end = n - (j + 3), n - i
                                orfs.append(
                                    ORF(
                                        start=start,
                                        end=end,
                                        strand=strand_label,
                                        frame=frame,
                                        length=j + 3 - i,
                                        protein_length=protein_len,
                                    )
                                )
                            i = j + 3
                            break
                        j += 3
                    else:
                        break
                else:
                    i += 3
    return sorted(orfs, key=lambda o: o.protein_length, reverse=True)


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FastaRecord:
    """Single record from a FASTA stream."""

    record_id: str
    description: str
    sequence: str


def parse_fasta(text: str) -> list[FastaRecord]:
    """Parse FASTA-formatted text. Plain sequences are also accepted as a single record."""
    records: list[FastaRecord] = []
    current_id: str | None = None
    current_desc: str = ""
    current_chunks: list[str] = []

    for line in io.StringIO(text):
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append(
                    FastaRecord(
                        record_id=current_id,
                        description=current_desc,
                        sequence="".join(current_chunks),
                    )
                )
            header = line[1:].strip()
            parts = header.split(maxsplit=1)
            current_id = parts[0] if parts else "seq"
            current_desc = parts[1] if len(parts) > 1 else ""
            current_chunks = []
        else:
            current_chunks.append(line)

    if current_id is not None:
        records.append(
            FastaRecord(
                record_id=current_id,
                description=current_desc,
                sequence="".join(current_chunks),
            )
        )
    elif current_chunks or text.strip():
        # Treat whole input as a raw sequence
        records.append(
            FastaRecord(
                record_id="seq_1",
                description="raw input",
                sequence=text.strip(),
            )
        )

    return records


def iter_kmers(sequences: Iterable[str], k: int = KMER_SIZE) -> Iterator[str]:
    for seq in sequences:
        yield from get_kmers(seq, k)
