"""Mutation impact analysis between a reference and a variant DNA sequence."""
from __future__ import annotations

from dataclasses import dataclass

from Bio.Seq import Seq

from ..core.sequence import clean_sequence
from .classifier import predict_sequence


@dataclass(slots=True)
class MutationEvent:
    """Single mismatch between reference and variant sequences."""

    position: int  # 1-indexed for human readability
    ref_base: str
    alt_base: str
    codon_index: int | None
    ref_codon: str | None
    alt_codon: str | None
    ref_aa: str | None
    alt_aa: str | None
    effect: str  # synonymous | missense | nonsense | frameshift_region | unknown

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "ref_base": self.ref_base,
            "alt_base": self.alt_base,
            "codon_index": self.codon_index,
            "ref_codon": self.ref_codon,
            "alt_codon": self.alt_codon,
            "ref_aa": self.ref_aa,
            "alt_aa": self.alt_aa,
            "effect": self.effect,
        }


@dataclass(slots=True)
class MutationReport:
    """Aggregated comparison report between two sequences."""

    aligned_length: int
    total_mismatches: int
    is_length_mismatch: bool
    mutation_rate: float
    events: list[MutationEvent]
    ref_prediction: dict
    alt_prediction: dict
    probability_shifts: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "aligned_length": self.aligned_length,
            "total_mismatches": self.total_mismatches,
            "is_length_mismatch": self.is_length_mismatch,
            "mutation_rate": round(self.mutation_rate, 4),
            "events": [e.to_dict() for e in self.events],
            "ref_prediction": self.ref_prediction,
            "alt_prediction": self.alt_prediction,
            "probability_shifts": {
                k: round(v, 4) for k, v in self.probability_shifts.items()
            },
        }


def _translate_codon(codon: str) -> str | None:
    if len(codon) != 3 or any(b not in "ACGT" for b in codon):
        return None
    try:
        return str(Seq(codon).translate())
    except Exception:
        return None


def _classify_effect(ref_aa: str | None, alt_aa: str | None) -> str:
    if ref_aa is None or alt_aa is None:
        return "unknown"
    if ref_aa == alt_aa:
        return "synonymous"
    if alt_aa == "*":
        return "nonsense"
    if ref_aa == "*" and alt_aa != "*":
        return "stop_loss"
    return "missense"


def compare(reference: str, variant: str, *, max_events: int = 200) -> MutationReport:
    """Compare two sequences and report mutation events plus prediction shifts."""
    ref = clean_sequence(reference, allow_n=False)
    alt = clean_sequence(variant, allow_n=False)
    aligned_len = min(len(ref), len(alt))
    is_len_mismatch = len(ref) != len(alt)

    events: list[MutationEvent] = []
    for i in range(aligned_len):
        if ref[i] != alt[i]:
            codon_idx = i // 3
            codon_start = codon_idx * 3
            ref_codon = ref[codon_start : codon_start + 3]
            alt_codon = alt[codon_start : codon_start + 3]
            ref_aa = _translate_codon(ref_codon) if len(ref_codon) == 3 else None
            alt_aa = _translate_codon(alt_codon) if len(alt_codon) == 3 else None
            events.append(
                MutationEvent(
                    position=i + 1,
                    ref_base=ref[i],
                    alt_base=alt[i],
                    codon_index=codon_idx,
                    ref_codon=ref_codon if len(ref_codon) == 3 else None,
                    alt_codon=alt_codon if len(alt_codon) == 3 else None,
                    ref_aa=ref_aa,
                    alt_aa=alt_aa,
                    effect=_classify_effect(ref_aa, alt_aa),
                )
            )
            if len(events) >= max_events:
                break

    ref_pred = predict_sequence(ref).to_dict()
    alt_pred = predict_sequence(alt).to_dict()

    classes = sorted(set(ref_pred["probabilities"]) | set(alt_pred["probabilities"]))
    shifts = {
        cls: alt_pred["probabilities"].get(cls, 0.0)
        - ref_pred["probabilities"].get(cls, 0.0)
        for cls in classes
    }

    return MutationReport(
        aligned_length=aligned_len,
        total_mismatches=len(events),
        is_length_mismatch=is_len_mismatch,
        mutation_rate=len(events) / aligned_len if aligned_len else 0.0,
        events=events,
        ref_prediction=ref_pred,
        alt_prediction=alt_pred,
        probability_shifts=shifts,
    )
