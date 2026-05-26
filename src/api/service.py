"""High-level orchestration that the API endpoints call into."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from ..core.knowledge import DISEASE_INFO, list_diseases
from ..core.sequence import (
    compute_stats,
    find_orfs,
    parse_fasta,
    validate_sequence,
)
from ..ml.classifier import load_models, predict_sequence
from ..ml.explain import saliency
from ..ml.mutation import compare as compare_mutations
from ..ml.ood import detect as detect_ood
from ..ml.similarity import find_similar


def warmup() -> None:
    """Pre-load models and indices at startup."""
    load_models()
    try:
        from ..ml.similarity import build_similarity_index
        from ..ml.ood import build_ood_stats

        build_similarity_index(force=False)
        build_ood_stats(force=False)
    except FileNotFoundError:
        # Dataset missing — skip silently; endpoints will handle gracefully.
        pass

    # Warm up RAG retriever (sentence-transformer + FAISS index)
    try:
        from ..rag import retriever as rag_retriever

        rag_retriever.warmup()
    except Exception:
        # RAG is optional; never block API startup if it fails to warm
        pass


def health() -> dict[str, Any]:
    bundle = load_models()
    return {"status": "ok", "version": "0.2.0", "classes": bundle.classes}


def diseases_payload() -> list[dict]:
    return [d.to_dict() for d in list_diseases()]


def disease_payload(name: str) -> dict:
    info = DISEASE_INFO.get(name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown disease: {name}")
    return info.to_dict()


def validate_payload(sequence: str) -> dict[str, Any]:
    result = validate_sequence(sequence)
    payload: dict[str, Any] = result.to_dict()
    if result.is_valid:
        payload["stats"] = compute_stats(result.cleaned).to_dict()
    return payload


def stats_payload(sequence: str) -> dict[str, Any]:
    result = validate_sequence(sequence)
    if not result.is_valid:
        raise HTTPException(status_code=400, detail=result.message or "Invalid sequence.")
    return {
        "sequence_length": result.length,
        "stats": compute_stats(result.cleaned).to_dict(),
        "orfs": [orf.to_dict() for orf in find_orfs(result.cleaned)[:10]],
    }


def predict_payload(
    sequence: str,
    *,
    include_explain: bool = True,
    include_ood: bool = True,
    include_similar: bool = True,
    top_k_similar: int = 5,
    model: str = "tfidf",
) -> dict[str, Any]:
    result = validate_sequence(sequence)
    if not result.is_valid:
        raise HTTPException(status_code=400, detail=result.message or "Invalid sequence.")

    cleaned = result.cleaned

    if model == "tfidf":
        prediction = predict_sequence(cleaned)
        prediction_dict = prediction.to_dict()
        prediction_dict["model"] = "tfidf"
        primary_class = prediction.predicted_class
    elif model == "dnabert2":
        from ..ml import dnabert as dnabert_mod

        if not dnabert_mod.is_available():
            raise HTTPException(
                status_code=503,
                detail=(
                    "DNABERT-2 head not trained. Run "
                    "`python -m src.ml.dnabert embed --source train` then "
                    "`python -m src.ml.dnabert train`."
                ),
            )
        dna_pred = dnabert_mod.predict_sequence_dnabert(cleaned)
        prediction_dict = dna_pred.to_dict()
        prediction_dict["model"] = "dnabert2"
        prediction_dict["top_kmers"] = []  # not applicable for transformer
        primary_class = dna_pred.predicted_class
    elif model == "ensemble":
        from ..ml import dnabert as dnabert_mod

        tfidf_pred = predict_sequence(cleaned)
        if dnabert_mod.is_available():
            dna_pred = dnabert_mod.predict_sequence_dnabert(cleaned)
            classes = sorted(set(tfidf_pred.probabilities) | set(dna_pred.probabilities))
            avg = {
                c: (
                    tfidf_pred.probabilities.get(c, 0.0)
                    + dna_pred.probabilities.get(c, 0.0)
                )
                / 2
                for c in classes
            }
            best = max(avg, key=avg.get)
            confidence = avg[best]
            entropy = _entropy_dict(avg)
            prediction_dict = {
                "predicted_class": best,
                "confidence": round(confidence, 4),
                "entropy": round(entropy, 4),
                "probabilities": {k: round(v, 4) for k, v in avg.items()},
                "top_kmers": tfidf_pred.top_kmers,
                "model": "ensemble",
                "components": {
                    "tfidf": tfidf_pred.to_dict(),
                    "dnabert2": dna_pred.to_dict(),
                },
            }
            primary_class = best
        else:
            # Gracefully fall back to TF-IDF
            prediction_dict = tfidf_pred.to_dict()
            prediction_dict["model"] = "tfidf"
            prediction_dict["note"] = "DNABERT-2 unavailable, used TF-IDF only."
            primary_class = tfidf_pred.predicted_class
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    stats = compute_stats(cleaned)
    orfs = [orf.to_dict() for orf in find_orfs(cleaned)[:10]]

    payload: dict[str, Any] = {
        "sequence_length": result.length,
        "validation": result.to_dict(),
        "stats": stats.to_dict(),
        "prediction": prediction_dict,
        "orfs": orfs,
    }

    if include_explain:
        payload["explanation"] = saliency(
            cleaned, target_class=primary_class
        ).to_dict()

    if include_ood:
        try:
            payload["ood"] = detect_ood(cleaned).to_dict()
        except FileNotFoundError as exc:
            payload["ood"] = {"error": str(exc)}

    if include_similar:
        try:
            hits = find_similar(cleaned, top_k=top_k_similar)
            payload["similar"] = [hit.to_dict() for hit in hits]
        except FileNotFoundError as exc:
            payload["similar"] = [{"error": str(exc)}]

    return payload


def _entropy_dict(probs: dict[str, float]) -> float:
    import math

    total = sum(probs.values()) or 1.0
    entropy = 0.0
    n = 0
    for v in probs.values():
        p = v / total
        if p > 0:
            entropy -= p * math.log(p)
        n += 1
    h_max = math.log(n) if n > 1 else 1.0
    return entropy / h_max if h_max > 0 else 0.0


def models_status() -> dict[str, Any]:
    """Return status of available prediction models."""
    from ..ml import dnabert as dnabert_mod

    return {
        "tfidf": {"available": True, "description": "TF-IDF k-mers + RF/SVM/LR ensemble"},
        "dnabert2": {
            "available": dnabert_mod.is_available(),
            "description": "DNABERT-2 transformer + calibrated logistic head",
        },
        "ensemble": {
            "available": dnabert_mod.is_available(),
            "description": "Average of TF-IDF and DNABERT-2 probabilities",
        },
    }


def mutation_payload(reference: str, variant: str) -> dict[str, Any]:
    ref_check = validate_sequence(reference)
    alt_check = validate_sequence(variant)
    if not ref_check.is_valid:
        raise HTTPException(status_code=400, detail=f"Reference: {ref_check.message}")
    if not alt_check.is_valid:
        raise HTTPException(status_code=400, detail=f"Variant: {alt_check.message}")
    report = compare_mutations(ref_check.cleaned, alt_check.cleaned)
    return report.to_dict()


def batch_payload(file_text: str) -> dict[str, Any]:
    records = parse_fasta(file_text)
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0
    for rec in records:
        check = validate_sequence(rec.sequence)
        if not check.is_valid:
            failed += 1
            results.append(
                {
                    "record_id": rec.record_id,
                    "description": rec.description,
                    "length": check.length,
                    "prediction": None,
                    "error": check.message,
                }
            )
            continue
        try:
            pred = predict_sequence(check.cleaned)
            results.append(
                {
                    "record_id": rec.record_id,
                    "description": rec.description,
                    "length": check.length,
                    "prediction": pred.to_dict(),
                    "error": None,
                }
            )
            successful += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            results.append(
                {
                    "record_id": rec.record_id,
                    "description": rec.description,
                    "length": check.length,
                    "prediction": None,
                    "error": f"Prediction error: {exc}",
                }
            )
    return {
        "total": len(records),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
