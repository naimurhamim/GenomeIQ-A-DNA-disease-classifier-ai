"""Pydantic schemas for the public API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SequenceRequest(BaseModel):
    sequence: str = Field(..., description="Raw DNA sequence (A/T/G/C/N).")
    include_explain: bool = Field(
        default=True, description="Compute saliency / per-base contribution scores."
    )
    include_ood: bool = Field(
        default=True, description="Run out-of-distribution detection."
    )
    include_similar: bool = Field(
        default=True, description="Return nearest known sequences."
    )
    top_k_similar: int = Field(default=5, ge=1, le=25)
    model: str = Field(
        default="tfidf",
        description="Prediction backbone: 'tfidf' (default), 'dnabert2', or 'ensemble'.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    classes: list[str]


class DiseaseInfoResponse(BaseModel):
    name: str
    icon: str
    color: str
    short_description: str
    long_description: str
    key_genes: list[str]
    pathways: list[str]
    references: list[str]


class ValidationResponse(BaseModel):
    is_valid: bool
    length: int
    invalid_chars: list[str]
    message: str | None = None
    stats: dict[str, Any] | None = None


class PredictionResponse(BaseModel):
    sequence_length: int
    validation: dict[str, Any]
    stats: dict[str, Any]
    prediction: dict[str, Any]
    explanation: dict[str, Any] | None = None
    ood: dict[str, Any] | None = None
    similar: list[dict[str, Any]] | None = None
    orfs: list[dict[str, Any]] | None = None


class MutationRequest(BaseModel):
    reference: str
    variant: str


class ChatRequestSchema(BaseModel):
    message: str = Field(..., description="User question.")
    provider: str = Field(default="demo", description="LLM provider: demo, gemini, ollama.")
    top_k: int = Field(default=5, ge=1, le=15)
    history: list[dict[str, Any]] = Field(default_factory=list)
    prediction_context: dict[str, Any] | None = Field(
        default=None, description="Optional last prediction to ground the answer."
    )


class BatchPredictionItem(BaseModel):
    record_id: str
    description: str
    length: int
    prediction: dict[str, Any] | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: list[BatchPredictionItem]


class StatsResponse(BaseModel):
    sequence_length: int
    stats: dict[str, Any]
    orfs: list[dict[str, Any]]
