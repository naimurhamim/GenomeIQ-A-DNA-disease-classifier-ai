"""High-level chat orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import providers, retriever


@dataclass(slots=True)
class ChatRequest:
    message: str
    provider: str = "demo"
    top_k: int = 5
    history: list[dict] = field(default_factory=list)
    prediction_context: dict | None = None


@dataclass(slots=True)
class ChatResponse:
    answer: str
    provider: str
    retrieved: list[dict]
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "provider": self.provider,
            "retrieved": self.retrieved,
            "note": self.note,
        }


def chat(req: ChatRequest) -> ChatResponse:
    """Run a single chat turn: retrieve → generate → respond."""
    hits = retriever.search(req.message, top_k=req.top_k)

    history = [providers.ChatMessage(role=h.get("role", "user"), content=h.get("content", "")) for h in req.history]

    provider = (req.provider or "demo").lower()
    if provider == "gemini":
        result = providers.generate_gemini(
            req.message, hits, history=history, prediction_context=req.prediction_context
        )
    elif provider == "ollama":
        result = providers.generate_ollama(
            req.message, hits, history=history, prediction_context=req.prediction_context
        )
    else:
        result = providers.generate_demo(
            req.message, hits, history=history, prediction_context=req.prediction_context
        )

    return ChatResponse(
        answer=result.answer,
        provider=result.provider,
        retrieved=[h.to_dict() for h in hits],
        note=result.note,
    )


def status() -> dict[str, Any]:
    """Return availability of providers + KB stats."""
    from .knowledge_base import all_facts

    facts = all_facts()
    return {
        "providers": providers.list_provider_status(),
        "knowledge_base": {
            "facts_total": len(facts),
            "categories": sorted({f.category for f in facts}),
        },
    }
