"""LLM provider abstraction with three free tiers."""
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Iterable

import requests


@dataclass(slots=True)
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass(slots=True)
class GenerationResult:
    answer: str
    provider: str
    note: str | None = None


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are GenomeIQ Copilot — a concise scientific assistant for a DNA disease
    classification tool. You help users understand classifier predictions and the
    underlying biology.

    Always:
        * Stay grounded in the retrieved knowledge-base context. If the context
          does not contain the answer, say so honestly.
        * Be concise (3-6 short paragraphs at most).
        * Mention relevant gene symbols and pathways when helpful.
        * Remind the user that GenomeIQ is a research tool, not a clinical service,
          when the user asks for diagnoses, treatments or medical advice.
        * Refuse to provide individualised medical advice.
    """
).strip()


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _format_context(retrieved: Iterable) -> str:
    lines = []
    for item in retrieved:
        f = item.fact
        lines.append(f"### {f.title}\n{f.content}\nSource: {f.source or 'GenomeIQ KB'}\n")
    return "\n".join(lines).strip()


def generate_demo(
    user_message: str,
    retrieved: Iterable,
    *,
    history: list[ChatMessage] | None = None,
    prediction_context: dict | None = None,
) -> GenerationResult:
    """Deterministic templated answer — no LLM, no API key, always works."""
    parts: list[str] = []
    if prediction_context:
        cls = prediction_context.get("predicted_class")
        conf = prediction_context.get("confidence")
        if cls is not None and conf is not None:
            parts.append(
                f"Your most recent prediction was **{cls}** with "
                f"{conf * 100:.1f}% confidence."
            )

    facts = list(retrieved)
    if facts:
        parts.append("Here is what I found in the knowledge base:")
        for hit in facts[:4]:
            parts.append(f"- **{hit.fact.title}.** {hit.fact.content}")
        if any(h.fact.source for h in facts):
            sources = {h.fact.source for h in facts if h.fact.source}
            parts.append("Sources: " + ", ".join(sorted(sources)))
    else:
        parts.append(
            "I couldn't find an exact match in the knowledge base for that question. "
            "Try mentioning a specific disease (Cancer, Diabetes, Alzheimer's, Normal) "
            "or gene symbol (BRCA1, INS, APP, ACTB ...)."
        )

    parts.append(
        "_GenomeIQ is a research preview and not a clinical service. "
        "Configure a Gemini or Ollama provider for richer natural-language answers._"
    )
    return GenerationResult(answer="\n\n".join(parts), provider="demo")


def generate_gemini(
    user_message: str,
    retrieved: Iterable,
    *,
    history: list[ChatMessage] | None = None,
    prediction_context: dict | None = None,
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
) -> GenerationResult:
    """Use Google Gemini's free-tier API to generate a grounded answer."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return GenerationResult(
            answer="Gemini provider selected but GEMINI_API_KEY is not set.",
            provider="gemini",
            note="Set GEMINI_API_KEY in the environment or via the settings panel.",
        )

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        return GenerationResult(
            answer="Gemini selected but google-genai package is not installed.",
            provider="gemini",
            note="Install with: pip install google-genai",
        )

    parts: list[str] = ["## Context"]
    parts.append(_format_context(retrieved) or "(no relevant facts found)")
    if prediction_context:
        parts.append("\n## Recent prediction")
        cls = prediction_context.get("predicted_class")
        conf = prediction_context.get("confidence")
        probs = prediction_context.get("probabilities") or {}
        line = f"Predicted class: {cls}, confidence {conf * 100:.1f}%."
        if probs:
            line += " Probabilities: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in probs.items())
        parts.append(line)

    parts.append("\n## User question")
    parts.append(user_message)
    user_prompt = "\n".join(parts)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
            ),
        )
        text = (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return GenerationResult(
            answer=f"Gemini call failed: {exc}",
            provider="gemini",
            note="Check the API key and your free-tier quota.",
        )

    if not text:
        return GenerationResult(
            answer="Gemini returned an empty response.",
            provider="gemini",
            note="The query may have been blocked by safety filters.",
        )
    return GenerationResult(answer=text, provider="gemini")


def generate_ollama(
    user_message: str,
    retrieved: Iterable,
    *,
    history: list[ChatMessage] | None = None,
    prediction_context: dict | None = None,
    base_url: str | None = None,
    model: str = "llama3.2:3b",
) -> GenerationResult:
    """Use a local Ollama server for offline generation."""
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    parts: list[str] = [SYSTEM_PROMPT, "", "## Context"]
    parts.append(_format_context(retrieved) or "(no relevant facts found)")
    if prediction_context:
        parts.append("\n## Recent prediction")
        cls = prediction_context.get("predicted_class")
        conf = prediction_context.get("confidence")
        line = f"Predicted class: {cls}, confidence {conf * 100:.1f}%."
        parts.append(line)
    parts.append("\n## User question")
    parts.append(user_message)
    prompt = "\n".join(parts)

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
    except requests.RequestException as exc:
        return GenerationResult(
            answer=f"Ollama is not reachable at {base_url}: {exc}",
            provider="ollama",
            note=(
                "Install Ollama from https://ollama.com and run "
                f"'ollama pull {model}' before retrying."
            ),
        )

    if resp.status_code != 200:
        return GenerationResult(
            answer=f"Ollama error {resp.status_code}: {resp.text[:200]}",
            provider="ollama",
        )

    data = resp.json()
    text = (data.get("response") or "").strip()
    if not text:
        return GenerationResult(
            answer="Ollama returned an empty response.",
            provider="ollama",
        )
    return GenerationResult(answer=text, provider="ollama")


def list_provider_status() -> dict:
    """Report which providers are usable on this machine right now."""
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    try:
        from google import genai  # noqa: F401  # type: ignore
        gemini_pkg_ok = True
    except ImportError:
        gemini_pkg_ok = False

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_ok = False
    try:
        ollama_ok = requests.get(f"{ollama_url}/api/tags", timeout=2).status_code == 200
    except Exception:
        pass

    return {
        "demo": {"available": True, "description": "Template-based, no LLM. Works offline."},
        "gemini": {
            "available": gemini_pkg_ok and has_gemini_key,
            "description": "Google Gemini (free tier, 1500 req/day).",
            "needs": (
                ([] if gemini_pkg_ok else ["pip install google-genai"])
                + ([] if has_gemini_key else ["GEMINI_API_KEY env var"])
            ),
        },
        "ollama": {
            "available": ollama_ok,
            "description": "Local Ollama runtime (offline, private).",
            "needs": [] if ollama_ok else ["Install Ollama from ollama.com and run `ollama serve`"],
        },
    }
