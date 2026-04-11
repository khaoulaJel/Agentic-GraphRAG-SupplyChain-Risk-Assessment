"""Entity extraction helpers for GraphRAG retrieval."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# UPGRADE: relationship extraction added alongside entity extraction.
# Grounding: Edge et al. (Microsoft GraphRAG, 2024) show that extracting
# relationships at query time narrows graph traversal and reduces hallucination
# by giving the router explicit edge-type hints (e.g. SUPPLIES_TO, AFFECTS).
ENTITY_EXTRACTION_PROMPT = """
You are a supply chain entity parser. Extract all named entities AND relationships from the query below.

Return ONLY valid JSON in this exact format:
{{
    "companies": [],
    "countries": [],
    "entities": [],
    "facilities": [],
    "locations": [],
    "materials": [],
    "organizations": [],
    "products": [],
    "regulations": [],
    "regulatory_bodies": [],
    "risk_events": [],
    "relationships": []
}}

For "relationships", extract subject→predicate→object triples as strings, e.g.:
  "TSMC SUPPLIES_TO Apple", "China AFFECTS semiconductor supply"

If a category has no entities, return an empty list. No explanations, no markdown.

Query: {query}
"""

DEFAULT_ENTITY_MODEL = os.getenv("RETRIEVAL_ENTITY_MODEL", "gemini-2.0-flash")

# FIX: Anthropic client path was also passing DEFAULT_ENTITY_MODEL (a Gemini
# model string) to claude.messages.create — wrong SDK, wrong model name.
# Separate constant for the Anthropic model used in intent/entity calls.
DEFAULT_ANTHROPIC_MODEL = os.getenv(
    "RETRIEVAL_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
)

_MAX_RETRIES = int(os.getenv("ENTITY_EXTRACTION_RETRIES", "2"))

DEFAULT_ENTITY_PAYLOAD: dict[str, list] = {
    "companies": [],
    "countries": [],
    "entities": [],
    "facilities": [],
    "locations": [],
    "materials": [],
    "organizations": [],
    "products": [],
    "regulations": [],
    "regulatory_bodies": [],
    "risk_events": [],
    "relationships": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    return stripped


def build_default_entity_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for default Gemini entity extraction.")
    return ChatGoogleGenerativeAI(model=DEFAULT_ENTITY_MODEL, api_key=api_key, temperature=0)


def _extract_text_from_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, "text"):
                    parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(parts)
    if hasattr(response, "text"):
        return str(response.text)
    raise TypeError("Unsupported LLM response format for entity extraction.")


def _call_entity_llm(prompt: str, llm_client: Any) -> str:
    """Call either Anthropic-style or LangChain-style chat clients."""
    # Anthropic-style: client.messages.create(...)
    if hasattr(llm_client, "messages") and hasattr(llm_client.messages, "create"):
        # FIX: was passing DEFAULT_ENTITY_MODEL ("gemini-2.0-flash") to the
        # Anthropic SDK — invalid model string for that SDK.
        response = llm_client.messages.create(
            model=DEFAULT_ANTHROPIC_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        if hasattr(response, "content") and response.content:
            first = response.content[0]
            if hasattr(first, "text"):
                return str(first.text).strip()
        return _extract_text_from_response(response).strip()

    # LangChain-style: client.invoke([...])
    if hasattr(llm_client, "invoke"):
        response = llm_client.invoke([HumanMessage(content=prompt)])
        return _extract_text_from_response(response).strip()

    raise TypeError("llm_client must support messages.create(...) or invoke(...).")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(
    query: str, llm_client: Any | None = None
) -> dict[str, list[str]]:
    """
    Extract structured entity lists (and relationships) from a natural-language query.

    Retries up to _MAX_RETRIES times on JSON parse failure, since LLM formatting
    errors are non-deterministic (Bubeck et al., 2023).
    """
    client = llm_client or build_default_entity_llm()
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial + retries
        try:
            raw = _call_entity_llm(prompt, client)
            parsed = json.loads(_strip_code_fences(raw))
            return {key: list(parsed.get(key, [])) for key in DEFAULT_ENTITY_PAYLOAD}
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Entity extraction parse failed (attempt %d/%d): %s",
                attempt,
                _MAX_RETRIES + 1,
                exc,
            )
            if attempt <= _MAX_RETRIES:
                time.sleep(0.5 * attempt)  # simple back-off

    logger.error("Entity extraction failed after retries: %s", last_error)
    return dict(DEFAULT_ENTITY_PAYLOAD)