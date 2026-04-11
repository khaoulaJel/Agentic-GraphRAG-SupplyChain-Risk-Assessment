"""Entity extraction helpers for GraphRAG retrieval."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


# Expanded entity extraction prompt to match current DB schema
ENTITY_EXTRACTION_PROMPT = """
You are a supply chain entity parser. Extract all named entities from the query below.

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
    "risk_events": []
}}

If a category has no entities, return an empty list. No explanations.

Query: {query}
"""

# Default to Gemini flash model to support lower-cost / free-tier usage.
DEFAULT_ENTITY_MODEL = os.getenv("RETRIEVAL_ENTITY_MODEL", "gemini-2.0-flash")


# Expanded default entity payload
DEFAULT_ENTITY_PAYLOAD = {
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
    "risk_events": []
}


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if an LLM wraps the JSON output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()
    return stripped


def build_default_entity_llm() -> ChatGoogleGenerativeAI:
    """Return a Gemini client suitable for free-tier entity extraction usage."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for default Gemini entity extraction.")
    return ChatGoogleGenerativeAI(model=DEFAULT_ENTITY_MODEL, api_key=api_key, temperature=0)


def _extract_text_from_response(response: Any) -> str:
    """Normalize common response objects to plain text."""
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
    # Anthropic-style client: client.messages.create(...)
    if hasattr(llm_client, "messages") and hasattr(llm_client.messages, "create"):
        response = llm_client.messages.create(
            model=DEFAULT_ENTITY_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        if hasattr(response, "content") and response.content:
            first = response.content[0]
            if hasattr(first, "text"):
                return str(first.text).strip()
        return _extract_text_from_response(response).strip()

    # LangChain-style client: client.invoke([...])
    if hasattr(llm_client, "invoke"):
        response = llm_client.invoke([HumanMessage(content=prompt)])
        return _extract_text_from_response(response).strip()

    raise TypeError("llm_client must support messages.create(...) or invoke(...).")


def extract_entities(query: str, llm_client: Any | None = None) -> dict[str, list[str]]:
    """Extract structured entity lists from a natural-language user query."""
    client = llm_client or build_default_entity_llm()
    prompt = ENTITY_EXTRACTION_PROMPT.format(query=query)
    raw = _call_entity_llm(prompt, client)
    parsed = json.loads(_strip_code_fences(raw))

    # Guarantee the expected payload shape even if one key is omitted.
    return {key: list(parsed.get(key, [])) for key in DEFAULT_ENTITY_PAYLOAD}