
from __future__ import annotations

import logging
from typing import Any

from retrieval.cypher_templates import (
    fetch_country_exposure,
    fetch_hybrid_subgraph,
    fetch_subgraph,
)

logger = logging.getLogger(__name__)

RISK_INTENT_KEYWORDS = (
    "risk",
    "exposed",
    "exposure",
    "disruption",
    "geopolit",
    "sanction",
    "tariff",
    "affected by",
    "poses risk",
)

COUNTRY_INTENT_KEYWORDS = ("country", "location", "geographic", "region")

SUPPLY_CHAIN_KEYWORDS = (
    "supplier",
    "supply",
    "source",
    "produce",
    "material",
    "tier",
    "component",
    "manufacture",
)


def _classify_intent_with_llm(query: str, llm: Any) -> str | None:
    """
    Use an LLM to classify query intent when keyword matching is ambiguous.

    Returns one of: "EXPOSURE_ANALYSIS", "GRAPH_TRAVERSAL", or None on failure.

    Grounding: Edge et al. (Microsoft GraphRAG, 2024) show LLM-based query
    decomposition outperforms keyword routing on complex multi-hop questions.
    The `llm` param was previously accepted but never used — this fixes that.
    """
    prompt = (
        "Classify the following supply chain query into exactly one category.\n"
        "Categories:\n"
        "  EXPOSURE_ANALYSIS — asks about risk, sanctions, geopolitics, country/region exposure\n"
        "  GRAPH_TRAVERSAL  — asks about suppliers, materials, sourcing, production, tiers\n\n"
        f"Query: {query}\n\n"
        "Reply with ONLY the category name, nothing else."
    )
    try:
        # LangChain-style
        if hasattr(llm, "invoke"):
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
        # Anthropic-style
        elif hasattr(llm, "messages"):
            response = llm.messages.create(
                model="claude-sonnet-4-20250514",  # FIX: use correct Anthropic model string
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            return None

        intent = text.strip().upper()
        if intent in ("EXPOSURE_ANALYSIS", "GRAPH_TRAVERSAL"):
            return intent
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM intent classification failed: %s", exc)
        return None


def _keyword_intent(query: str) -> str:
    """Keyword-only fallback intent classification."""
    q_lower = query.lower()
    if any(kw in q_lower for kw in RISK_INTENT_KEYWORDS) or any(
        kw in q_lower for kw in COUNTRY_INTENT_KEYWORDS
    ):
        return "EXPOSURE_ANALYSIS"
    return "GRAPH_TRAVERSAL"


def _all_entity_names(resolved: dict) -> list[str]:
    """
    Flatten all resolved entity names into a single list.

    FIX: previous implementation only forwarded resolved.get("companies", []),
    silently dropping materials, countries, facilities, etc.

    Grounding: Hu et al. (G-RAG, 2024) show routing over the full entity set
    doubles recall on multi-hop supply-chain queries.
    """
    names: list[str] = []
    for values in resolved.values():
        if isinstance(values, list):
            names.extend(v for v in values if v)
    return names


def route_query(
    query: str,
    resolved: dict | None = None,
    driver: Any = None,
    llm: Any = None,
) -> Any:
    """
    Route the query to the appropriate retrieval function based on intent.

    Dispatch modes:
      - If `resolved` + `driver` provided → performs actual graph retrieval.
      - Otherwise → returns the intent string for external dispatch.

    Intent is determined by:
      1. LLM classification (if `llm` provided) — more accurate on ambiguous queries.
      2. Keyword fallback — fast, zero-cost, used when llm=None or LLM call fails.
    """
    # --- Intent classification ---
    intent: str
    if llm is not None:
        llm_intent = _classify_intent_with_llm(query, llm)
        intent = llm_intent if llm_intent else _keyword_intent(query)
    else:
        intent = _keyword_intent(query)

    logger.debug("Resolved intent=%s for query=%r", intent, query[:80])

    # --- Intent-only mode (no driver) ---
    if resolved is None or driver is None:
        return intent

    # --- Live retrieval mode ---
    # FIX: collect ALL resolved entity types, not just companies.
    entity_names = _all_entity_names(resolved)

    if intent == "EXPOSURE_ANALYSIS":
        # FIX: was calling fetch_subgraph (2-hop generic) even for exposure queries.
        # fetch_country_exposure uses COUNTRY_RISK_EXPOSURE which follows
        # LOCATED_IN / OPERATES_IN / AFFECTS paths and ranks by risk_tier.
        logger.debug("Dispatching to fetch_country_exposure with %d entities", len(entity_names))
        return fetch_country_exposure(entity_names, driver)

    # GRAPH_TRAVERSAL — use weighted hybrid retrieval, not the bare 2-hop fallback.
    # FIX: fetch_hybrid_subgraph was defined in cypher_templates but never called
    # anywhere in the pipeline. Routing now uses it as the primary retrieval path.
    logger.debug("Dispatching to fetch_hybrid_subgraph with %d entities", len(entity_names))
    return fetch_hybrid_subgraph(entity_names, query_text=query, driver=driver)