"""Intent-aware router for selecting retrieval Cypher templates."""

from __future__ import annotations

from typing import Any

from retrieval.cypher_templates import (
    COUNTRY_EXPOSURE_QUERY,
    fetch_subgraph,
)


RISK_INTENT_KEYWORDS = ("risk", "exposed", "political", "geopolit")



# Agentic query router using LLM to select retrieval tool
def route_query_agentically(query: str, llm):
    prompt = f"""
    Analyze this supply chain query: \"{query}\"
    Decide which retrieval tool is most appropriate:

    1. EXPOSURE_ANALYSIS: Use if the user is asking about risks, dependencies, 
       or how a company/product is affected by a country or material.
    2. KNOWLEDGE_GRAPH_SEARCH: Use for general \"What is\" or \"Tell me about\" questions.
    3. WEB_SEARCH: Use if the query requires 2024-2026 real-time news not in a database.

    Return ONLY the name of the tool.
    """
    decision = llm.invoke(prompt).content
    return decision