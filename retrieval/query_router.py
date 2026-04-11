"""Intent-aware router for selecting retrieval Cypher templates."""

from __future__ import annotations

from typing import Any

from retrieval.cypher_templates import (
    COUNTRY_EXPOSURE_QUERY,
    fetch_subgraph,
)


RISK_INTENT_KEYWORDS = ("risk", "exposed", "political", "geopolit")




# Supply chain risk-aware query router
RISK_INTENT_KEYWORDS = (
    "risk", "exposed", "exposure", "disruption", "geopolit", "sanction", "tariff", "affected by", "poses risk"
)

def route_query(query: str, llm=None):
    q_lower = query.lower()
    if any(kw in q_lower for kw in RISK_INTENT_KEYWORDS) or any(word in q_lower for word in ["country", "location", "geographic"]):
        return "EXPOSURE_ANALYSIS"
    if any(word in q_lower for word in ["supplier", "supply", "source", "produce", "material", "tier"]):
        return "GRAPH_TRAVERSAL"
    return "GRAPH_TRAVERSAL"  # default