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

def route_query(query: str, resolved=None, driver=None, llm=None):
    """
    Route the query to the appropriate retrieval function based on intent.
    If resolved and driver are provided, dispatch to the correct retrieval function.
    """
    q_lower = query.lower()
    # If resolved and driver are provided, do actual retrieval
    if resolved is not None and driver is not None:
        if any(kw in q_lower for kw in RISK_INTENT_KEYWORDS) or any(word in q_lower for word in ["country", "location", "geographic"]):
            # Use country exposure retrieval
            companies = resolved.get("companies", [])
            return fetch_subgraph(companies, driver)  # or fetch_country_exposure(companies, driver) if available
        # Default: supply chain subgraph
        companies = resolved.get("companies", [])
        return fetch_subgraph(companies, driver)
    # Otherwise, just intent classification
    if any(kw in q_lower for kw in RISK_INTENT_KEYWORDS) or any(word in q_lower for word in ["country", "location", "geographic"]):
        return "EXPOSURE_ANALYSIS"
    if any(word in q_lower for word in ["supplier", "supply", "source", "produce", "material", "tier"]):
        return "GRAPH_TRAVERSAL"
    return "GRAPH_TRAVERSAL"  # default