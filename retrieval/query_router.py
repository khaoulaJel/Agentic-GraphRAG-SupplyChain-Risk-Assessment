"""Intent-aware router for selecting retrieval Cypher templates."""

from __future__ import annotations

from typing import Any

from retrieval.cypher_templates import (
    COUNTRY_EXPOSURE_QUERY,
    fetch_subgraph,
)


RISK_INTENT_KEYWORDS = ("risk", "exposed", "political", "geopolit")


def route_query(query: str, resolved_entities: dict[str, list[str]], driver: Any) -> list[dict]:
    """
    Select the optimal Cypher template based on keyword intent heuristics.

    Returns a flat list of dict records ready for serialization.
    """
    query_lower = query.lower()
    all_names = [name for names in resolved_entities.values() for name in names]


    if any(keyword in query_lower for keyword in RISK_INTENT_KEYWORDS):
        target_entities = (
            resolved_entities.get("companies", []) +
            resolved_entities.get("products", []) +
            resolved_entities.get("organizations", [])
        )
        if target_entities:
            with driver.session() as session:
                result = session.run(COUNTRY_EXPOSURE_QUERY, company_names=target_entities)
                return [record.data() for record in result]

    return fetch_subgraph(all_names, driver)