"""Deterministic Cypher templates for GraphRAG retrieval."""

from __future__ import annotations

from typing import Any

TWO_HOP_NEIGHBORHOOD = """
// Traverse any relationship type, but prioritize by relationship importance if possible
MATCH path = (anchor)-[*1..2]-(neighbor)
WHERE anchor.name IN $entity_names
WITH anchor, neighbor, relationships(path) AS rels,
     nodes(path) AS path_nodes,
     [r IN rels | type(r)] AS rel_types
// Optionally, filter/prioritize important relationships (e.g., SUPPLIES_TO, SOURCES_FROM, AFFECTS, CONTRIBUTES_TO, etc.)
// For now, return all, but you can add ORDER BY or WHERE for rel_types if needed
RETURN
    anchor.name         AS anchor_name,
    labels(anchor)      AS anchor_types,
    rel_types           AS relationship_types,
    neighbor.name       AS neighbor_name,
    labels(neighbor)    AS neighbor_types,
    properties(neighbor) AS neighbor_props
ORDER BY CASE WHEN 'SUPPLIES_TO' IN rel_types THEN 0 WHEN 'SOURCES_FROM' IN rel_types THEN 1 ELSE 2 END
LIMIT 150
"""
COUNTRY_EXPOSURE_QUERY = """
// Traverse any relationship type between Company and Material, and Material to Country/Location
MATCH (co:Company)-[*1..3]-(m:Material)-[*1..2]->(c)
WHERE co.name IN $company_names
    AND (c:Country OR c:Location)
RETURN
        co.name     AS company,
        m.name      AS material,
        c.name      AS country_or_location,
        c.risk_tier AS country_risk,
        labels(c)   AS location_labels
"""



PATH_COUNT_QUERY = """
MATCH path = (co:Company {name: $company})-[*1..3]->(m:Material {name: $material})
RETURN count(DISTINCT path) AS path_count
"""


def fetch_subgraph(entity_names: list[str], driver: Any) -> list[dict[str, Any]]:
    """Fetch the default 2-hop neighborhood around all resolved entity names."""
    if not entity_names:
        return []

    with driver.session() as session:
        result = session.run(TWO_HOP_NEIGHBORHOOD, entity_names=entity_names)
        return [record.data() for record in result]
