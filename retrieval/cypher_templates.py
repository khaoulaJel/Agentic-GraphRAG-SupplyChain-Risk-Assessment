
"""Supply-chain optimized Cypher templates matching the actual schema."""
from typing import Any

# Prioritized relationships for supply chain + risk (high weight = more important)
RELATION_WEIGHTS = {
    "SUPPLIES_TO": 8,
    "SOURCES_FROM": 8,
    "HAS_SUPPLIER": 7,
    "PRODUCES": 6,
    "MADE_FROM": 6,
    "AFFECTS": 7,
    "AFFECTED_BY": 7,
    "POSES_RISK": 7,
    "POTENTIAL_RISK": 6,
    "LOCATED_IN": 5,
    "OPERATES_IN": 5,
    "ALTERNATIVE_TO": 5,
    "PART_OF": 4,
    "COMPRISING": 4,
    "REGULATED_BY": 4,
    "SUBJECT_TO": 4,
    # Lower weight for noisy ones
    "COLLABORATES_WITH": 2,
    "MENTIONED_IN": 1,
}

HYBRID_SUPPLY_CHAIN_RETRIEVAL = """
// Hybrid anchor-based retrieval tailored to your schema
MATCH (anchor)
WHERE anchor.name IN $entity_names
   OR toLower(anchor.name) CONTAINS toLower($query_text)

// Expand with prioritized relationships, limit depth
MATCH path = (anchor)-[r*1..4]-(context)
WHERE all(rel IN relationships(path) WHERE type(rel) IN keys($relation_weights))

// Compute relevance score based on relationship importance
WITH anchor, path,
     reduce(score = 0, rel IN relationships(path) | 
            score + coalesce($relation_weights[type(rel)], 1)) AS path_score,
     size(relationships(path)) AS hops

RETURN 
    anchor.name AS anchor_name,
    labels(anchor) AS anchor_labels,
    [n IN nodes(path) | {
        name: n.name, 
        labels: labels(n), 
        risk_tier: n.risk_tier,
        tier: n.tier,
        country: n.country
    }] AS path_nodes,
    [rel IN relationships(path) | {
        type: type(rel),
        start: startNode(rel).name,
        end: endNode(rel).name
    }] AS path_rels,
    path_score,
    hops
ORDER BY path_score DESC, hops ASC
LIMIT 150
"""

COUNTRY_RISK_EXPOSURE = """
// Improved exposure analysis using your risk_tier and key supply/risk relations
MATCH (start)
WHERE start.name IN $entity_names

MATCH path = (start)-[*1..5]-(loc)
WHERE (loc:Country OR loc:Location)
  AND any(r IN relationships(path) WHERE type(r) IN ['SUPPLIES_TO', 'SOURCES_FROM', 'HAS_SUPPLIER', 'LOCATED_IN', 'OPERATES_IN', 'AFFECTS', 'POSES_RISK'])

WITH path,
     reduce(score = 0, r IN relationships(path) | 
            score + coalesce($relation_weights[type(r)], 1)) AS exposure_score,
     loc.name AS location_name,
     loc.risk_tier AS risk_tier

RETURN 
    [n IN nodes(path) | n.name] AS path_summary,
    exposure_score,
    risk_tier,
    location_name
ORDER BY exposure_score DESC, risk_tier DESC NULLS LAST
LIMIT 100
"""

def fetch_hybrid_subgraph(entity_names: list[str], query_text: str = "", driver: Any = None, relation_weights: dict = None) -> list[dict]:
    """Recommended main retrieval function for most queries."""
    if not entity_names or not driver:
        return []
    weights = relation_weights or RELATION_WEIGHTS
    with driver.session() as session:
        result = session.run(
            HYBRID_SUPPLY_CHAIN_RETRIEVAL,
            entity_names=[name for name in entity_names if name],
            query_text=query_text or "",
            relation_weights=weights
        )
        return [record.data() for record in result]

def fetch_country_exposure(entity_names: list[str], driver: Any = None) -> list[dict]:
    """Dedicated for risk/geopolitical queries."""
    if not entity_names or not driver:
        return []
    with driver.session() as session:
        result = session.run(
            COUNTRY_RISK_EXPOSURE,
            entity_names=entity_names,
            relation_weights=RELATION_WEIGHTS
        )
        return [record.data() for record in result]

# Fallbacks
TWO_HOP_NEIGHBORHOOD = """
MATCH path = (anchor)-[*1..2]-(neighbor)
WHERE anchor.name IN $entity_names
WITH anchor, neighbor, relationships(path) AS rels,
     nodes(path) AS path_nodes,
     [r IN rels | type(r)] AS rel_types
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
MATCH (startNode) WHERE startNode.name IN $entity_names
MATCH (loc) WHERE (loc:Country OR loc:Location)
MATCH path = shortestPath((startNode)-[*..5]-(loc))
WHERE any(r IN relationships(path) WHERE type(r) IN ['SUPPLIES_TO', 'SOURCES_FROM', 'AFFECTS'])
RETURN path, loc.risk_tier as risk
ORDER BY risk DESC
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
