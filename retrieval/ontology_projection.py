


from typing import Any
from retrieval.settings import Settings

def _session_kwargs(database):
    if database:
        return {"database": database}
    return {}

def _index_exists(driver, index_name: str, database: str = None) -> bool:
    query = """
    SHOW INDEXES
    YIELD name
    WHERE name = $index_name
    RETURN count(*) > 0 AS exists
    """
    with driver.session(**_session_kwargs(database)) as session:
        rec = session.run(query, index_name=index_name).single()
        return bool(rec and rec["exists"])

def _create_fulltext_index_if_missing(driver, settings: Settings):
    if _index_exists(driver, settings.fulltext_index_name, settings.neo4j_database):
        return
    cypher = f"""
    CREATE FULLTEXT INDEX {settings.fulltext_index_name}
    IF NOT EXISTS
    FOR (n:{settings.node_label})
    ON EACH [n.name, n.{settings.retrieval_text_property}]
    """
    with driver.session(**_session_kwargs(settings.neo4j_database)) as session:
        session.run(cypher)

def _prepare_ontology_projection(driver, settings: Settings):
    properties_for_text = [
        "name", "type", "country", "tier", "category", "risk_tier", "location", "jurisdiction", "date", "severity", "entity_type", "source"
    ]
    labels_query = "CALL db.labels() YIELD label RETURN collect(label) AS labels"
    with driver.session(**_session_kwargs(settings.neo4j_database)) as session:
        labels_rec = session.run(labels_query).single()
        existing_labels = set(labels_rec["labels"] if labels_rec else [])
        active_source_labels = [lbl for lbl in settings.source_labels if lbl in existing_labels]
        if "Entity" in active_source_labels and len(active_source_labels) > 1:
            active_source_labels = [lbl for lbl in active_source_labels if lbl != "Entity"]
        if not active_source_labels:
            raise RuntimeError("No configured source labels found in DB. Set KG_SOURCE_LABELS to labels that exist in your graph.")
        session.run(f"MATCH (n:{settings.node_label}) REMOVE n:{settings.node_label}")
        query = f"""
        MATCH (n)
        WHERE n.name IS NOT NULL
          AND any(lbl IN labels(n) WHERE lbl IN $source_labels)
        WITH n, [lbl IN labels(n) WHERE lbl IN $source_labels][0] AS src_label, $prop_names AS prop_names
        SET n:{settings.node_label}
        WITH n, src_label, prop_names,
             reduce(txt = '', p IN prop_names |
                txt + CASE
                    WHEN n[p] IS NULL THEN ''
                    ELSE toString(n[p]) + ' '
                END
             ) AS composed
        SET n.{settings.retrieval_text_property} = trim(composed),
            n.{settings.retrieval_label_property} = src_label
        RETURN count(n) AS projected
        """
        session.run(query, source_labels=active_source_labels, prop_names=properties_for_text)
