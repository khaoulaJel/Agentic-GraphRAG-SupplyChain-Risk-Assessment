"""
builder.py — Neo4j graph builder for EV Battery Supply Chain.
Provides helpers to connect to Neo4j and ingest triples.
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

_driver = None


def get_driver():
    """Return a singleton Neo4j driver."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )
    return _driver


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def run_cypher(query: str, **params) -> list[dict]:
    """Execute a Cypher query and return results as list of dicts."""
    with get_driver().session() as session:
        result = session.run(query, **params)
        return [record.data() for record in result]


def merge_triple(subject: str, subject_label: str,
                 relation: str,
                 obj: str, obj_label: str):
    """MERGE a single (subject)-[relation]->(object) triple into Neo4j."""
    cypher = (
        f"MERGE (a:{subject_label} {{name: $subj}}) "
        f"MERGE (b:{obj_label} {{name: $obj}}) "
        f"MERGE (a)-[:{relation}]->(b)"
    )
    run_cypher(cypher, subj=subject, obj=obj)


def ingest_triples(triples: list[dict]):
    """Ingest a list of {subject, subject_label, relation, object, object_label} dicts."""
    for t in triples:
        merge_triple(
            t["subject"], t.get("subject_label", "Company"),
            t["relation"],
            t["object"], t.get("object_label", "Company"),
        )
