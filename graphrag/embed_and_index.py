from __future__ import annotations

import math
import time
from typing import Any

from google import genai
from google.genai import types
from neo4j import GraphDatabase

from graphrag.config import (
    GOOGLE_API_KEY,
    GEMINI_EMBED_DIMENSIONS,
    GEMINI_EMBED_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEXES,
    session_kwargs,
    validate_config,
)


DIMENSIONS = GEMINI_EMBED_DIMENSIONS
EMBED_MODEL = GEMINI_EMBED_MODEL
BATCH_SIZE = 10
SLEEP_BETWEEN = 1.0

LABELS_TO_EMBED = [
    "RiskEvent",
    "Company",
    "Material",
    "Entity",
    "Location",
    "Country",
    "Regulation",
    "Product",
    "Organization",
    "Classification",
]

DROP_QUERIES = [
    "DROP INDEX entity_embedding IF EXISTS",
    "DROP INDEX entity_embedding_idx IF EXISTS",
    "DROP INDEX company_embedding IF EXISTS",
    "DROP INDEX node_embedding IF EXISTS",
    "DROP INDEX embedding IF EXISTS",
    "DROP INDEX supply_chain_index IF EXISTS",
]

client = genai.Client(api_key=GOOGLE_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def build_embed_text(node: dict[str, Any], label: str) -> str:
    if node.get("retrieval_text"):
        return str(node["retrieval_text"]).strip()

    name = str(node.get("name", ""))
    fallbacks = {
        "Company": f"{name} company tier:{node.get('tier', '')} country:{node.get('country', '')} type:{node.get('type', '')}",
        "Material": f"{name} material category:{node.get('category', '')}",
        "Country": f"{name} country risk_tier:{node.get('risk_tier', '')}",
        "Facility": f"{name} facility type:{node.get('type', '')}",
        "RiskEvent": f"{name} risk event type:{node.get('type', '')} date:{node.get('date', '')}",
        "RegulatoryBody": f"{name} regulatory body",
        "Organization": f"{name} organization",
        "Product": f"{name} product",
        "Classification": f"{name} classification",
        "Regulation": f"{name} regulation",
        "Location": f"{name} location",
        "Entity": f"{name} entity type:{node.get('entity_type', '')}",
    }
    return fallbacks.get(label, f"{label} {name}").strip()


def embed_text(text: str, retries: int = 3) -> list[float] | None:
    for attempt in range(retries):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vec = result.embeddings[0].values
            if len(vec) != DIMENSIONS:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {DIMENSIONS}, got {len(vec)}"
                )
            return vec
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Embedding error: {exc}")
                return None
    return None


def drop_old_indexes() -> None:
    print("STEP 0: Dropping old mismatched vector indexes...")
    with driver.session(**session_kwargs()) as session:
        for q in DROP_QUERIES:
            try:
                session.run(q)
                print(f"  Dropped: {q}")
            except Exception as exc:  # noqa: BLE001
                print(f"  Skip: {q} ({exc})")


def fetch_nodes(label: str) -> list[dict[str, Any]]:
    cypher = f"""
    MATCH (n:{label})
    WHERE n.embedding IS NULL
    RETURN n
    """
    with driver.session(**session_kwargs()) as session:
        rows = session.run(cypher)
        out: list[dict[str, Any]] = []
        for record in rows:
            node = dict(record["n"])
            node["_label"] = label
            out.append(node)
        return out


def write_embedding(label: str, name: str, embedding: list[float]) -> None:
    cypher = f"""
    MATCH (n:{label} {{name: $name}})
    SET n.embedding = $embedding,
        n.updated = timestamp()
    """
    with driver.session(**session_kwargs()) as session:
        session.run(cypher, name=name, embedding=embedding)


def create_vector_index(label: str) -> str:
    index_name = VECTOR_INDEXES[label]
    with driver.session(**session_kwargs()) as session:
        existing = session.run(
            """
            SHOW VECTOR INDEXES
            YIELD name, options
            WHERE name = $name
            RETURN name, options
            """,
            name=index_name,
        ).single()

        # If an index already exists with the wrong dimensions, rebuild it.
        if existing:
            options = existing.get("options") or {}
            index_cfg = options.get("indexConfig") or {}
            existing_dims = index_cfg.get("vector.dimensions")
            if existing_dims is not None and int(existing_dims) != DIMENSIONS:
                print(
                    f"  Rebuilding {index_name}: dimensions {existing_dims} -> {DIMENSIONS}"
                )
                session.run(f"DROP INDEX {index_name} IF EXISTS")

    cypher = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (n:{label}) ON (n.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {DIMENSIONS},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    with driver.session(**session_kwargs()) as session:
        session.run(cypher)
    print(f"  Vector index created: {index_name}")
    return index_name


def embed_label(label: str) -> dict[str, Any]:
    print("\n" + "=" * 50)
    print(f"Processing label: {label}")

    nodes = fetch_nodes(label)
    print(f"  Found {len(nodes)} nodes without embeddings")

    if not nodes:
        print("  Skipping - all nodes already embedded.")
        return {"label": label, "total": 0, "success": 0, "failed": 0}

    success = 0
    failed = 0
    batches = math.ceil(len(nodes) / BATCH_SIZE)

    for i in range(0, len(nodes), BATCH_SIZE):
        batch = nodes[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{batches} ({len(batch)} nodes)...")

        for node in batch:
            name = node.get("name")
            if not name:
                failed += 1
                continue

            text = build_embed_text(node, label)
            embedding = embed_text(text)
            if embedding:
                write_embedding(label, str(name), embedding)
                success += 1
            else:
                print(f"    Failed to embed: {name}")
                failed += 1

        if i + BATCH_SIZE < len(nodes):
            time.sleep(SLEEP_BETWEEN)

    return {"label": label, "total": len(nodes), "success": success, "failed": failed}


def run_full_pipeline() -> list[str]:
    validate_config()

    drop_old_indexes()

    print("\nSTEP 1: Embedding all nodes")
    print(f"Model: {EMBED_MODEL} | Dimensions: {DIMENSIONS}")

    stats = []
    for label in LABELS_TO_EMBED:
        stats.append(embed_label(label))

    print("\nSTEP 2: Creating vector indexes")
    index_names: list[str] = []
    for label in LABELS_TO_EMBED:
        index_names.append(create_vector_index(label))

    print("\n" + "=" * 50)
    print("EMBEDDING SUMMARY")
    print("=" * 50)
    total_success = 0
    for s in stats:
        print(f"  {s['label']:20} {s['success']:4} embedded, {s['failed']:3} failed")
        total_success += int(s["success"])

    print(f"\n  Total embedded: {total_success}")
    print(f"  Indexes created: {index_names}")

    return index_names


if __name__ == "__main__":
    names = run_full_pipeline()
    print("\nDone! Run verify_embeddings.py to confirm.")
    print(f"Indexes: {names}")
