from __future__ import annotations

import os
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


client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    vec = result.embeddings[0].values
    if len(vec) != GEMINI_EMBED_DIMENSIONS:
        raise RuntimeError(
            f"Query embedding dim mismatch: expected {GEMINI_EMBED_DIMENSIONS}, got {len(vec)}"
        )
    return vec


def _online_vector_indexes(session) -> set[str]:
    rows = session.run("SHOW VECTOR INDEXES YIELD name, state RETURN name, state")
    return {r["name"] for r in rows if str(r["state"]).upper() == "ONLINE"}


def vector_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    validate_config()
    query_vector = embed_query(query)
    all_results: list[dict[str, Any]] = []
    debug = os.getenv("GRAPHRAG_DEBUG_VECTOR", "0").lower() in {"1", "true", "yes"}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN
        node.name AS name,
        coalesce(node.retrieval_label, $label) AS label,
        node.retrieval_text AS retrieval_text,
        node.entity_type AS entity_type,
        node.source AS source,
        score
    ORDER BY score DESC
    """

    with driver.session(**session_kwargs()) as session:
        online_indexes = _online_vector_indexes(session)
        if debug:
            print(f"[vector] online indexes: {sorted(online_indexes)}")

        for label, index_name in VECTOR_INDEXES.items():
            if index_name not in online_indexes:
                if debug:
                    print(f"[vector] skip {label}: index '{index_name}' not online")
                continue
            try:
                rows = session.run(
                    cypher,
                    index_name=index_name,
                    embedding=query_vector,
                    top_k=top_k,
                    label=label,
                )
                materialized = [dict(r) for r in rows]
                all_results.extend(materialized)
                if debug:
                    print(f"[vector] {label}: {len(materialized)} hits from {index_name}")
            except Exception as exc:  # noqa: BLE001
                if debug:
                    print(f"[vector] {label}: query failed on {index_name} -> {exc}")
                continue

    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    if debug:
        print(f"[vector] merged hits={len(all_results)}; returning top_k={top_k}")
    return all_results[:top_k]
