from __future__ import annotations

from google import genai
from google.genai import types
from neo4j import GraphDatabase

from graphrag.config import (
    GOOGLE_API_KEY,
    GEMINI_EMBED_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    VECTOR_INDEXES,
    session_kwargs,
)


LABELS = [
    "Entity",
    "Company",
    "Location",
    "Material",
    "Product",
    "Regulation",
    "RiskEvent",
    "Organization",
    "Classification",
    "Country",
]

client = genai.Client(api_key=GOOGLE_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def check_coverage() -> None:
    print("\n-- Embedding Coverage --")
    with driver.session(**session_kwargs()) as session:
        for label in LABELS:
            row = session.run(
                f"""
                MATCH (n:{label})
                RETURN count(n) AS total,
                       count(n.embedding) AS has_embedding
                """
            ).single()

            dim_row = session.run(
                f"""
                MATCH (n:{label})
                WHERE n.embedding IS NOT NULL
                RETURN size(n.embedding) AS dims
                LIMIT 1
                """
            ).single()

            total = int(row["total"]) if row else 0
            has_emb = int(row["has_embedding"]) if row else 0
            dims = int(dim_row["dims"]) if dim_row else 0
            pct = (has_emb / total * 100.0) if total > 0 else 0.0
            status = "OK" if has_emb == total else "WARN"
            print(f"  [{status}] {label:15} {has_emb:4}/{total:4} ({pct:5.1f}%) dims={dims}")


def check_indexes() -> None:
    print("\n-- Vector Indexes --")
    with driver.session(**session_kwargs()) as session:
        rows = session.run(
            """
            SHOW VECTOR INDEXES
            YIELD name, state, labelsOrTypes, properties, options
            RETURN name, state, labelsOrTypes, properties, options
            ORDER BY name
            """
        )
        for row in rows:
            print(
                f"  {row['name']:35} state={row['state']:<8} "
                f"labels={row['labelsOrTypes']} props={row['properties']}"
            )


def _embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def test_vector_search(query: str = "child labor risk in mining") -> None:
    print("\n-- Test Vector Search --")
    print(f"  Query: {query}")
    qvec = _embed_query(query)
    print(f"  Query embedding dims: {len(qvec)}")

    with driver.session(**session_kwargs()) as session:
        for label in ["RiskEvent", "Company", "Material", "Entity"]:
            index_name = VECTOR_INDEXES[label]
            try:
                rows = session.run(
                    """
                    CALL db.index.vector.queryNodes($index, 3, $vec)
                    YIELD node, score
                    RETURN node.name AS name,
                           node.retrieval_text AS text,
                           score
                    ORDER BY score DESC
                    """,
                    index=index_name,
                    vec=qvec,
                )
                out = [dict(r) for r in rows]
                if not out:
                    print(f"  [{label}] no results")
                    continue
                print(f"  [{label}] top results:")
                for r in out:
                    print(f"    score={r['score']:.4f} | {r['name']}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label}] index error: {exc}")


if __name__ == "__main__":
    check_coverage()
    check_indexes()
    test_vector_search("child labor risk in mining")
    test_vector_search("lithium suppliers for battery manufacturing")
