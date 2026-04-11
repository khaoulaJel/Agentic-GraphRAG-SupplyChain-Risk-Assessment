"""
Set up and run a Hybrid Retriever on an existing Neo4j graph (Aura-friendly).

What this script does:
1) Checks if FULLTEXT and VECTOR indexes already exist (to avoid Aura quota issues).
2) Creates FULLTEXT index on Entity(name, description, text) only when missing.
3) Uses Neo4jVector.from_existing_graph to embed existing nodes in place (no node duplication).
4) Initializes neo4j_graphrag.retrievers.HybridRetriever.
5) Exposes get_supply_chain_context(query, top_k, hop_count) for multi-hop context retrieval.

CPU note:
- Uses sentence-transformers/all-MiniLM-L6-v2, which is lightweight for 16GB RAM CPU machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from neo4j.graph import Node, Path

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever
from neo4j_graphrag.types import RetrieverResultItem


load_dotenv()


@dataclass
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://<your-aura-id>.databases.neo4j.io")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    # Keep unset/empty to let Aura use its default database.
    neo4j_database: Optional[str] = os.getenv("NEO4J_DATABASE")

    # Ontology-aware retrieval projection label.
    node_label: str = os.getenv("KG_NODE_LABEL", "RetrieverNode")

    embedding_property: str = os.getenv("KG_EMBEDDING_PROPERTY", "embedding")
    vector_index_name: str = os.getenv("KG_VECTOR_INDEX", "entity_embedding_idx")
    fulltext_index_name: str = os.getenv("KG_FULLTEXT_INDEX", "entity_fulltext_idx")

    retrieval_text_property: str = os.getenv("KG_RETRIEVAL_TEXT_PROPERTY", "retrieval_text")
    retrieval_label_property: str = os.getenv("KG_RETRIEVAL_LABEL_PROPERTY", "retrieval_label")

    # Ontology labels from this project.
    source_labels: List[str] = None  # type: ignore[assignment]

    # Name + retrieval text are used for fulltext and embedding source text.
    text_properties: List[str] = None  # type: ignore[assignment]

    # Lightweight local embedding model for CPU.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    def __post_init__(self) -> None:
        if self.neo4j_database is not None and not self.neo4j_database.strip():
            self.neo4j_database = None
        if self.neo4j_database is None and self.neo4j_user.strip():
            # In some Aura setups, database name matches the Neo4j username.
            self.neo4j_database = self.neo4j_user
        if self.source_labels is None:
            labels_raw = os.getenv(
                "KG_SOURCE_LABELS",
                "Company,Material,Country,Facility,RegulatoryBody,RiskEvent,Location,Product,Organization,Regulation,Entity",
            )
            self.source_labels = [x.strip() for x in labels_raw.split(",") if x.strip()]
        if self.text_properties is None:
            self.text_properties = [
                "name",
                self.retrieval_text_property,
                self.retrieval_label_property,
                "entity_type",
                "source",
            ]


def _session_kwargs(database: Optional[str]) -> Dict[str, str]:
    if database:
        return {"database": database}
    return {}


def _index_exists(driver, index_name: str, database: Optional[str]) -> bool:
    query = """
    SHOW INDEXES
    YIELD name
    WHERE name = $index_name
    RETURN count(*) > 0 AS exists
    """
    with driver.session(**_session_kwargs(database)) as session:
        rec = session.run(query, index_name=index_name).single()
        return bool(rec and rec["exists"])


def _create_fulltext_index_if_missing(
    driver,
    settings: Settings,
) -> None:
    if _index_exists(driver, settings.fulltext_index_name, settings.neo4j_database):
        print(f"FULLTEXT index already exists: {settings.fulltext_index_name}")
        return

    # Required Cypher command (kept explicit for visibility/reuse).
    cypher = f"""
    CREATE FULLTEXT INDEX {settings.fulltext_index_name}
    IF NOT EXISTS
    FOR (n:{settings.node_label})
    ON EACH [n.name, n.{settings.retrieval_text_property}]
    """

    with driver.session(**_session_kwargs(settings.neo4j_database)) as session:
        session.run(cypher)
    print(f"Created FULLTEXT index: {settings.fulltext_index_name}")


def _validate_connection(driver, settings: Settings) -> None:
    query = "RETURN 1 AS ok"
    try:
        with driver.session(**_session_kwargs(settings.neo4j_database)) as session:
            session.run(query).single()
    except ClientError as e:
        msg = str(e)
        if "DatabaseNotFound" in msg:
            raise RuntimeError(
                "Neo4j database name is invalid. Set NEO4J_DATABASE to your Aura database "
                "name. If your setup uses the same value as NEO4J_USER, you can set "
                "NEO4J_DATABASE to NEO4J_USER."
            ) from e
        raise


def _prepare_ontology_projection(driver, settings: Settings) -> None:
    # Build a normalized retrieval text from ontology properties so retrieval does not depend
    # on non-existent description/text fields.
    properties_for_text = [
        "name",
        "type",
        "country",
        "tier",
        "category",
        "risk_tier",
        "location",
        "jurisdiction",
        "date",
        "severity",
        "entity_type",
        "source",
    ]

    labels_query = "CALL db.labels() YIELD label RETURN collect(label) AS labels"

    with driver.session(**_session_kwargs(settings.neo4j_database)) as session:
        labels_rec = session.run(labels_query).single()
        existing_labels = set(labels_rec["labels"] if labels_rec else [])

        active_source_labels = [lbl for lbl in settings.source_labels if lbl in existing_labels]

        # If specific domain labels exist, ignore generic Entity to reduce noisy seeds.
        if "Entity" in active_source_labels and len(active_source_labels) > 1:
            active_source_labels = [lbl for lbl in active_source_labels if lbl != "Entity"]

        if not active_source_labels:
            raise RuntimeError(
                "No configured source labels found in DB. Set KG_SOURCE_LABELS to labels that "
                "exist in your graph."
            )

        # Clear stale projection from previous runs before rebuilding it.
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

        projected = session.run(
            query,
            source_labels=active_source_labels,
            prop_names=properties_for_text,
        ).single()

    count_projected = int(projected["projected"]) if projected else 0
    print(
        f"Prepared ontology projection on label '{settings.node_label}' for "
        f"{count_projected} node(s)."
    )


def _format_retriever_item(record) -> RetrieverResultItem:
    node = record.get("node")
    score = record.get("score")

    if isinstance(node, dict):
        props = dict(node)
        element_id = (
            node.get("element_id")
            or node.get("elementId")
            or node.get("id")
        )
        labels = node.get("labels", [])
        if isinstance(labels, str):
            labels = [labels]
    elif node is not None:
        props = dict(node.items())
        element_id = getattr(node, "element_id", None)
        labels = list(getattr(node, "labels", []))
    else:
        props = {}
        element_id = None
        labels = []

    name = props.get("name", "")
    summary = props.get("retrieval_text") or name

    if not labels:
        lbl = props.get("retrieval_label")
        if isinstance(lbl, str) and lbl:
            labels = [lbl]

    metadata = {
        "score": score,
        "element_id": element_id,
        "labels": labels,
        "name": name,
    }

    content = f"labels: {labels}\nname: {name}\nsummary: {summary}".strip()
    return RetrieverResultItem(content=content, metadata=metadata)


class HybridSupplyChainRetriever:
    def __init__(self, settings: Settings):
        if not settings.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is empty. Add it to your environment/.env file.")

        self.settings = settings
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        _validate_connection(self.driver, settings)

        # LangChain-compatible embedder needed by Neo4jVector.from_existing_graph.
        self.lc_embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)

        # neo4j-graphrag embedder used by HybridRetriever.
        self.embedder = SentenceTransformerEmbeddings(model=settings.embedding_model)

        # 0) Project ontology labels/properties to retrieval label/text.
        _prepare_ontology_projection(self.driver, settings)

        # 1) Create/check fulltext index first (Aura-safe).
        _create_fulltext_index_if_missing(self.driver, settings)

        # 2) Build/update vector index and embeddings from existing nodes.
        #    This avoids re-inserting documents and updates in-place embedding property.
        self.vector_store = Neo4jVector.from_existing_graph(
            embedding=self.lc_embedder,
            url=settings.neo4j_uri,
            username=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            index_name=settings.vector_index_name,
            node_label=settings.node_label,
            text_node_properties=settings.text_properties,
            embedding_node_property=settings.embedding_property,
            search_type="hybrid",
            keyword_index_name=settings.fulltext_index_name,
        )

        if _index_exists(self.driver, settings.vector_index_name, settings.neo4j_database):
            print(f"VECTOR index ready: {settings.vector_index_name}")
        else:
            print(
                "Warning: vector index was not detected after from_existing_graph call. "
                "Check label/property names."
            )

        # 3) HybridRetriever setup (vector + fulltext).
        self.retriever = HybridRetriever(
            driver=self.driver,
            vector_index_name=settings.vector_index_name,
            fulltext_index_name=settings.fulltext_index_name,
            embedder=self.embedder,
            return_properties=settings.text_properties,
            result_formatter=_format_retriever_item,
            neo4j_database=settings.neo4j_database,
        )

    def close(self) -> None:
        self.driver.close()

    @staticmethod
    def _path_to_text(path: Path) -> str:
        node_names: List[str] = []
        rel_types: List[str] = []

        for n in path.nodes:
            label_text = ":".join(sorted(n.labels))
            name = n.get("name") or n.get("id") or n.element_id
            node_names.append(f"({label_text} {name})")

        for r in path.relationships:
            rel_types.append(type(r).__name__)

        if not rel_types:
            return node_names[0] if node_names else ""

        segments: List[str] = []
        for i, rel in enumerate(rel_types):
            segments.append(node_names[i])
            segments.append(f"-[{rel}]-")
        segments.append(node_names[-1])
        return "".join(segments)

    def _fetch_multi_hop_paths(
        self,
        element_id: str,
        hop_count: int,
        max_paths_per_seed: int,
    ) -> List[str]:
        safe_hops = max(1, min(hop_count, 4))

        cypher = f"""
        MATCH (n)
        WHERE elementId(n) = $element_id
        OPTIONAL MATCH p = (n)-[*1..{safe_hops}]-(m)
        WITH collect(DISTINCT p)[..$max_paths] AS paths
        RETURN paths
        """

        with self.driver.session(**_session_kwargs(self.settings.neo4j_database)) as session:
            rec = session.run(
                cypher,
                element_id=element_id,
                max_paths=max_paths_per_seed,
            ).single()

        if not rec:
            return []

        paths: List[Path] = [p for p in rec.get("paths", []) if p is not None]
        return [self._path_to_text(p) for p in paths]

    def get_supply_chain_context(
        self,
        query: str,
        top_k: int = 6,
        hop_count: int = 2,
        max_paths_per_seed: int = 8,
    ) -> str:
        """
        Retrieve hybrid search context and expand each seed with multi-hop graph paths.

        Performance tuning:
        - top_k: lower values (3-6) are faster and cheaper; higher values improve recall.
        - hop_count: 1-2 is usually enough for latency-sensitive use; 3-4 can explode path count.
        - max_paths_per_seed: hard cap to keep responses bounded on Aura.
        """
        if not query.strip():
            return ""

        top_k = max(1, min(top_k, 20))
        hop_count = max(1, min(hop_count, 4))

        result = self.retriever.search(query_text=query, top_k=top_k)
        items = result.items if result else []

        lines: List[str] = []
        for i, item in enumerate(items, start=1):
            md: Dict[str, Any] = item.metadata or {}
            eid = md.get("element_id")
            score = md.get("score")
            seed_text = item.content or ""

            lines.append(f"Seed {i} (score={score}):")
            lines.append(seed_text)

            if eid:
                paths = self._fetch_multi_hop_paths(
                    element_id=eid,
                    hop_count=hop_count,
                    max_paths_per_seed=max_paths_per_seed,
                )
                if paths:
                    lines.append("Related multi-hop paths:")
                    lines.extend([f"- {p}" for p in paths])
            lines.append("")

        return "\n".join(lines).strip()


def get_supply_chain_context(query: str, top_k: int = 6, hop_count: int = 2) -> str:
    settings = Settings()
    retriever = HybridSupplyChainRetriever(settings)
    try:
        return retriever.get_supply_chain_context(
            query=query,
            top_k=top_k,
            hop_count=hop_count,
        )
    finally:
        retriever.close()


def print_required_cypher(settings: Settings) -> None:
    print("\nCypher for FULLTEXT index (manual/reference):")
    print(
        f"""
CREATE FULLTEXT INDEX {settings.fulltext_index_name}
IF NOT EXISTS
FOR (n:{settings.node_label})
ON EACH [n.name, n.{settings.retrieval_text_property}]
""".strip()
    )


def main() -> None:
    settings = Settings()
    print_required_cypher(settings)

    retriever = HybridSupplyChainRetriever(settings)
    try:
        sample_query = "What products are influenced by disruption of cobalt supply"
        context = retriever.get_supply_chain_context(
            sample_query,
            top_k=6,   # Decrease for speed, increase for recall.
            hop_count=2,  # 1-2 recommended for fast CPU/Aura response.
        )
        print("\n=== Retrieved Context ===")
        print(context)
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
