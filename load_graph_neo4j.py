"""
Load typed supply-chain triples into Neo4j with high-performance batching.

Key behavior:
- Loads from both processed JSONL files:
  - data/processed/triples_silver_layer.jsonl
  - data/processed/triples_silver_layer_extracted.jsonl
- Uses explicit subject/object types from triple payload (no text heuristics)
- Creates typed relationship labels from predicate values
- Deduplicates by (subject, predicate, object, source)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

TRIPLES_FILES = [
    Path("./data/processed/triples_silver_layer.jsonl"),
    Path("./data/processed/triples_silver_layer_extracted.jsonl"),
]

BATCH_SIZE = 500


TYPE_TO_LABEL = {
    "COMPANY": "Company",
    "LOCATION": "Location",
    "MATERIAL": "Material",
    "RISK": "RiskEvent",
    "RISK_EVENT": "RiskEvent",
    "RISKEVENT": "RiskEvent",
    "RISK/EVENT": "RiskEvent",
    "PRODUCT": "Product",
    "PRODUCT_SERVICE": "Product",
    "PRODUCT/SERVICE": "Product",
    "ORGANIZATION": "Organization",
    "ORGANIZATION_BODY": "Organization",
    "ORGANIZATION/BODY": "Organization",
    "REGULATION": "Regulation",
    "REGULATION_STANDARD": "Regulation",
    "REGULATION/STANDARD": "Regulation",
    "ENTITY": "Entity",
}


class TripleLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=30)
        print(f"Connected to Neo4j at {uri}")

    def close(self) -> None:
        self.driver.close()
        print("Closed Neo4j connection")

    @staticmethod
    def _sanitize_properties(properties: Any) -> Dict[str, Any]:
        if not properties or not isinstance(properties, dict):
            return {}

        clean: Dict[str, Any] = {}
        for key, value in properties.items():
            if value is None:
                clean[key] = None
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                if all(isinstance(item, (str, int, float, bool, type(None))) for item in value):
                    clean[key] = value
                else:
                    clean[key] = json.dumps(value)
            elif isinstance(value, dict):
                clean[key] = json.dumps(value)
            else:
                clean[key] = str(value)

        return clean

    @staticmethod
    def _safe_identifier(value: str, default_value: str) -> str:
        if not value:
            return default_value
        out = re.sub(r"[^A-Za-z0-9_]", "_", value.upper())
        out = re.sub(r"_+", "_", out).strip("_")
        if not out:
            return default_value
        if not out[0].isalpha():
            out = f"R_{out}"
        return out

    @classmethod
    def _safe_relationship_type(cls, predicate: str) -> str:
        return cls._safe_identifier(predicate, "RELATED_TO")

    @classmethod
    def _normalize_type_to_label(cls, raw_type: Any) -> str:
        if raw_type is None:
            return "Entity"

        raw = str(raw_type).strip().upper()
        raw = raw.replace("-", "_")
        raw = re.sub(r"\s+", "_", raw)

        if raw in TYPE_TO_LABEL:
            return TYPE_TO_LABEL[raw]

        # If unknown type exists in data, keep it as a stable, safe custom label.
        # This still avoids heuristics and preserves source-provided taxonomy.
        safe = cls._safe_identifier(raw, "ENTITY")
        if safe == "ENTITY":
            return "Entity"
        return safe.title().replace("_", "")

    def setup_database(self, labels: List[str]) -> None:
        print("\nSetting up database constraints...")

        with self.driver.session() as session:
            for label in sorted(set(labels)):
                constraint_name = f"{label.lower()}_name_unique"
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label})
                REQUIRE n.name IS UNIQUE
                """
                try:
                    session.run(query)
                    print(f"  Constraint ready: {label}.name")
                except Exception as e:
                    error_text = str(e)
                    if "IndexAlreadyExists" in error_text or "already exists an index" in error_text:
                        # Aura can have pre-existing non-unique indexes that block
                        # creating a uniqueness constraint with IF NOT EXISTS.
                        # Keep loading and ensure at least an index exists.
                        try:
                            idx_name = f"{label.lower()}_name_idx"
                            session.run(
                                f"""
                                CREATE INDEX {idx_name} IF NOT EXISTS
                                FOR (n:{label})
                                ON (n.name)
                                """
                            )
                            print(f"  Index kept for {label}.name (constraint skipped due to existing index)")
                        except Exception as idx_error:
                            print(f"  Warning: could not prepare schema for {label}.name: {idx_error}")
                    else:
                        raise

    def _prepare_triples_from_file(self, file_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
        prepared: List[Dict[str, Any]] = []
        labels_used: List[str] = []
        seen: set = set()

        if not file_path.exists():
            print(f"File not found: {file_path}")
            return prepared, labels_used

        print(f"\nLoading {file_path.name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    triple = json.loads(line)
                except json.JSONDecodeError:
                    continue

                subject = str(triple.get("subject", "")).strip()
                predicate = str(triple.get("predicate", "")).strip()
                obj = str(triple.get("object", "")).strip()
                source = str(triple.get("source", "extraction")).strip() or "extraction"

                if not subject or not predicate or not obj:
                    continue

                props = triple.get("properties", {})
                clean_props = self._sanitize_properties(props)

                subject_type_raw = clean_props.get("subject_type", triple.get("subject_type", "ENTITY"))
                object_type_raw = clean_props.get("object_type", triple.get("object_type", "ENTITY"))

                subject_label = self._normalize_type_to_label(subject_type_raw)
                object_label = self._normalize_type_to_label(object_type_raw)
                rel_type = self._safe_relationship_type(predicate)

                dedupe_key = (subject, predicate, obj, source)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                prepared.append(
                    {
                        "subject": subject,
                        "subject_label": subject_label,
                        "subject_type_raw": str(subject_type_raw),
                        "predicate": predicate,
                        "rel_type": rel_type,
                        "object": obj,
                        "object_label": object_label,
                        "object_type_raw": str(object_type_raw),
                        "source": source,
                        "evidence": str(triple.get("evidence", "")),
                        "properties_json": json.dumps(clean_props) if clean_props else "{}",
                    }
                )

                labels_used.append(subject_label)
                labels_used.append(object_label)

        print(f"  Prepared {len(prepared)} triples from {file_path.name}")
        return prepared, labels_used

    def _ingest_batch(self, batch: List[Dict[str, Any]]) -> int:
        if not batch:
            return 0

        grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for row in batch:
            key = (row["subject_label"], row["object_label"], row["rel_type"])
            grouped.setdefault(key, []).append(row)

        total = 0

        with self.driver.session() as session:
            for (subject_label, object_label, rel_type), group_rows in grouped.items():
                query = f"""
                UNWIND $triples AS t
                MERGE (s:{subject_label} {{name: t.subject}})
                ON CREATE SET s.created = timestamp(), s.source = t.source
                SET s.updated = timestamp(), s.entity_type = t.subject_type_raw

                WITH s, t
                MERGE (o:{object_label} {{name: t.object}})
                ON CREATE SET o.created = timestamp(), o.source = t.source
                SET o.updated = timestamp(), o.entity_type = t.object_type_raw

                WITH s, o, t
                MERGE (s)-[r:{rel_type} {{source: t.source}}]->(o)
                ON CREATE SET r.created = timestamp()
                SET r.updated = timestamp(),
                    r.predicate = t.predicate,
                    r.evidence = t.evidence,
                    r.properties_json = t.properties_json
                RETURN count(r) as c
                """

                result = session.run(query, triples=group_rows)
                rec = result.single()
                total += int(rec["c"]) if rec and rec["c"] is not None else 0

        return total

    def load_all(self, file_paths: List[Path]) -> int:
        all_rows: List[Dict[str, Any]] = []
        labels_used: List[str] = []

        for file_path in file_paths:
            rows, labels = self._prepare_triples_from_file(file_path)
            all_rows.extend(rows)
            labels_used.extend(labels)

        if not all_rows:
            print("No valid triples found in input files.")
            return 0

        self.setup_database(labels_used)

        print(f"\nTotal prepared triples: {len(all_rows)}")
        num_batches = (len(all_rows) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Ingesting in {num_batches} batches of {BATCH_SIZE}...")

        total_loaded = 0
        for i in tqdm(range(0, len(all_rows), BATCH_SIZE), desc="Batches", unit="batch"):
            batch = all_rows[i:i + BATCH_SIZE]
            total_loaded += self._ingest_batch(batch)

        return total_loaded

    def print_graph_stats(self) -> None:
        with self.driver.session() as session:
            print("\n" + "=" * 60)
            print("NODE STATISTICS")
            node_q = """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, count(*) AS c
            ORDER BY c DESC
            """
            for record in session.run(node_q):
                print(f"  {record['label']}: {record['c']}")

            print("\nRELATIONSHIP STATISTICS")
            rel_q = """
            MATCH ()-[r]->()
            RETURN type(r) AS rel, count(*) AS c
            ORDER BY c DESC
            """
            for record in session.run(rel_q):
                print(f"  {record['rel']}: {record['c']}")

            total_q = """
            MATCH (n) WITH count(n) AS nodes
            MATCH ()-[r]->() RETURN nodes, count(r) AS rels
            """
            total = session.run(total_q).single()
            print("\nTOTAL")
            print(f"  Nodes: {total['nodes']}")
            print(f"  Relationships: {total['rels']}")



def main() -> None:
    if not NEO4J_PASSWORD:
        print("Error: NEO4J_PASSWORD not found in environment variables")
        return

    loader = TripleLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    start = time.time()

    try:
        total_loaded = loader.load_all(TRIPLES_FILES)
        elapsed = time.time() - start

        loader.print_graph_stats()

        rate = (total_loaded / elapsed) if elapsed > 0 else 0.0
        print("\n" + "=" * 60)
        print(f"Loaded triples: {total_loaded}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Rate: {rate:.0f} triples/sec")
        print("Graph loading complete")

    finally:
        loader.close()


if __name__ == "__main__":
    main()
