

import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Settings:
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: Optional[str] = os.getenv("NEO4J_DATABASE")
    node_label: str = os.getenv("KG_NODE_LABEL", "RetrieverNode")
    embedding_property: str = os.getenv("KG_EMBEDDING_PROPERTY", "embedding")
    vector_index_name: str = os.getenv("KG_VECTOR_INDEX", "entity_embedding_idx")
    fulltext_index_name: str = os.getenv("KG_FULLTEXT_INDEX", "entity_fulltext_idx")
    retrieval_text_property: str = os.getenv("KG_RETRIEVAL_TEXT_PROPERTY", "retrieval_text")
    retrieval_label_property: str = os.getenv("KG_RETRIEVAL_LABEL_PROPERTY", "retrieval_label")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    source_labels: List[str] = None
    text_properties: List[str] = None

    def __post_init__(self):
        if self.neo4j_database is not None and not self.neo4j_database.strip():
            self.neo4j_database = None
        if self.neo4j_database is None and self.neo4j_user.strip():
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
