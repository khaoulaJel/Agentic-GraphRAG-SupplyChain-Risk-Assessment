from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Existing vector index name in Neo4j. Change via env if needed.
VECTOR_INDEX_NAME = os.getenv("NEO4J_VECTOR_INDEX", "embedding")

# Gemini models requested in prompt.
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIMENSIONS = int(os.getenv("GEMINI_EMBED_DIMENSIONS", "3072"))

VECTOR_INDEXES = {
    "RiskEvent": "riskevent_embedding_idx",
    "Company": "company_embedding_idx",
    "Material": "material_embedding_idx",
    "Entity": "entity_embedding_idx",
    "Location": "location_embedding_idx",
    "Country": "country_embedding_idx",
    "Regulation": "regulation_embedding_idx",
    "Product": "product_embedding_idx",
    "Organization": "organization_embedding_idx",
    "Classification": "classification_embedding_idx",
}


def session_kwargs() -> dict[str, str]:
    if NEO4J_DATABASE and NEO4J_DATABASE.strip():
        return {"database": NEO4J_DATABASE}
    return {}


def validate_config() -> None:
    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD is required.")
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in .env.")
