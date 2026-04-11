"""Entity alias resolution for retrieval-time normalization."""

from __future__ import annotations

import logging
from pathlib import Path


# Free alternative: Use sentence-transformers (all-MiniLM-L6-v2)
from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    emb = _model.encode([text], normalize_embeddings=True)
    return emb[0].tolist() if isinstance(emb, np.ndarray) or hasattr(emb, 'tolist') else emb[0]

def resolve_entities_semantically(extracted: dict, driver):
    resolved = {}
    with driver.session() as session:
        for entity_type, names in extracted.items():
            resolved_names = []
            for name in names:
                vector = get_embedding(name)
                cypher = """
                CALL db.index.vector.queryNodes('company_embeddings', 1, $vector)
                YIELD node, score
                WHERE score > 0.85
                RETURN node.name as canonical_name
                """
                result = session.run(cypher, vector=vector).single()
                if result:
                    resolved_names.append(result['canonical_name'])
                else:
                    resolved_names.append(name)
            resolved[entity_type] = resolved_names
    return resolved