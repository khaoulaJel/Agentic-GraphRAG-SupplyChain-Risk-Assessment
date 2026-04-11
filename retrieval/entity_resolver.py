
from __future__ import annotations
def _get_alias_logger():
    import logging
    return logging.getLogger("alias")
"""Entity alias resolution for retrieval-time normalization."""

import logging
from pathlib import Path


# Free alternative: Use sentence-transformers (all-MiniLM-L6-v2)

# Hybrid vector + fuzzy + type-aware entity resolver (2026 best practice)
from sentence_transformers import SentenceTransformer
import numpy as np
from thefuzz import fuzz  # pip install thefuzz

_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    emb = _model.encode([text], normalize_embeddings=True)
    return emb[0].tolist() if isinstance(emb, np.ndarray) or hasattr(emb, 'tolist') else emb[0]

# Alias for compatibility with test and __init__.py
def resolve_entities(extracted: dict, driver):
    return resolve_entities_semantically(extracted, driver)

def resolve_entities_semantically(extracted: dict, driver):
    resolved = {}
    with driver.session() as session:
        for entity_type, names in extracted.items():
            resolved_names = []
            for name in names:
                # 1. Vector lookup (top 5)
                vector = get_embedding(name)
                cypher = """
                CALL db.index.vector.queryNodes('company_embeddings', 5, $vector)
                YIELD node, score
                WHERE score > 0.78
                RETURN node.name as canonical_name, score, labels(node) as node_labels
                """
                results = list(session.run(cypher, vector=vector))

                # 2. Fuzzy fallback and type-aware
                best = None
                best_score = 0
                for res in results:
                    fuzzy = fuzz.token_sort_ratio(name.lower(), res["canonical_name"].lower())
                    combined = (res["score"] * 0.7) + (fuzzy / 100 * 0.3)
                    # Type-aware: prefer same label
                    label_match = not res["node_labels"] or entity_type.lower() in [l.lower() for l in res["node_labels"]]
                    if combined > best_score and label_match:
                        best_score = combined
                        best = res["canonical_name"]

                if best:
                    resolved_names.append(best)
                else:
                    # Full-text or exact match fallback
                    cypher_fallback = """
                    MATCH (n) WHERE toLower(n.name) = toLower($name) OR toLower(n.name) CONTAINS toLower($name)
                    RETURN n.name AS canonical_name LIMIT 1
                    """
                    fallback_res = session.run(cypher_fallback, name=name).single()
                    if fallback_res:
                        resolved_names.append(fallback_res['canonical_name'])
                    else:
                        resolved_names.append(name)
            resolved[entity_type] = resolved_names
    return resolved