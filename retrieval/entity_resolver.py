
from typing import Any
from retrieval.settings import Settings
from retrieval.result_item import RetrieverResultItem

def resolve_entities(extracted: dict[str, list[str]], driver: Any, settings: Settings = None) -> dict[str, list[str]]:
    return resolve_entities_hybrid(extracted, driver, settings)

def resolve_entities_hybrid(
    extracted: dict[str, list[str]], driver: Any, settings: Settings = None
) -> dict[str, list[str]]:
    if settings is None:
        from retrieval.settings import Settings as _Settings
        settings = _Settings()
    resolved: dict[str, list[str]] = {}
    with driver.session() as session:
        for entity_type, names in extracted.items():
            resolved_names: list[str] = []
            for name in names:
                if not name:
                    continue
                cypher_fulltext = f"""
                CALL db.index.fulltext.queryNodes('{settings.fulltext_index_name}', $name) YIELD node, score
                RETURN node.name AS canonical_name, score, labels(node) AS node_labels
                ORDER BY score DESC LIMIT 3
                """
                ft_results = list(session.run(cypher_fulltext, name=name))
                cypher_vector = f"""
                CALL db.index.vector.queryNodes('{settings.vector_index_name}', 5, $vector)
                YIELD node, score
                RETURN node.name AS canonical_name, score, labels(node) AS node_labels
                """
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(settings.embedding_model)
                vector = model.encode([name], normalize_embeddings=True)[0].tolist()
                v_results = list(session.run(cypher_vector, vector=vector))
                candidates = {}
                for res in ft_results:
                    cname = res["canonical_name"]
                    candidates[cname] = {"score": res["score"], "labels": res["node_labels"], "source": "fulltext"}
                for res in v_results:
                    cname = res["canonical_name"]
                    if cname in candidates:
                        candidates[cname]["score"] = max(candidates[cname]["score"], res["score"])
                        candidates[cname]["source"] = "hybrid"
                    else:
                        candidates[cname] = {"score": res["score"], "labels": res["node_labels"], "source": "vector"}
                best = None
                best_score = 0.0
                for cname, meta in candidates.items():
                    label_match = not meta["labels"] or entity_type.lower() in [lbl.lower() for lbl in meta["labels"]]
                    if meta["score"] > best_score and label_match:
                        best_score = meta["score"]
                        best = cname
                if best:
                    resolved_names.append(best)
                else:
                    resolved_names.append(name)
            resolved[entity_type] = resolved_names
    return resolved