
from retrieval.entity_extractor import extract_entities
from retrieval.entity_resolver import resolve_entities_hybrid
from retrieval.query_router import route_query
from retrieval.cypher_templates import fetch_hybrid_subgraph, fetch_country_exposure
from retrieval.result_item import RetrieverResultItem
from retrieval.settings import Settings

class SupplyChainGraphRetriever:
    def __init__(self, driver, llm_client=None, settings=None):
        self.driver = driver
        self.llm = llm_client
        self.settings = settings or Settings()

    def _multi_hop_expand(self, seed_nodes, hop_count=2, max_paths_per_seed=8):
        expanded = {}
        cypher = f"""
        MATCH (n)
        WHERE n.name = $seed
        OPTIONAL MATCH p = (n)-[*1..{hop_count}]-(m)
        WITH collect(DISTINCT p)[..$max_paths] AS paths
        RETURN paths
        """
        for seed in seed_nodes:
            with self.driver.session(**({"database": self.settings.neo4j_database} if self.settings.neo4j_database else {})) as session:
                rec = session.run(cypher, seed=seed, max_paths=max_paths_per_seed).single()
                if rec and rec["paths"]:
                    expanded[seed] = rec["paths"]
        return expanded

    def retrieve(self, query: str, top_k: int = 6, hop_count: int = 2) -> dict:
        raw_entities = extract_entities(query, self.llm)
        resolved = resolve_entities_hybrid(raw_entities, self.driver, self.settings)
        route = route_query(query, resolved, self.driver, self.llm)
        all_names = []
        for names_list in resolved.values():
            all_names.extend([n for n in names_list if n])
        if route == "EXPOSURE_ANALYSIS":
            seeds = fetch_country_exposure(all_names, self.driver)
        else:
            seeds = fetch_hybrid_subgraph(all_names, query, self.driver)
        expanded = self._multi_hop_expand([s["anchor_name"] if "anchor_name" in s else s["path_summary"][0] for s in seeds[:top_k]], hop_count=hop_count)
        results = []
        for s in seeds[:top_k]:
            name = s.get("anchor_name") or (s.get("path_summary") and s["path_summary"][0])
            content = f"Seed: {name}\nSummary: {s}"
            metadata = {"expanded_paths": expanded.get(name, []), **s}
            results.append(RetrieverResultItem(content=content, metadata=metadata))
        return {
            "query": query,
            "route": route,
            "raw_entities": raw_entities,
            "resolved_entities": resolved,
            "results": results
        }
