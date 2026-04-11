from retrieval.entity_extractor import extract_entities
from retrieval.entity_resolver import resolve_entities_semantically
from retrieval.cypher_templates import fetch_hybrid_subgraph, fetch_country_exposure
from retrieval.query_router import route_query

class SupplyChainGraphRetriever:
    def __init__(self, driver, llm_client=None):
        self.driver = driver
        self.llm = llm_client
    
    def retrieve(self, query: str) -> dict:
        # Step 1: Extract entities
        raw_entities = extract_entities(query, self.llm)
        
        # Step 2: Resolve to canonical names
        resolved = resolve_entities_semantically(raw_entities, self.driver)
        
        # Step 3: Route decision
        route = route_query(query, self.llm)
        
        # Flatten all resolved entity names
        all_names = []
        for names_list in resolved.values():
            all_names.extend([n for n in names_list if n])
        
        # Step 4: Fetch subgraph based on route
        if route == "EXPOSURE_ANALYSIS":
            subgraph = fetch_country_exposure(all_names, self.driver)
        else:
            subgraph = fetch_hybrid_subgraph(all_names, query, self.driver)
        
        return {
            "query": query,
            "route": route,
            "raw_entities": raw_entities,
            "resolved_entities": resolved,
            "subgraph": subgraph[:120]  # safety cap
        }
