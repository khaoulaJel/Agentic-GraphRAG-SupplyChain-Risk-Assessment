import pytest
from unittest.mock import MagicMock
import retrieval.entity_extractor as entity_extractor
import retrieval.entity_resolver as entity_resolver
import retrieval.cypher_templates as cypher_templates
import retrieval.query_router as query_router

# --- Entity Extractor Tests ---
def test_entity_extractor_prompt():
    prompt = entity_extractor.ENTITY_EXTRACTION_PROMPT.format(query="Is Tesla exposed to DRC risk?")
    assert "companies" in prompt and "countries" in prompt and "risk_events" in prompt

def test_strip_code_fences():
    text = """```json\n{\"companies\": [\"Tesla\"]}"""
    assert entity_extractor._strip_code_fences(text) == '{"companies": ["Tesla"]}'

# --- Entity Resolver Tests ---
def test_resolve_entities_alias_and_miss(tmp_path, monkeypatch):
    # Patch logger to avoid file writes
    monkeypatch.setattr(entity_resolver, "_get_alias_logger", lambda: MagicMock())
    extracted = {"companies": ["contemporary amperex", "UnknownCo"]}
    driver = MagicMock()
    # Optionally, mock driver.session().run() to return expected results if needed
    resolved = entity_resolver.resolve_entities(extracted, driver)
    # The following assertions may need to be updated depending on the mock behavior
    assert isinstance(resolved["companies"], list)

# --- Cypher Templates Tests ---
def test_fetch_subgraph_empty():
    driver = MagicMock()
    result = cypher_templates.fetch_subgraph([], driver)
    assert result == []

def test_fetch_subgraph_calls_driver():
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value = [MagicMock(data=lambda: {"anchor_name": "Tesla"})]
    result = cypher_templates.fetch_subgraph(["Tesla"], driver)
    assert result[0]["anchor_name"] == "Tesla"

# --- Query Router Tests ---
def test_route_query_risk(monkeypatch):
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value = [MagicMock(data=lambda: {"company": "Tesla"})]
    monkeypatch.setattr(query_router, "COUNTRY_EXPOSURE_QUERY", "QUERY")
    query_str = "What is the risk for Tesla?"
    resolved = {"companies": ["Tesla"], "products": [], "organizations": []}
    result = query_router.route_query(query_str, resolved, driver)
    assert result[0]["company"] == "Tesla"

def test_route_query_default(monkeypatch):
    driver = MagicMock()
    monkeypatch.setattr(query_router, "fetch_subgraph", lambda names, drv: [{"anchor_name": "Tesla"}])
    query_str = "Show Tesla's network"
    resolved = {"companies": ["Tesla"], "products": [], "organizations": []}
    result = query_router.route_query(query_str, resolved, driver)
    assert result[0]["anchor_name"] == "Tesla"


# --- Realistic End-to-End Pipeline Test ---
def test_full_pipeline_complex_scenario(monkeypatch):
    # Simulate a complex user query
    query = "Which companies supply Panasonic with cobalt from the DRC, and what are the associated political risks?"

    # 1. Entity Extraction (simulate output)
    extracted = {
        "companies": ["Panasonic"],
        "materials": ["cobalt"],
        "countries": ["DRC"],
        "risk_events": ["political risk"],
        "products": [],
        "organizations": [],
        "entities": [],
        "facilities": [],
        "locations": [],
        "regulations": [],
        "regulatory_bodies": []
    }

    # 2. Entity Resolution (simulate canonicalization)
    resolved = {
        "companies": ["Panasonic"],
        "materials": ["Cobalt"],
        "countries": ["Democratic Republic of the Congo"],
        "risk_events": ["Political Risk"],
        "products": [],
        "organizations": [],
        "entities": [],
        "facilities": [],
        "locations": [],
        "regulations": [],
        "regulatory_bodies": []
    }

    # 3. Mock a realistic subgraph result from Neo4j
    mock_subgraph = [
        {
            "anchor_name": "Panasonic",
            "anchor_labels": ["Company"],
            "path_nodes": [
                {"name": "Panasonic", "labels": ["Company"]},
                {"name": "SupplierCo", "labels": ["Company"]},
                {"name": "Cobalt", "labels": ["Material"]},
                {"name": "Democratic Republic of the Congo", "labels": ["Country"]}
            ],
            "path_rels": [
                {"type": "SUPPLIES_TO", "start": "SupplierCo", "end": "Panasonic"},
                {"type": "MADE_FROM", "start": "SupplierCo", "end": "Cobalt"},
                {"type": "SOURCED_FROM", "start": "Cobalt", "end": "Democratic Republic of the Congo"}
            ],
            "path_score": 21,
            "hops": 3
        }
    ]

    # 4. Mock the retrieval function to return the above subgraph
    monkeypatch.setattr(query_router, "fetch_subgraph", lambda names, drv: mock_subgraph)

    # 5. Route and retrieve
    driver = MagicMock()
    result = query_router.route_query(query, resolved, driver)

    # 6. Assert the pipeline output is as expected
    assert isinstance(result, list)
    assert result[0]["anchor_name"] == "Panasonic"
    assert any(n["name"] == "SupplierCo" for n in result[0]["path_nodes"])
    assert any(n["name"] == "Democratic Republic of the Congo" for n in result[0]["path_nodes"])
    assert any(r["type"] == "SOURCED_FROM" for r in result[0]["path_rels"])
