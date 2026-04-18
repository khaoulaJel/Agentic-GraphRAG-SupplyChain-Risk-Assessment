"""
seed_graph.py — Seed the Neo4j graph with supply chain data.

TODO: Populate with real data once ontology and data sources are finalized.

Run: python -m scripts.seed_graph
"""

from graph.builder import get_driver, run_cypher, close
from graph.schema import COUNTRY_RISK


# ── SEED DATA ────────────────────────────────────────────────────────────────
# TODO: Add your entities and relationships here after data gathering.

companies = []
materials = []
countries = []
facilities = []
supplies_to = []
sources_from = []
produces = []


# ── INGESTION ────────────────────────────────────────────────────────────────

def seed():
    if not companies:
        print("No seed data defined yet. Populate the lists above first.")
        return

    driver = get_driver()
    with driver.session() as s:
        for c in companies:
            s.run(
                "MERGE (n:Company {name: $name}) "
                "SET n += $props",
                name=c["name"], props=c,
            )
        # TODO: Add ingestion for materials, countries, facilities, edges

    # Print summary
    print("\n── Graph Summary ──────────────────")
    for row in run_cypher("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY label"):
        print(f"  {row['label']:<20} {row['cnt']} nodes")
    print()
    for row in run_cypher("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY rel"):
        print(f"  {row['rel']:<20} {row['cnt']} edges")


if __name__ == "__main__":
    seed()
    close()
