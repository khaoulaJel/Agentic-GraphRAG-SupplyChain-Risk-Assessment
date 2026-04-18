"""
run_schema.py — Apply Neo4j schema constraints and indexes.
Run: python -m scripts.run_schema
"""

from graph.builder import get_driver, close

with open("schema.cypher", "r") as f:
    statements = [s.strip() for s in f.read().split(";") if s.strip()]

with get_driver().session() as session:
    for stmt in statements:
        try:
            session.run(stmt)
            print(f"OK: {stmt[:70]}...")
        except Exception as e:
            print(f"ERR: {e}")

close()
print("\nSchema applied.")
