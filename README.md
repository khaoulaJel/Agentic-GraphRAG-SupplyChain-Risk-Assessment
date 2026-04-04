# Agentic GraphRAG for Supply Chain Risk Assessment

EV battery supply chain risk assessment using Neo4j knowledge graph + LangGraph agent.

## Setup

1. **Neo4j Aura**: Create free account at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)
2. **API Keys**: Get OpenAI and Tavily API keys
3. **Environment**:
   ```bash
   cp .env.example .env
   # Fill in your credentials in .env
   conda env create -f environment.yml
   conda activate graphrag-ev
   ```

## Usage

```bash
# 1. Apply schema constraints
python run_schema.py

# 2. Seed the graph with known EV battery supply chain data
python seed_graph.py

# 3. (Optional) Ingest additional documents via LLM extraction
python ingestion/ingest_documents.py

# 4. Run sample queries
python main.py
```

## Project Structure

```
graph/
  schema.py          — Ontology, aliases, risk tiers, extraction prompt
  builder.py         — Neo4j connection and triple ingestion helpers
ingestion/
  ingest_documents.py — LLM-based document extraction pipeline
reasoning/
  traversal.py       — Graph traversal (supplier lookup, subgraph extraction)
  path_extraction.py — Sourcing path analysis
  impact.py          — Risk scoring (concentration, path count, geopolitical)
seed_graph.py        — Hand-crafted seed data for Tesla/CATL/Panasonic/LG
schema.cypher        — Neo4j constraints and indexes
main.py              — Entry point with sample queries
ontology.md          — Ontology documentation
```

## Focal Scope

- **Companies**: Tesla, CATL, Panasonic, LG Energy Solution
- **Materials**: Lithium, Cobalt, Nickel
- **Geographies**: DRC, Chile, Australia, China
