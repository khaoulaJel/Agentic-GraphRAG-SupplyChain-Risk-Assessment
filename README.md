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

## GraphRAG

Use this flow when working with the modular GraphRAG pipeline in the `graphrag/` package.

1. Prepare your `.env` with Neo4j and Gemini keys: (already done hh)
  - `NEO4J_URI`
  - `NEO4J_USERNAME`
  - `NEO4J_PASSWORD`
  - `GOOGLE_API_KEY`
2. Build or refresh embeddings and vector indexes: (skipi tahadi)
  ```bash
  python -m graphrag.embed_and_index
  ```
3. Verify coverage and index health:
  ```bash
  python -m graphrag.verify_embeddings
  ```
4. Run sample hybrid GraphRAG queries:
  ```bash
  python -m graphrag.main
  ```

Notes:
- `graphrag/main.py` contains example questions you can edit quickly.
- If results look stale after data changes, rerun `embed_and_index`.

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

## GraphRAG Sample Results

The following examples summarize what teammates can expect when running GraphRAG on the current EV supply chain graph.

### Query 1: What risk events are associated with Tesla's supply chain?

Result highlights:
- Identified human-rights related risks, including forced labor and child labor exposure in upstream sourcing.
- Surfaced commodity and market risks such as lithium oversupply and cobalt substitution dynamics.
- Retrieved geopolitical and policy pressure points, including export controls and subsidy-related uncertainty.
- Connected supplier-level risks (for example CATL-linked acute and reputation risks) with broader operational disruption themes.

Primary supporting sources:
- tesla_impact_2023
- usgs_lithium_2024
- usgs_cobalt_2024
- catl_sustainability_2023
- ita_semiconductor_2023

### Query 2: Which companies supply materials to Tesla?

Result highlights:
- Returned major battery suppliers: Panasonic, CATL, and LG Energy Solution.
- Returned upstream raw-material and resource suppliers including Glencore, Lithium Americas, Ioneer, TerraVolta Resources, Standard Lithium, and Equinor.
- Added material context (Nickel, Cobalt, Steel) and aligned it with supplier relationships for a clearer procurement picture.

Why this is useful:
- Combines graph structure (who supplies whom) with semantic context (what risks and materials are involved).
- Produces concise, source-aware answers suitable for risk briefings and analyst handoff.

Current note:
- One legacy index (`entity_embedding_idx`) still reports a dimensionality mismatch (384 vs 3072). Core hybrid answers remain usable via the other label-specific indexes.
