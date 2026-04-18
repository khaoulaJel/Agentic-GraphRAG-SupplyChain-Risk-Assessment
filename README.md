# Agentic GraphRAG Supply Chain Risk Assessment

Hybrid supply-chain risk QA using a Neo4j knowledge graph, vector retrieval, and a LangGraph agent.

## Final Architecture

1. Data preparation
- `data_extraction/convert_pdfs_to_markdown.py`: PDF/HTML to markdown
- `scripts/extract_entities_gliner.py`: entity extraction
- `scripts/convert_extracted_to_triples.py`: entity output to triples
- `scripts/load_graph_neo4j.py`: load triples into Neo4j

2. Knowledge graph layer
- `schema.cypher` + `scripts/run_schema.py`: constraints/indexes
- `scripts/seed_graph.py`: seed baseline EV supply chain graph
- `graph/`: schema and graph-building utilities

3. GraphRAG retrieval layer (`graphrag/`)
- `embed_and_index.py`: create/update embeddings and vector indexes
- `vector_retriever.py`: vector retrieval
- `graph_retriever.py`: Cypher retrieval
- `hybrid_chain.py`: graph + vector synthesis
- `main.py`: sample hybrid queries
- `compare_rag_vs_graphrag.py`: GraphRAG vs RAG-only vs Graph-only

4. Agent layer (`agent/`)
- `triage.py`: simple vs complex routing
- `router.py`: tool-using ReAct router
- `tools.py`: graph, vector, and Tavily tools
- `graph.py`, `run.py`: LangGraph workflow and chat loop entrypoint

## Setup

```bash
conda env create -f environment.yml
conda activate graphrag-ev
copy .env.example .env
```

Required env vars:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY` (agent news tool)

## Run the System

```bash
# 1) Initialize graph schema
python -m scripts.run_schema

# 2) Seed baseline graph
python -m scripts.seed_graph

# 2.1) (Optional) Extraction and triple loading pipeline
python data_extraction/convert_pdfs_to_markdown.py
python -m scripts.extract_entities_gliner
python -m scripts.convert_extracted_to_triples
python -m scripts.load_graph_neo4j

# 3) Build/refresh embeddings and vector indexes
python -m graphrag.embed_and_index

# 4) Verify embedding/index health
python -m graphrag.verify_embeddings

# 5) Run GraphRAG sample queries
python -m graphrag.main

# 6) Run retrieval-mode comparison
python -m graphrag.compare_rag_vs_graphrag --mode both

# 7) Run agent chat
python -m agent.run
```

## Optional Benchmark

```bash
python -m benchmarks.benchmark6_e2e_ragas --top-k 5 --output-csv benchmark6_results.csv
```

## Minimal Project Map

```text
agent/                 # LangGraph orchestration (triage + router + tools)
graphrag/              # Hybrid retrieval and synthesis pipeline
graph/                 # KG schema/building helpers
data_extraction/       # Source-doc conversion utilities
data/                  # Raw, processed, and markdown corpora
scripts/               # Entry scripts (schema, extraction, loading, seeding)
benchmarks/            # Evaluation scripts
docs/                  # Project documentation (ontology)
```
