from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase

from graphrag.config import (
    GOOGLE_API_KEY,
    GEMINI_CHAT_MODEL,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    validate_config,
)


def build_graph_retriever() -> dict[str, Any]:
    validate_config()

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )
    graph.refresh_schema()

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template="""
You are an expert Neo4j Cypher query writer for a supply chain knowledge graph.

SCHEMA:
{schema}

NODE LABELS: Company, Material, Country, Facility, RiskEvent, RegulatoryBody,
             Organization, Product, Classification, Regulation, Location, Entity

KEY PROPERTIES ON ALL NODES:
- name (unique, use for matching)
- retrieval_text (rich text description, use in RETURN)
- entity_type (original type from extraction)
- source (source document, e.g. "tesla_impact_2023")

RELATIONSHIP TYPES (use exact casing):
SUPPLIES_TO, HAS_SUPPLIER, SOURCES_FROM, PRODUCED_BY, PRODUCES,
MADE_FROM, CONTAINS, USES, UTILIZES, LOCATED_IN, OPERATES_IN,
OWNS, OWN, OWNES, PART_OF, MEMBER_OF, INCLUDES, COMPRISING,
REGULATED_BY, REGULATES, OVERSEES, SUBJECT_TO,
POSES_RISK, POTENTIAL_RISK, AFFECTED_BY, AFFECTS, RESPONDS_TO,
COLLABORATES_WITH, BUILDS_PARTNERSHIPS_WITH, SUPPORTS, CONTRIBUTES_TO,
DEVELOPS, ESTABLISHES, EVALUATES, IDENTIFIES, MAPS,
RATED_BY, RECOGNIZED_BY, AWARDED_BY, RECEIVED_AWARD_FROM,
CREATES_IMPACT, REVEALS, REDUCES_EMISSIONS, RECYCLES,
ALTERNATIVE_TO, NEEDS, REQUIRES, MENTIONS, MENTIONED_IN,
IS_A, CATALYST_FOR, CHAMPIONS, PURSUING, COORDINATES,
COMMUNICATES_WITH, ENGAGES_IN_DIALOGUE_WITH, RECEIVES_INFORMATION_FROM,
SOURCES_INFORMATION_FROM, ENSURES_MANAGEMENT_OF, IMPLEMENTED_SYSTEM

CYPHER RULES:
1. Always use case-insensitive matching: toLower(n.name) CONTAINS toLower('search term')
2. For ownership queries always use: -[:OWN|OWNES|OWNS]->
3. Always RETURN node.retrieval_text alongside node.name for richer context
4. Always add LIMIT 10 at the end
5. Do not use properties that are not in the schema

Write a Cypher query for this question:
{question}

Return ONLY the Cypher query, no explanation.
""",
    )

    schema_text = getattr(graph, "schema", "") or ""
    return {"llm": llm, "prompt": cypher_prompt, "schema": schema_text}


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _strip_fences(text: str) -> str:
    out = text.strip()
    if out.startswith("```"):
        out = out.strip("`")
        out = out.replace("cypher", "", 1).strip()
    return out


def _is_read_only_cypher(cypher: str) -> bool:
    forbidden = [" create ", " merge ", " delete ", " set ", " remove ", " drop "]
    c = f" {cypher.lower()} "
    return not any(tok in c for tok in forbidden)


def _run_read_only_cypher(cypher: str) -> list[dict[str, Any]]:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(cypher)
        return [dict(r) for r in rows]


def graph_search(question: str) -> dict[str, Any]:
    retriever = build_graph_retriever()
    llm = retriever["llm"]
    prompt = retriever["prompt"]
    schema = retriever["schema"]

    prompt_text = prompt.format(schema=schema, question=question)
    response = llm.invoke(prompt_text)
    cypher = _strip_fences(_extract_text(response))

    if not _is_read_only_cypher(cypher):
        return {
            "answer": "Generated Cypher was non-read-only and was blocked.",
            "cypher": cypher,
            "raw_results": [],
        }

    try:
        raw_context = _run_read_only_cypher(cypher)
    except Exception as exc:  # noqa: BLE001
        return {
            "answer": f"Graph query execution failed: {exc}",
            "cypher": cypher,
            "raw_results": [],
        }

    return {
        "answer": "Graph query executed successfully.",
        "cypher": cypher,
        "raw_results": raw_context,
    }
