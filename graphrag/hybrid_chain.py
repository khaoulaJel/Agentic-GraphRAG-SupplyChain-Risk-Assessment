from __future__ import annotations

import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from graphrag.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL, validate_config
from graphrag.graph_retriever import graph_search
from graphrag.vector_retriever import vector_search


validate_config()

llm = ChatGoogleGenerativeAI(
    model=GEMINI_CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

HYBRID_PROMPT = ChatPromptTemplate.from_template(
    """
You are a supply chain risk and visibility expert.
Answer the question below using BOTH contexts provided.

--- GRAPH CONTEXT (structured relationships from the knowledge graph) ---
{graph_context}

--- VECTOR CONTEXT (semantically similar nodes found by embedding search) ---
{vector_context}

--- QUESTION ---
{question}

Instructions:
- Prioritize graph context for relationship/structural questions (who supplies whom, owned by, etc.)
- Prioritize vector context for risk/event/descriptive questions
- If both agree, state that confidently
- If they conflict, mention both perspectives
- Cite the source document when available (from the source field)
- If neither context answers the question, say so clearly

ANSWER:
"""
)


def format_vector_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No semantically similar nodes found."
    lines: list[str] = []
    for r in results:
        lines.append(
            f"- [{r.get('label', 'Unknown')}] {r.get('name', '')} "
            f"(score: {r.get('score', 0):.3f}, source: {r.get('source', 'unknown')})\n"
            f"  {r.get('retrieval_text', '')}"
        )
    return "\n".join(lines)


def format_graph_results(result: dict[str, Any]) -> str:
    if not result.get("raw_results"):
        return result.get("answer", "No graph results found.")
    lines = [f"Query used: {result.get('cypher', '')}", ""]
    for row in result["raw_results"][:10]:
        lines.append(str(row))
    return "\n".join(lines)


def hybrid_query(question: str, top_k: int = 5) -> str:
    print(f"\n[HYBRID QUERY] {question}")

    print("[1/3] Running vector search...")
    vector_results = vector_search(question, top_k=top_k)
    print(f"[vector] hits used for synthesis: {len(vector_results)}")

    print("[2/3] Running graph search...")
    graph_result = graph_search(question)

    vector_context = format_vector_results(vector_results)
    graph_context = format_graph_results(graph_result)

    print("[3/3] Synthesizing with Gemini...")
    chain = HYBRID_PROMPT | llm | StrOutputParser()
    invoke_config: dict[str, Any] = {
        "run_name": "graphrag_hybrid_synthesis",
        "tags": ["graphrag", "hybrid", "synthesis"],
    }
    # LangSmith traces are enabled by env vars; this config adds readable run metadata.
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}:
        print("[trace] LangSmith tracing is enabled for synthesis run")

    answer = chain.invoke(
        {
            "question": question,
            "graph_context": graph_context,
            "vector_context": vector_context,
        }
        ,
        config=invoke_config,
    )
    return answer
