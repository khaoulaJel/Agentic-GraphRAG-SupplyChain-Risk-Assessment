from __future__ import annotations

import os

from langchain_core.tools import tool

from graphrag.hybrid_chain import hybrid_query
from graphrag.vector_retriever import vector_search


@tool
def search_knowledge_graph(query: str) -> str:
    """
    Search the supply chain knowledge graph using hybrid retrieval
    (graph traversal + vector similarity). Use this for questions about
    companies, materials, suppliers, risks, regulations, countries,
    facilities, and relationships between supply chain entities.
    """
    try:
        result = hybrid_query(query)
        return result if result else "No results found in knowledge graph."
    except Exception as e:  # noqa: BLE001
        return f"Knowledge graph search failed: {str(e)}"


@tool
def search_vectors(query: str) -> str:
    """
    Search the vector index for semantically similar supply chain nodes.
    Use this for broad or exploratory questions where exact entity names
    are unknown, or to find related concepts.
    """
    try:
        results = vector_search(query, top_k=5)
        if not results:
            return "No similar nodes found."
        lines: list[str] = []
        for r in results:
            lines.append(
                f"[{r.get('label', '?')}] {r.get('name', '')} "
                f"(score: {r.get('score', 0):.3f}) - {r.get('retrieval_text', '')}"
            )
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"Vector search failed: {str(e)}"


@tool
def search_news(query: str) -> str:
    """
    Search the internet for recent news about the query topic using Tavily.
    Use this for current events, recent disruptions, recent regulatory changes,
    or anything that requires up-to-date information beyond the knowledge graph.
    """
    try:
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        if not tavily_key:
            return "News search failed: TAVILY_API_KEY is not set."

        from langchain_community.tools.tavily_search import TavilySearchResults

        tavily = TavilySearchResults(max_results=3)
        results = tavily.invoke(query)
        if not results:
            return "No news results found."
        lines: list[str] = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            lines.append(f"- {title}: {content[:300]}...")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"News search failed: {str(e)}"
