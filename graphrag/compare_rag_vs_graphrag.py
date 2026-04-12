from __future__ import annotations

import argparse
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from graphrag.config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, validate_config
from graphrag.graph_retriever import graph_search
from graphrag.hybrid_chain import HYBRID_PROMPT, format_graph_results, format_vector_results
from graphrag.vector_retriever import vector_search


DEFAULT_QUESTIONS = [
    "What risk events are associated with Tesla's supply chain?",
    "Which countries are most exposed to cobalt supply risk in this graph?",
]


def _build_llm() -> ChatGoogleGenerativeAI:
    validate_config()
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def run_query(question: str, *, use_graph: bool, top_k: int, llm: ChatGoogleGenerativeAI) -> dict[str, Any]:
    t0 = time.perf_counter()

    vector_results = vector_search(question, top_k=top_k)
    vector_context = format_vector_results(vector_results)

    if use_graph:
        graph_result = graph_search(question)
        graph_context = format_graph_results(graph_result)
        cypher = graph_result.get("cypher", "")
    else:
        graph_context = "Graph retrieval disabled for this run (RAG-only baseline)."
        cypher = ""

    chain = HYBRID_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": question,
            "graph_context": graph_context,
            "vector_context": vector_context,
        },
        config={
            "run_name": "compare_rag_vs_graphrag",
            "tags": ["comparison", "graphrag", "rag"],
        },
    )

    elapsed = time.perf_counter() - t0
    return {
        "question": question,
        "mode": "GraphRAG" if use_graph else "RAG-only",
        "elapsed_sec": elapsed,
        "vector_hits": len(vector_results),
        "cypher": cypher,
        "answer": answer,
    }


def print_result(result: dict[str, Any]) -> None:
    print("\n" + "-" * 80)
    print(f"Mode: {result['mode']}")
    print(f"Time: {result['elapsed_sec']:.2f}s | Vector hits: {result['vector_hits']}")
    if result.get("cypher"):
        print(f"Cypher: {result['cypher']}")
    print("Answer:")
    print(result["answer"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GraphRAG (graph+vector) with RAG-only (vector-only)."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "graphrag", "rag"],
        default="both",
        help="Comparison mode: both, graphrag only, or rag-only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k vector results used for synthesis.",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Custom question. Repeat --question for multiple inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = _build_llm()
    questions = args.questions or DEFAULT_QUESTIONS

    for q in questions:
        print("\n" + "=" * 80)
        print(f"Question: {q}")
        print("=" * 80)

        if args.mode in {"both", "graphrag"}:
            graph_result = run_query(q, use_graph=True, top_k=args.top_k, llm=llm)
            print_result(graph_result)

        if args.mode in {"both", "rag"}:
            rag_result = run_query(q, use_graph=False, top_k=args.top_k, llm=llm)
            print_result(rag_result)


if __name__ == "__main__":
    main()
