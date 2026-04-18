from __future__ import annotations

import argparse
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from graphrag.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL, validate_config
from graphrag.graph_retriever import graph_search
from graphrag.hybrid_chain import HYBRID_PROMPT, format_graph_results, format_vector_results
from graphrag.vector_retriever import vector_search


DEFAULT_QUESTIONS = [
    "Which companies operate in regions with similar supply chain risk profiles to Tesla's cobalt suppliers?",
    "Which companies directly supply lithium to Tesla?",
    "What risk events are currently affecting lithium and cobalt mining operations?",
]

QUERY_LABELS = [
    "Graph-only",
    "RAG-only",
    "GraphRAG hybrid",
]


def _build_llm() -> ChatGoogleGenerativeAI:
    validate_config()
    return ChatGoogleGenerativeAI(
        model=GEMINI_CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


def retrieve(question: str, top_k: int) -> dict[str, Any]:
    vector_results = vector_search(question, top_k=top_k)
    vector_context = format_vector_results(vector_results)
    graph_result = graph_search(question)
    graph_context = format_graph_results(graph_result)
    cypher = graph_result.get("cypher", "")

    return {
        "vector_results": vector_results,
        "vector_context": vector_context,
        "graph_context": graph_context,
        "cypher": cypher,
    }


def synthesize(
    question: str,
    *,
    mode: str,
    vector_context: str,
    graph_context: str,
    llm: ChatGoogleGenerativeAI,
) -> dict[str, Any]:
    t0 = time.perf_counter()

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
        "mode": mode,
        "elapsed_sec": elapsed,
        "answer": answer,
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"\n[{result['mode']}]")
    print(result["answer"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GraphRAG (graph+vector) with RAG-only (vector-only)."
    )
    parser.add_argument(
        "--mode",
        choices=["both", "graphrag", "rag", "graph"],
        default="both",
        help="Comparison mode: both, graphrag only, rag-only, or graph-only.",
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
    query_labels = QUERY_LABELS if not args.questions else ["Custom"] * len(questions)

    for i, (q, label) in enumerate(zip(questions, query_labels), 1):
        print("\n" + "=" * 80)
        print(f"Query {i} of {len(questions)} - favors: {label}")
        print(f"Q: {q}")
        print("=" * 80)

        retrieved = retrieve(q, top_k=args.top_k)

        if args.mode in {"both", "graphrag"}:
            print_result(
                synthesize(
                    q,
                    mode="GraphRAG",
                    vector_context=retrieved["vector_context"],
                    graph_context=retrieved["graph_context"],
                    llm=llm,
                )
            )

        if args.mode in {"both", "rag"}:
            print_result(
                synthesize(
                    q,
                    mode="RAG-only",
                    vector_context=retrieved["vector_context"],
                    graph_context="Graph retrieval disabled for this run (RAG-only baseline).",
                    llm=llm,
                )
            )

        if args.mode in {"both", "graph"}:
            print_result(
                synthesize(
                    q,
                    mode="Graph-only",
                    vector_context="Vector retrieval disabled for this run (Graph-only mode).",
                    graph_context=retrieved["graph_context"],
                    llm=llm,
                )
            )


if __name__ == "__main__":
    main()
