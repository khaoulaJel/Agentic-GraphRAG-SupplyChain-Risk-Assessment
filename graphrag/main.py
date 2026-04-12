from __future__ import annotations

from graphrag.hybrid_chain import hybrid_query


if __name__ == "__main__":
    questions = [
        "What risk events are associated with Tesla's supply chain?",
        "Which companies supply materials to Tesla?",
        "Which countries are most exposed to cobalt supply risk in this graph?",
        "What are the key upstream dependencies for LG Energy Solution?",
        "Summarize the main lithium-related risks and affected companies.",
    ]

    for q in questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        print("=" * 60)
        try:
            answer = hybrid_query(q)
            print(f"\nA: {answer}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"\nA: Hybrid query failed: {exc}\n")
