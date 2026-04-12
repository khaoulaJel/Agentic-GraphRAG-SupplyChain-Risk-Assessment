from __future__ import annotations

from agent.graph import agent_graph


def run_agent(query: str, history: list[dict]) -> dict:
    """
    Run one turn of the agent.

    Args:
        query: the user's current message
        history: list of {"role": "user"/"assistant", "content": "..."} dicts

    Returns:
        dict with keys: final_answer, complexity, retrieved_context
    """
    initial_state = {
        "query": query,
        "history": history,
        "complexity": "",
        "retrieved_context": "",
        "final_answer": "",
    }

    result = agent_graph.invoke(initial_state)

    return {
        "final_answer": result["final_answer"],
        "complexity": result["complexity"],
        "retrieved_context": result.get("retrieved_context", ""),
    }


def chat_loop() -> None:
    """Simple terminal chat loop for testing."""
    print("Supply Chain Intelligence Agent")
    print("Type 'quit' to exit\n")

    history: list[dict] = []

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        history.append({"role": "user", "content": query})

        result = run_agent(query, history)
        answer = result["final_answer"]

        history.append({"role": "assistant", "content": answer})

        print(f"\nAgent [{result['complexity']}]: {answer}\n")


if __name__ == "__main__":
    chat_loop()
