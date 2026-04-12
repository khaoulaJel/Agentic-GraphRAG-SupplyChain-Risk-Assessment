from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The original user query
    query: str
    # Full conversation history (list of {"role": "user"/"assistant", "content": "..."})
    history: Annotated[list, add_messages]
    # Triage decision: "simple" or "complex"
    complexity: str
    # Retrieved context from tools (raw string, concatenated)
    retrieved_context: str
    # Final answer to return to the user
    final_answer: str
