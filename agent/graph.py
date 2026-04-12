from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.router import router_node, simple_answer_node
from agent.state import AgentState
from agent.triage import triage_node


def route_after_triage(state: AgentState) -> str:
    """Conditional edge: route based on triage decision."""
    return state["complexity"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", triage_node)
    graph.add_node("simple_answer", simple_answer_node)
    graph.add_node("router", router_node)

    graph.set_entry_point("triage")

    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "simple": "simple_answer",
            "complex": "router",
        },
    )

    graph.add_edge("simple_answer", END)
    graph.add_edge("router", END)

    return graph.compile()


agent_graph = build_graph()
