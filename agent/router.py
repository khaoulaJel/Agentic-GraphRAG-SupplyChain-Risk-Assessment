from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from agent.state import AgentState
from agent.tools import search_knowledge_graph, search_news, search_vectors
from graphrag.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL


TOOLS = [search_knowledge_graph, search_vectors, search_news]

llm = ChatGoogleGenerativeAI(
    model=GEMINI_CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

react_agent = create_react_agent(llm, TOOLS)

SYSTEM_PROMPT = """You are a supply chain intelligence assistant with access to:
1. search_knowledge_graph - for structured data: companies, suppliers, materials, risks, regulations
2. search_vectors - for semantic/exploratory search across supply chain nodes
3. search_news - for recent news and current events

Use the most relevant tool(s) for the query.
For supply chain relationship questions, prefer search_knowledge_graph.
For recent events or disruptions, use search_news.
For broad exploratory questions, use search_vectors.
You may call multiple tools if needed. Summarize what you found."""

SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """
You are a supply chain expert assistant.

Conversation history:
{history}

User question: {query}

Retrieved context:
{context}

Answer the question using the retrieved context above.
Be specific, cite company names, materials, or risk events when relevant.
If the context does not answer the question, say so clearly.

Answer:
"""
)


def _message_to_pair(msg: object) -> tuple[str, str]:
    if isinstance(msg, dict):
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        return role, content

    role = str(getattr(msg, "type", getattr(msg, "role", "user")))
    content_obj = getattr(msg, "content", "")
    if isinstance(content_obj, list):
        content = " ".join(str(part) for part in content_obj)
    else:
        content = str(content_obj)

    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    return role_map.get(role, role), content


def _history_to_text(history: list[object], last_n: int = 6) -> str:
    pairs = [_message_to_pair(m) for m in history[-last_n:]]
    return "\n".join(f"{role}: {content}" for role, content in pairs)


def _collect_tool_outputs(agent_result: dict) -> list[str]:
    tool_names = {t.name for t in TOOLS}
    outputs: list[str] = []
    for msg in agent_result.get("messages", []):
        msg_type = getattr(msg, "type", None)
        msg_name = getattr(msg, "name", None)
        if msg_type == "tool" and getattr(msg, "content", None):
            outputs.append(str(msg.content))
        elif msg_name in tool_names and getattr(msg, "content", None):
            outputs.append(str(msg.content))
    return outputs


def router_node(state: AgentState) -> AgentState:
    print("[ROUTER] Running rerouting agent...")

    agent_result = react_agent.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["query"]},
            ]
        }
    )

    tool_outputs = _collect_tool_outputs(agent_result)
    retrieved_context = "\n\n---\n\n".join(tool_outputs) if tool_outputs else "No context retrieved."
    print(f"[ROUTER] Retrieved {len(tool_outputs)} tool result(s)")

    history_str = _history_to_text(state.get("history", []), last_n=6)

    synth_chain = (
        SYNTHESIS_PROMPT
        | ChatGoogleGenerativeAI(
            model=GEMINI_CHAT_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2,
        )
        | StrOutputParser()
    )

    final_answer = synth_chain.invoke(
        {
            "query": state["query"],
            "history": history_str or "No history yet.",
            "context": retrieved_context,
        }
    )

    return {**state, "retrieved_context": retrieved_context, "final_answer": final_answer}


def simple_answer_node(state: AgentState) -> AgentState:
    """Answer directly from conversation history with no retrieval."""
    print("[SIMPLE] Answering from memory...")

    history_str = _history_to_text(state.get("history", []), last_n=6)

    prompt = ChatPromptTemplate.from_template(
        """
You are a supply chain assistant. Answer this query using only the conversation history.

Conversation history:
{history}

Query: {query}

Answer:
"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "query": state["query"],
            "history": history_str or "No history yet.",
        }
    )

    return {**state, "final_answer": answer}
