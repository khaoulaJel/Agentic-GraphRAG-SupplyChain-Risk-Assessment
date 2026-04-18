from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.state import AgentState
from graphrag.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL


llm = ChatGoogleGenerativeAI(
    model=GEMINI_CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

TRIAGE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a query classifier for a supply chain intelligence system.

Conversation history:
{history}

New query: {query}

Classify the query as exactly one of:
- "simple": if it is a greeting, chitchat, a follow-up/clarification on something
  already in the conversation history, or can be fully answered from the history alone
- "complex": if it requires searching a knowledge graph, retrieving documents,
  or finding recent news about supply chains, companies, materials, risks, or regulations

Reply with ONLY the single word: simple or complex
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


def triage_node(state: AgentState) -> AgentState:
    history_str = _history_to_text(state.get("history", []), last_n=6)

    chain = TRIAGE_PROMPT | llm | StrOutputParser()
    decision = chain.invoke(
        {
            "query": state["query"],
            "history": history_str or "No history yet.",
        }
    ).strip().lower()

    complexity = "simple" if decision == "simple" else "complex"
    print(f"[TRIAGE] -> {complexity}")

    return {**state, "complexity": complexity}
