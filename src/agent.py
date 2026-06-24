"""
agent.py — OPTIONAL agentic layer (Phase 8 stretch).

Wraps the RAG pipeline as a tool and lets a LangGraph ReAct agent decide when,
and how many times, to call it. This turns single-shot Q&A into multi-step work:
"Compare Delta's and United's fuel hedging in 2024" becomes two scoped tool calls
(DAL 2024, UAL 2024) that the agent then synthesizes into one answer.

This is also where LangSmith tracing finally lights up — the agent runs through
LangChain/LangGraph, which LangSmith auto-traces (pure LlamaIndex did not).

Run from project root:
    python -m src.agent
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.retrieve import load_index
from src.generate import answer, MODEL

load_dotenv()
_index = load_index()


@tool
def query_filings(question: str, company: str = "", year: int = 0) -> str:
    """Answer a question from the airline 10-K filings.

    Use `company` (ticker: DAL, UAL, AAL, LUV, ALK) and/or `year` (e.g. 2024) to
    scope the search to a single filing. Call this once per (company, year) you
    need; for a comparison, call it multiple times and combine the results.
    """
    text, _ = answer(
        question,
        _index,
        company=company or None,   # empty string -> no filter
        year=year or None,         # 0 -> no filter
    )
    return text


SYSTEM_PROMPT = (
    "You are a financial research assistant over US airline 10-K filings. "
    "You have one tool, query_filings, which answers a question scoped to a "
    "company and/or year. For comparison questions, call the tool separately for "
    "each company/year and then synthesize the results. Always preserve the "
    "[TICKER YEAR] citations from the tool outputs in your final answer. If a tool "
    "result says it doesn't know, report that rather than guessing."
)

_llm = ChatGroq(model=MODEL, temperature=0)
_agent = create_react_agent(_llm, tools=[query_filings], prompt=SYSTEM_PROMPT)


def ask(question: str) -> str:
    """Run the agent on a question and return its final text answer."""
    result = _agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    q = "Compare what Delta and United said about fuel hedging in their 2024 filings."
    print(f"Q: {q}\n")
    print(ask(q))