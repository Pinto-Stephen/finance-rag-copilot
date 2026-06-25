from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.retrieve import load_index
from src.generate import answer, MODEL

load_dotenv()


SYSTEM_PROMPT = (
    "You are a financial research assistant over US airline 10-K filings. "
    "You have one tool, query_filings, which answers a question scoped to a "
    "company and/or year. For comparison questions, call the tool separately for "
    "each company/year and then synthesize the results. Always preserve the "
    "[TICKER YEAR] citations from the tool outputs in your final answer. If a tool "
    "result says it doesn't know, report that rather than guessing."
)


def build_agent(index):
    """Create a ReAct agent whose query_filings tool runs against `index`."""

    @tool
    def query_filings(question: str, company: str = "", year: int = 0) -> str:
        """Answer a question from the airline 10-K filings.

        Use `company` (ticker: DAL, UAL, AAL, LUV, ALK) and/or `year` (e.g. 2024) to
        scope the search to a single filing. Call this once per (company, year) you
        need; for a comparison, call it multiple times and combine the results.
        """
        text, _ = answer(
            question,
            index,
            company=company or None,   # empty string -> no filter
            year=year or None,         # 0 -> no filter
        )
        return text

    llm = ChatGroq(model=MODEL, temperature=0)
    return create_react_agent(llm, tools=[query_filings], prompt=SYSTEM_PROMPT)


def ask(question: str, index=None, agent=None) -> str:
    """Run the agent on a question and return its final text answer.

    Pass `agent` (built once via build_agent) to reuse it; otherwise an index is
    loaded and an agent is built on the fly for standalone/CLI use.
    """
    if agent is None:
        agent = build_agent(index if index is not None else load_index())
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


if __name__ == "__main__":
    q = "Compare what Delta and United said about fuel hedging in their 2024 filings."
    print(f"Q: {q}\n")
    print(ask(q))
