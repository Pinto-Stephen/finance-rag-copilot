import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.retrieve import load_index
from src.generate import answer
from src.agent import build_agent, ask as agent_ask

st.set_page_config(page_title="Airline 10-K Copilot", page_icon="✈️", layout="wide")

TICKERS = ["All", "DAL", "UAL", "AAL", "LUV", "ALK"]
YEARS = ["All", 2021, 2022, 2023, 2024, 2025]


@st.cache_resource(show_spinner="Loading index, embeddings and reranker...")
def get_index():
    """Load the index once and reuse across reruns (one Qdrant connection)."""
    return load_index()


@st.cache_resource(show_spinner="Building the agent...")
def get_agent():
    """Build the ReAct agent once, sharing the single cached Qdrant index so the
    single-shot and agent paths never open a second (deadlocking) connection."""
    return build_agent(get_index())


index = get_index()

st.title("✈️ Airline 10-K Research Copilot")
st.caption("Answers are grounded in the filings and cited as [TICKER YEAR]. "
           "If the filings don't cover something, the assistant says so.")

with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Answering mode",
        ["Single-shot", "Agent (multi-step)"],
        help="Single-shot runs one scoped retrieval. Agent can make several scoped "
             "calls (e.g. Delta vs United) and synthesize them.",
    )
    st.header("Scope")
    agent_mode = mode == "Agent (multi-step)"
    company = st.selectbox("Company", TICKERS, disabled=agent_mode)
    year = st.selectbox("Fiscal year", YEARS, disabled=agent_mode)
    if agent_mode:
        st.caption("In Agent mode the assistant chooses the company/year scope per "
                   "tool call, so these filters are ignored.")
    else:
        st.caption("Filters apply to every question. 'All' means no filter.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['company']} {s['year']}]**  ·  relevance {s['score']:.3f}")
            st.caption(s["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (including each answer's sources).
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if prompt := st.chat_input("Ask about the airline 10-Ks..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if agent_mode:
            with st.spinner("Agent is researching (may make several scoped calls)..."):
                text = agent_ask(prompt, agent=get_agent())
            nodes = []  # the agent synthesizes; citations are carried inline as [TICKER YEAR]
        else:
            with st.spinner("Retrieving and generating..."):
                text, nodes = answer(
                    prompt,
                    index,
                    company=None if company == "All" else company,
                    year=None if year == "All" else year,
                )
        st.markdown(text)
        sources = [
            {
                "company": n.metadata.get("company"),
                "year": n.metadata.get("year"),
                "score": float(n.score or 0.0),
                "text": n.text[:400],
            }
            for n in nodes
        ]
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )