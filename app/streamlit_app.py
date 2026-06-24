import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.retrieve import load_index
from src.generate import answer

st.set_page_config(page_title="Airline 10-K Copilot", page_icon="✈️", layout="wide")

TICKERS = ["All", "DAL", "UAL", "AAL", "LUV", "ALK"]
YEARS = ["All", 2021, 2022, 2023, 2024, 2025]


@st.cache_resource(show_spinner="Loading index, embeddings and reranker...")
def get_index():
    """Load the index once and reuse across reruns (one Qdrant connection)."""
    return load_index()


index = get_index()

st.title("✈️ Airline 10-K Research Copilot")
st.caption("Answers are grounded in the filings and cited as [TICKER YEAR]. "
           "If the filings don't cover something, the assistant says so.")

with st.sidebar:
    st.header("Scope")
    company = st.selectbox("Company", TICKERS)
    year = st.selectbox("Fiscal year", YEARS)
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