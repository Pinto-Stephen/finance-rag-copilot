import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


def run_with_live_status(work, phase_messages):
    """Run a blocking call in a worker thread while the main thread animates a live
    'thinking' indicator — a rotating status message plus an elapsed-time counter — so
    the UI never looks frozen during retrieval/generation. Returns (result, seconds).

    `work` is a zero-arg callable; it runs in the worker and must NOT touch Streamlit
    (answer/agent_ask don't). Every st.* update stays on the main thread, as Streamlit
    requires. The phase messages advance ~every 3s and hold on the last one until done.
    """
    placeholder = st.empty()
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(work)
        while not future.done():
            elapsed = time.time() - start
            msg = phase_messages[min(int(elapsed // 3), len(phase_messages) - 1)]
            placeholder.markdown(f"🧠 **{msg}** &nbsp;·&nbsp; ⏱️ {elapsed:0.1f}s")
            time.sleep(0.2)
        result = future.result()   # re-raises any error from the worker
    placeholder.empty()
    return result, time.time() - start


def scope_label(company, year):
    """Human phrase for the current single-shot scope, used in status messages."""
    if company == "All" and year == "All":
        return "all airline filings"
    if company == "All":
        return f"all airlines, FY{year}"
    if year == "All":
        return f"{company} filings"
    return f"{company} FY{year}"


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
            phases = [
                "Planning which filings to consult…",
                "Querying the filings — may run several scoped searches…",
                "Reading and re-ranking passages from each call…",
                "Synthesizing the results into one cited answer…",
                "Almost there — preserving [TICKER YEAR] citations…",
            ]
            text, elapsed = run_with_live_status(
                lambda: agent_ask(prompt, agent=get_agent()), phases
            )
            nodes = []  # the agent synthesizes; citations are carried inline as [TICKER YEAR]
        else:
            scope = scope_label(company, year)
            phases = [
                f"Searching {scope} for relevant passages…",
                f"Re-ranking the strongest matches for {scope}…",
                "Reading the top passages and drafting a grounded answer…",
                "Writing citations as [TICKER YEAR]…",
                "Almost there — finalizing the answer…",
            ]
            (text, nodes), elapsed = run_with_live_status(
                lambda: answer(
                    prompt,
                    index,
                    company=None if company == "All" else company,
                    year=None if year == "All" else year,
                ),
                phases,
            )
        st.markdown(text)
        st.caption(f"Answered in {elapsed:0.1f}s")
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