import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.retrieve import load_index
from src.generate import answer
from src.agent import build_agent, ask as agent_ask
from config.settings import CORPORA, DEFAULT_CORPUS

st.set_page_config(page_title="Research Copilot", page_icon="📚", layout="wide")

TICKERS = ["All", "DAL", "UAL", "AAL", "LUV", "ALK"]
YEARS = ["All", 2021, 2022, 2023, 2024, 2025]

# display label -> corpus key, in registry order (SEC first, so it's the default).
CORPUS_OPTIONS = {cfg["display_name"]: key for key, cfg in CORPORA.items()}


@st.cache_resource(show_spinner="Loading index, embeddings and reranker...")
def get_index(corpus):
    """Load a corpus's index once and reuse across reruns (cached per corpus; all
    corpora share one Qdrant connection under the hood)."""
    return load_index(corpus)


@st.cache_resource(show_spinner="Building the agent...")
def get_agent(corpus):
    """Build the ReAct agent once per corpus, sharing that corpus's cached Qdrant
    index so the single-shot and agent paths never open a second (deadlocking)
    connection."""
    return build_agent(get_index(corpus), corpus=corpus)


st.title("📚 Research Copilot")
st.caption("Answers are grounded in the selected corpus's documents and cited from "
           "them. If the documents don't cover something, the assistant says so.")

with st.sidebar:
    st.header("Corpus")
    corpus_display = st.radio(
        "Corpus",
        list(CORPUS_OPTIONS),
        help="Which document collection to search. Company/year filters apply to the "
             "Airlines 10-K corpus only.",
    )
    corpus = CORPUS_OPTIONS[corpus_display]
    is_sec = corpus == DEFAULT_CORPUS

    st.header("Mode")
    mode = st.radio(
        "Answering mode",
        ["Single-shot", "Agent (multi-step)"],
        help="Single-shot runs one scoped retrieval. Agent can make several scoped "
             "calls (e.g. Delta vs United) and synthesize them.",
    )
    st.header("Scope")
    agent_mode = mode == "Agent (multi-step)"
    scope_disabled = agent_mode or not is_sec
    company = st.selectbox("Company", TICKERS, disabled=scope_disabled)
    year = st.selectbox("Fiscal year", YEARS, disabled=scope_disabled)
    if not is_sec:
        st.caption("Company/year filters apply to the Airlines 10-K corpus only.")
    elif agent_mode:
        st.caption("In Agent mode the assistant chooses the company/year scope per "
                   "tool call, so these filters are ignored.")
    else:
        st.caption("Filters apply to every question. 'All' means no filter.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Filters only meaningful for SEC; force them off for other corpora regardless of the
# (disabled) widget's retained value.
company = company if is_sec else "All"
year = year if is_sec else "All"

index = get_index(corpus)


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


def source_label(node, corpus):
    """Citation label for a retrieved chunk. SEC uses [TICKER YEAR]; other corpora
    use their citation/title, matching how generate.format_context tags them."""
    m = node.metadata
    if corpus == DEFAULT_CORPUS:
        return f"[{m.get('company')} {m.get('year')}]"
    return f"[{m.get('citation') or m.get('title') or m.get('doc_id')}]"


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}**  ·  relevance {s['score']:.3f}")
            st.caption(s["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (including each answer's sources).
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources"))

if prompt := st.chat_input(f"Ask about the {corpus_display} corpus..."):
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
                lambda: agent_ask(prompt, agent=get_agent(corpus)), phases
            )
            nodes = []  # the agent synthesizes; citations are carried inline as [TICKER YEAR]
        else:
            scope = scope_label(company, year) if is_sec else corpus_display
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
                    corpus=corpus,
                    company=None if company == "All" else company,
                    year=None if year == "All" else year,
                ),
                phases,
            )
        st.markdown(text)
        st.caption(f"Answered in {elapsed:0.1f}s")
        sources = [
            {
                "label": source_label(n, corpus),
                "score": float(n.score or 0.0),
                "text": n.text[:400],
            }
            for n in nodes
        ]
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )