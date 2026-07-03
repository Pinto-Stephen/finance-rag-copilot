# Multi-Corpus Research Copilot

A retrieval-augmented (RAG) system that answers questions over **three independent
document corpora**, each in its own Qdrant collection, selectable from the UI:

| Corpus | Collection | What's in it |
|--------|------------|--------------|
| **Airlines 10-K** | `airline_10k` (~6.9k chunks) | SEC 10-K filings of five US airlines — **Delta (DAL)**, **United (UAL)**, **American (AAL)**, **Southwest (LUV)**, **Alaska (ALK)** — FY2021–2025 |
| **NASA Reports** | `nasa_reports` (~3.3k chunks) | NASA technical reports on aircraft fuel efficiency & propulsion (ACEE program → electrified propulsion) |
| **RBI Circulars** | `rbi_circulars` (~4.3k chunks) | Reserve Bank of India master circulars & directions on banking regulation |

It does **hybrid retrieval** (BGE-M3 dense + BM25, RRF fusion) with **cross-encoder
reranking**, **grounded + cited** generation via Groq, an optional **LangGraph agent**
for multi-step queries, a **Streamlit** chat UI, and a **RAGAS** evaluation harness.
A single corpus registry (`config.settings.CORPORA`) threads the chosen corpus through
the entire path — `load_index → retrieve → generate.answer → agent → UI` — defaulting
to the SEC corpus so the original single-corpus behavior is unchanged.

## Architecture

| Stage | File | What it does |
|-------|------|--------------|
| Config | `config/settings.py` | `CORPORA` registry (corpus → data dir, Qdrant collection, display name), `DEFAULT_CORPUS`, model names, chunk sizes, top-k |
| Fetch | `fetch_nasa.py`, `fetch_rbi.py` | Download NASA (NTRS) reports and RBI circulars as PDFs into `data/raw/{nasa,rbi}/` plus a `_<corpus>_metadata.jsonl` sidecar (SEC 10-Ks were fetched separately from EDGAR into `data/raw/sec/{TICKER}/`) |
| Ingest | `src/ingest.py` | HTML/inline-XBRL reader for 10-Ks, PDF reader for NASA/RBI; attaches a **uniform cross-corpus metadata schema** (`corpus/title/citation/source_url/doc_id`; SEC also `company/year`); chunks with `SentenceSplitter`. `load_and_chunk_corpus(corpus)` |
| Index | `src/index_build.py` | `--corpus sec\|nasa\|rbi\|all`: BGE-M3 embeddings → each corpus's **own** Qdrant collection at `storage/qdrant`; wipes only its target collection so re-runs don't duplicate |
| Retrieve | `src/retrieve.py` | `load_index(corpus)` → dense (top-20, optional `company/year` filter) **+ per-corpus BM25** fused via RRF → `bge-reranker-v2-m3` rerank (top-5); one shared Qdrant client across corpora |
| Generate | `src/generate.py` | `answer(question, index, corpus, …)`: **per-corpus system prompt** → Groq `openai/gpt-oss-120b`; returns `(text, source_nodes)` |
| Agent | `src/agent.py` | `build_agent(index, corpus)`: LangGraph `create_react_agent` whose query tool is bound to the selected corpus |
| Eval | `eval/run_eval.py` + `eval/testset.json` | Routes each question to its corpus; RAGAS Faithfulness / ResponseRelevancy / ContextPrecision / ContextRecall via a Groq judge; per-corpus means → `eval/results.csv` |
| UI | `app/streamlit_app.py` | Chat UI with a **corpus selector**, per-corpus intros & example questions, single-shot / agent modes, and corpus-aware source citations |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on *nix)
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
GROQ_API_KEY=...            # required
LANGSMITH_TRACING=true      # optional (agent tracing)
LANGSMITH_API_KEY=...       # optional
LANGSMITH_PROJECT=...       # optional
```

## Run order

Run everything **from the project root**. The three Qdrant collections are pre-built,
so to just query you only need the query/UI steps.

```bash
# 1. (one-time) fetch source docs
python fetch_nasa.py                     # -> data/raw/nasa/ + sidecar
python fetch_rbi.py                      # -> data/raw/rbi/  + sidecar
#    (SEC 10-Ks already under data/raw/sec/{TICKER}/)

# 2. sanity-check ingest/chunking for a corpus
python -m src.ingest nasa

# 3. build a corpus's vector index (re-embeds its chunks; safe to re-run)
python -m src.index_build --corpus all   # or: --corpus sec | nasa | rbi

# 4. query from the CLI (defaults to the SEC corpus)
python -m src.retrieve                   # retrieval only
python -m src.generate                   # full grounded + cited answer

# 5. multi-step agent (e.g. "compare Delta vs United fuel hedging 2024")
python -m src.agent

# 6. the UI (pick the corpus from the sidebar)
streamlit run app/streamlit_app.py

# evaluation (several judge-LLM calls per question)
python -m eval.run_eval                  # full run  -> eval/results.csv
python -m eval.run_eval --rescore --max-workers 1   # re-score cached answers only
```

## Design notes

- **Separate collections, one registry:** each corpus lives in its own Qdrant
  collection (never merged); `CORPORA` maps a corpus key → collection/data-dir/label,
  so adding a corpus is one entry, not a rewrite. Everything downstream takes an
  explicit `corpus` argument defaulting to `sec`.
- **Persistent store:** Qdrant lives on disk at `storage/qdrant`; querying never
  re-embeds. Local Qdrant is single-writer, so all three collections are served through
  **one shared client** to avoid a second (deadlocking) connection.
- **Hybrid retrieval:** dense uses BGE-M3 with native `MetadataFilters` (SEC
  `company/year`); BM25 re-chunks each corpus into memory on first query and is cached
  **per corpus** (Qdrant persists vectors only).
- **Uniform metadata → uniform citations:** all corpora emit the same metadata keys, so
  citations and the UI treat them the same — `[TICKER YEAR]` for SEC, the document's
  `citation`/`title` for NASA/RBI.
- **Per-corpus prompts:** only the human-readable framing changes per corpus (e.g.
  "NASA technical reports", "RBI circulars"); the SEC prompt is byte-for-byte unchanged.
- **Eval on a free-tier judge:** RAGAS judges with Groq. Free-tier **daily token caps**
  make a full 4-metric run over the whole testset hard to complete in one pass, so
  `run_eval.py` supports `--rescore` (score cached answers without re-generating),
  per-metric judge routing, and `--max-workers` tuning; the committed `eval/results.csv`
  is therefore a partial-but-real scorecard.
