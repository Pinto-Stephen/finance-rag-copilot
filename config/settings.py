from pathlib import Path

# Resolve everything relative to the project root (this file lives in config/),
# so the app works no matter which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "raw" / "sec")
NASA_DATA_DIR    = str(PROJECT_ROOT / "data" / "raw" / "nasa")
RBI_DATA_DIR     = str(PROJECT_ROOT / "data" / "raw" / "rbi")
QDRANT_PATH      = str(PROJECT_ROOT / "storage" / "qdrant")

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL    = "openai/gpt-oss-120b"        # Groq; single source of truth

COLLECTION = "airline_10k"        # kept for back-compat (== CORPORA["sec"]["collection"])

# Each corpus lives in its own data dir and its own Qdrant collection (locked design
# decision: separate collections, never merged). index_build.py and the ingest layer
# resolve everything through this registry so a new corpus is one entry, not a rewrite.
# display_name is the human label the UI shows in its corpus selector.
CORPORA = {
    "sec":  {"data_dir": DEFAULT_DATA_DIR, "collection": COLLECTION,      "display_name": "Airlines 10-K"},
    "nasa": {"data_dir": NASA_DATA_DIR,    "collection": "nasa_reports",  "display_name": "NASA Reports"},
    "rbi":  {"data_dir": RBI_DATA_DIR,     "collection": "rbi_circulars", "display_name": "RBI Circulars"},
}

# The corpus every query-path function defaults to, so callers that pass no corpus get
# the original SEC behavior unchanged (backward compatibility).
DEFAULT_CORPUS = "sec"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
RETRIEVE_TOP_K = 20      # cast a wide net for the reranker
RERANK_TOP_N   = 5       # what actually reaches the LLM
