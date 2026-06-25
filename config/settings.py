from pathlib import Path

# Resolve everything relative to the project root (this file lives in config/),
# so the app works no matter which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "raw" / "sec")
QDRANT_PATH      = str(PROJECT_ROOT / "storage" / "qdrant")

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL    = "openai/gpt-oss-120b"        # Groq; single source of truth

COLLECTION = "airline_10k"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
RETRIEVE_TOP_K = 20      # cast a wide net for the reranker
RERANK_TOP_N   = 5       # what actually reaches the LLM
