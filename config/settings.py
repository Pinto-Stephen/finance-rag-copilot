
DEFAULT_DATA_DIR = "C:/Users/steph/PycharmProjects/finance-rag-copilot/data/raw/sec"
QDRANT_PATH = "storage/qdrant"

EMBED_MODEL  = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL    = "llama-3.3-70b-versatile"   # Groq

COLLECTION = "airline_10k"

CHUNK_SIZE     = 512
CHUNK_OVERLAP  = 64
RETRIEVE_TOP_K = 20      # cast a wide net for the reranker
RERANK_TOP_N   = 5       # what actually reaches the LLM