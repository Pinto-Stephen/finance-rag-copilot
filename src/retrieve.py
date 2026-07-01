from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import (MetadataFilter,MetadataFilters,FilterOperator,FilterCondition)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from config.settings import QDRANT_PATH,COLLECTION,EMBED_MODEL,RERANK_MODEL,RETRIEVE_TOP_K,RERANK_TOP_N,CORPORA,DEFAULT_CORPUS

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

_reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

# Lazily-built sparse retrievers, one per corpus. Qdrant persists vectors only (no
# docstore), so to add a BM25 keyword leg we re-chunk each corpus's source docs once
# and index them in memory, keyed by corpus so switching corpora doesn't rebuild.
_bm25 = {}

# All three corpora share one on-disk Qdrant, and local Qdrant allows only one client
# per path per process — so we open a single shared client and point per-corpus indexes
# at their own collection through it (opening a second client would deadlock).
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def load_index(corpus=DEFAULT_CORPUS):
    """Reconnect to a corpus's persisted Qdrant collection (defaults to SEC)."""
    collection = CORPORA[corpus]["collection"]
    vector_store = QdrantVectorStore(client=_get_client(), collection_name=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def _get_bm25(corpus=DEFAULT_CORPUS):
    """Build (once, per corpus) a BM25 retriever over that corpus's re-chunked docs.

    This re-reads and re-chunks the corpus's source docs the first time it's called
    for that corpus (~30s+); afterwards it's cached for the process. BM25 doesn't
    apply metadata filters reliably, so we over-fetch and post-filter in retrieve().
    """
    if corpus not in _bm25:
        # SEC keeps its frozen load_and_chunk() path (byte-for-byte unchanged); other
        # corpora resolve their raw docs through load_and_chunk_corpus.
        if corpus == DEFAULT_CORPUS:
            from src.ingest import load_and_chunk
            _, nodes = load_and_chunk()
        else:
            from src.ingest import load_and_chunk_corpus
            _, nodes = load_and_chunk_corpus(corpus)
        _bm25[corpus] = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=RETRIEVE_TOP_K * 4
        )
    return _bm25[corpus]


def _build_filters(company=None, year=None):
    """Optional metadata scoping"""
    active = []
    if company is not None:
        active.append(MetadataFilter(key="company", value=company, operator=FilterOperator.EQ))
    if year is not None:
        active.append(MetadataFilter(key="year", value=year, operator=FilterOperator.EQ))
    if not active:
        return None
    return MetadataFilters(filters=active, condition=FilterCondition.AND)


def _matches(node, company=None, year=None):
    """Apply the same company/year scope to BM25 hits that the vector store
    enforces natively via MetadataFilters."""
    m = node.metadata
    if company is not None and m.get("company") != company:
        return False
    if year is not None and m.get("year") != year:
        return False
    return True


def _dedup_key(node):
    """Vector and BM25 hits are different node objects (BM25 re-chunks the source,
    so its node ids differ), so we dedupe on content, not id."""
    return node.get_content()[:200].strip()


def _rrf(result_lists, k=60):
    """Reciprocal Rank Fusion: merge ranked lists into one candidate pool,
    deduping by content. Final ordering still comes from the reranker; RRF just
    decides which candidates make the pool."""
    scores, holder = {}, {}
    for ranked in result_lists:
        for rank, nws in enumerate(ranked):
            key = _dedup_key(nws.node)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            holder.setdefault(key, nws)
    return sorted(holder.values(), key=lambda n: scores[_dedup_key(n.node)], reverse=True)


def retrieve(query, index, corpus=DEFAULT_CORPUS, company=None, year=None):
    """Hybrid retrieval (dense + BM25) then cross-encoder rerank.

    Dense vector search and BM25 keyword search each cast a wide net (with the
    company/year scope applied to both); their results are fused via RRF and the
    pooled candidates are reranked down to RERANK_TOP_N. `index` and `corpus` must
    refer to the same corpus so the dense and sparse legs stay aligned.

    Returns a list of NodeWithScore, where node.score is the reranker's
    relevance score (it replaces the original retrieval scores).
    """
    dense = index.as_retriever(
        similarity_top_k=RETRIEVE_TOP_K,
        filters=_build_filters(company, year),
    ).retrieve(query)

    sparse = [
        n for n in _get_bm25(corpus).retrieve(query)
        if _matches(n, company, year)
    ][:RETRIEVE_TOP_K]

    candidates = _rrf([dense, sparse])
    if not candidates:
        return []
    return _reranker.postprocess_nodes(  # best RERANK_TOP_N
        candidates, query_bundle=QueryBundle(query)
    )


if __name__ == "__main__":
    index = load_index()

    q = "What did Delta say about fuel hedging?"
    results = retrieve(q, index, company="DAL")

    print(f"Query: {q}")
    print(f"Returned {len(results)} reranked chunks\n")
    for i, node in enumerate(results, 1):
        m = node.metadata
        print(f"[{i}] {m.get('company')} {m.get('year')}  rerank_score={node.score:.3f}")
        print("   ", node.text[:200].replace("\n", " "), "\n")
