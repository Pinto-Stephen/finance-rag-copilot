from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import (MetadataFilter,MetadataFilters,FilterOperator,FilterCondition)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from config.settings import QDRANT_PATH,COLLECTION,EMBED_MODEL,RERANK_MODEL,RETRIEVE_TOP_K,RERANK_TOP_N

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

_reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)


def load_index():
    """Reconnect to the persisted Qdrant index"""
    client = QdrantClient(path=QDRANT_PATH)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    return VectorStoreIndex.from_vector_store(vector_store)


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


def retrieve(query, index, company=None, year=None):
    """Vector search (top_k) then cross-encoder rerank (top_n).

    Returns a list of NodeWithScore, where node.score is the reranker's
    relevance score (it replaces the original vector similarity score).
    """
    retriever = index.as_retriever(
        similarity_top_k=RETRIEVE_TOP_K,
        filters=_build_filters(company, year),
    )
    candidates = retriever.retrieve(query)  # ~20 nodes
    return _reranker.postprocess_nodes(  # best 5
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
