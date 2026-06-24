from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from src.ingest import load_and_chunk, build_reader
from qdrant_client import QdrantClient
from config.settings import COLLECTION


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

docs, nodes = load_and_chunk()

client = QdrantClient(path="C:/Users/steph/PycharmProjects/finance-rag-copilot/storage/qdrant")
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION)

storage_context = StorageContext.from_defaults(vector_store=vector_store)


print(f"{len(nodes)} nodes to embed")
index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
client.close()