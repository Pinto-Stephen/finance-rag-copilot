"""
index_build.py — Embed a corpus's chunks with BGE-M3 and build its OWN Qdrant
collection. The three corpora (sec | nasa | rbi) live in SEPARATE collections by
design, so building one NEVER touches the others.

Run from the project root, choosing a corpus explicitly:
    python -m src.index_build --corpus sec
    python -m src.index_build --corpus nasa
    python -m src.index_build --corpus all

Each build WIPES & RECREATES only its target collection, so re-runs don't duplicate.
"""

import argparse

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from config.settings import QDRANT_PATH, EMBED_MODEL, CORPORA
from src.ingest import load_and_chunk_corpus, PDF_BACKEND


def wipe_collections(targets):
    """Empty each target collection before a rebuild, so re-runs never accumulate
    duplicate points. Only the targeted collections are touched; siblings are left as-is.

    Why not delete_collection()? On local Qdrant it is only a *logical* drop — it
    removes the collection from meta.json but leaves its on-disk storage.sqlite behind,
    and recreating the collection reattaches to that leftover storage and resurrects the
    old points (a fresh build then ADDS to the stale set instead of replacing it).
    Physically rmtree-ing the folder isn't reliable either: the build runs in this same
    process and Windows keeps the just-closed storage file locked, so the removal fails.
    Deleting all *points* via the client (a match-all filter) persists cleanly and
    leaves an empty collection for the build to repopulate.
    """
    client = QdrantClient(path=QDRANT_PATH)
    try:
        for corpus in targets:
            collection = CORPORA[corpus]["collection"]
            if client.collection_exists(collection):
                client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=models.Filter()),
                )
    finally:
        client.close()


def build_corpus(corpus, client):
    """Embed a single corpus's nodes into its (already-wiped) Qdrant collection.

    Only this corpus's collection is written — the other corpora's collections in the
    same on-disk Qdrant are left untouched.
    """
    collection = CORPORA[corpus]["collection"]
    docs, nodes = load_and_chunk_corpus(corpus)
    print(f"[{corpus}] {len(docs)} docs -> {len(nodes)} nodes -> collection '{collection}'")

    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    print(f"[{corpus}] done -> '{collection}'")


def main():
    parser = argparse.ArgumentParser(description="Build a corpus's Qdrant collection.")
    parser.add_argument(
        "--corpus",
        required=True,
        choices=["sec", "nasa", "rbi", "all"],
        help="Which corpus to (re)build. 'all' rebuilds every corpus in turn.",
    )
    args = parser.parse_args()

    targets = list(CORPORA) if args.corpus == "all" else [args.corpus]

    # Embedding model + storage path both come from config now (no hardcodes).
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    print(f"Embed model: {EMBED_MODEL} | PDF backend: {PDF_BACKEND} | Qdrant: {QDRANT_PATH}")

    # Wipe the targets first (own client, closed before folder removal), then open a
    # fresh client to build into the now-empty collections.
    wipe_collections(targets)

    client = QdrantClient(path=QDRANT_PATH)
    try:
        for corpus in targets:
            build_corpus(corpus, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
