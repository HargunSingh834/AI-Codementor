"""
ChromaDB setup — initializes persistent client and collection.
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Paths
CHROMA_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "leetcode_problems"

# Embedding function: all-MiniLM-L6-v2 via sentence-transformers
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Persistent client
_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


def get_collection():
    """Get or create the leetcode_problems collection."""
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ef,
        metadata={"hnsw:space": "l2"},
    )


def get_client():
    """Return the ChromaDB client (for health checks etc.)."""
    return _client
