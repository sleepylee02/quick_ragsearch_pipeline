"""Simple Q&A workflow using the Weaviate vector store."""

from typing import Callable, List
import weaviate

from src.storage.vector_store import WeaviateVectorStore


def create_vector_store(
    url: str,
    index_name: str = "Document",
    embedding: Callable[[str], List[float]] | None = None,
) -> WeaviateVectorStore:
    """Initialise a :class:`WeaviateVectorStore` for the workflow."""
    client = weaviate.Client(url)
    return WeaviateVectorStore(client, index_name=index_name, embedding=embedding)


def answer_query(store: WeaviateVectorStore, query: str) -> List[str]:
    """Retrieve relevant documents for ``query`` using the vector store."""
    results = store.similarity_search(query)
    return [r["text"] for r in results]
