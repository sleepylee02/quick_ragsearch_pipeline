import weaviate
from typing import Any, Callable, Dict, List, Optional


class WeaviateVectorStore:
    """Minimal wrapper around a Weaviate collection.

    The store supports adding plain text documents with optional metadata and
    hybrid similarity search combining BM25 and semantic vectors.
    """

    def __init__(
        self,
        client: weaviate.Client,
        index_name: str = "Document",
        text_key: str = "text",
        embedding: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.text_key = text_key
        self.embedding = embedding
        self._ensure_schema()

    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Create the schema for the index if it does not already exist."""
        schema = {
            "class": self.index_name,
            "vectorizer": "none",
            "properties": [
                {"name": self.text_key, "dataType": ["text"]},
            ],
        }
        if not self.client.schema.exists(self.index_name):
            self.client.schema.create_class(schema)

    # ------------------------------------------------------------------
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add a batch of texts with optional metadata to the store."""
        metadatas = metadatas or [{} for _ in texts]
        with self.client.batch as batch:
            batch.batch_size = 100
            for text, metadata in zip(texts, metadatas):
                vector = self.embedding(text) if self.embedding else None
                obj = {self.text_key: text, **metadata}
                batch.add_data_object(obj, self.index_name, vector=vector)

    # ------------------------------------------------------------------
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Return the top ``k`` most similar documents for ``query``.

        The search is "hybrid" -- it combines BM25 keyword search with
        semantic vector search.  The ``alpha`` parameter controls the weighting
        between the two methods (0 = only keyword, 1 = only vector).
        """
        vector = self.embedding(query) if self.embedding else None
        result = (
            self.client.query
            .get(self.index_name, [self.text_key])
            .with_hybrid(query, vector=vector, alpha=alpha)
            .with_limit(k)
            .do()
        )
        hits = result.get("data", {}).get("Get", {}).get(self.index_name, [])
        # Format results as a list of documents with text and metadata
        documents: List[Dict[str, Any]] = []
        for hit in hits:
            doc = {"text": hit.get(self.text_key, "")}
            for key, value in hit.items():
                if key != self.text_key:
                    doc[key] = value
            documents.append(doc)
        return documents
