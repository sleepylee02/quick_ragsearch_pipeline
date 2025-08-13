from typing import List, Tuple
import numpy as np


class SimpleVectorStore:
    """A minimal in-memory vector store using cosine similarity."""

    def __init__(self) -> None:
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []

    def add_texts(self, texts: List[str], embeddings: List[List[float]]) -> None:
        for text, emb in zip(texts, embeddings):
            self.texts.append(text)
            self.embeddings.append(np.array(emb))

    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[str, float]]:
        query = np.array(query_embedding)
        sims = []
        for text, emb in zip(self.texts, self.embeddings):
            if np.linalg.norm(emb) == 0 or np.linalg.norm(query) == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(emb, query) / (np.linalg.norm(emb) * np.linalg.norm(query)))
            sims.append((text, similarity))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]
