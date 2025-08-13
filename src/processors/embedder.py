from typing import List
from openai import OpenAI

from config import OPENAI_API_KEY, EMBEDDING_MODEL


class Embedder:
    """Generate embeddings using OpenAI."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [data.embedding for data in response.data]
