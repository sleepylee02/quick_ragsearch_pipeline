from typing import List
from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from ..processors.embedder import Embedder
from typing import Callable, List
import weaviate

from src.storage.vector_store import WeaviateVectorStore


class QAWorkflow:
    def __init__(self, store: SimpleVectorStore) -> None:
        self.store = store
        self.embedder = Embedder()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def ask(self, question: str) -> str:
        q_emb = self.embedder.embed([question])[0]
        results = self.store.similarity_search(q_emb, k=4)
        context = "\n".join(text for text, _ in results)
        prompt = f"Answer the question based on the context below.\nContext:\n{context}\n\nQuestion: {question}"
        response = self.client.responses.create(
            model=LLM_MODEL,
            input=prompt,
            max_output_tokens=500,
        )
        return response.output[0].content[0].text.strip()
      
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
