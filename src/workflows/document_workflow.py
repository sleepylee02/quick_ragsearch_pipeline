from typing import Dict

from ..processors.pdf_processor import PDFProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.text_processor import TextProcessor
from ..processors.embedder import Embedder
from ..storage.vector_store import SimpleVectorStore


class DocumentWorkflow:
    def __init__(self, store: SimpleVectorStore) -> None:
        self.pdf_processor = PDFProcessor()
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        self.embedder = Embedder()
        self.store = store

    def run(self, pdf_path: str) -> Dict[str, int]:
        text, images = self.pdf_processor.extract(pdf_path)
        image_descriptions = self.image_processor.describe(images) if images else []
        chunks = self.text_processor.process(text)
        combined = chunks + image_descriptions
        if combined:
            embeddings = self.embedder.embed(combined)
            self.store.add_texts(combined, embeddings)
        return {"chunks": len(chunks), "images": len(images)}
