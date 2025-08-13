from .storage.vector_store import SimpleVectorStore
from .workflows.document_workflow import DocumentWorkflow
from .workflows.qa_workflow import QAWorkflow


class LectureProcessor:
    def __init__(self) -> None:
        self.store = SimpleVectorStore()
        self.document_workflow = DocumentWorkflow(self.store)
        self.qa_workflow = QAWorkflow(self.store)

    def process_document(self, pdf_path: str):
        return self.document_workflow.run(pdf_path)

    def ask_question(self, question: str) -> str:
        return self.qa_workflow.ask(question)
