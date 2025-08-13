import os
from unittest.mock import patch
os.environ["OPENAI_API_KEY"] = "test"
from src.main import LectureProcessor


class DummyLLMResponse:
    def __init__(self, text: str):
        self.content = text


def fake_embed(texts):
    return [[float(len(t))] for t in texts]


def fake_llm(*args, **kwargs):
    return DummyLLMResponse("answer")


def test_document_and_qa_workflow():
    processor = LectureProcessor()
    with patch('src.processors.embedder.Embedder.embed', side_effect=fake_embed), \
         patch('src.workflows.qa_workflow.ChatOpenAI.invoke', side_effect=fake_llm):
        processor.process_document("examples/sample_pdfs/sample.pdf")
        answer = processor.ask_question("What does the document say?")
        assert answer == "answer"
