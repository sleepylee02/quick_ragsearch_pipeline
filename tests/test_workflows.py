import os
from unittest.mock import patch
os.environ["OPENAI_API_KEY"] = "test"
from src.main import LectureProcessor


class DummyResponse:
    def __init__(self, text: str):
        self.output = [type('obj', (object,), {'content': [type('c', (object,), {'text': text})()]})()]


def fake_embed(texts):
    return [[float(len(t))] for t in texts]


def fake_response(*args, **kwargs):
    return DummyResponse("answer")


def test_document_and_qa_workflow():
    processor = LectureProcessor()
    with patch('src.processors.embedder.Embedder.embed', side_effect=fake_embed), \
         patch.object(processor.qa_workflow.client.responses, 'create', side_effect=fake_response):
        processor.process_document("examples/sample_pdfs/sample.pdf")
        answer = processor.ask_question("What does the document say?")
        assert answer == "answer"
