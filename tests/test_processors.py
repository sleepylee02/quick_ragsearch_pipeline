from src.processors.pdf_processor import PDFProcessor
from src.processors.text_processor import TextProcessor


def test_pdf_processor_extracts_text():
    processor = PDFProcessor()
    text, images = processor.extract("examples/sample_pdfs/sample.pdf")
    assert "Hello World" in text
    assert images == []


def test_text_processor_chunks():
    tp = TextProcessor()
    chunks = tp.process("a" * 1500)
    assert len(chunks) == 2
