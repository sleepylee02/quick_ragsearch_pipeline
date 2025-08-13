from src.main import LectureProcessor

processor = LectureProcessor()
result = processor.process_document("examples/examples_pdfs/sample.pdf")
print(result)
print(processor.ask_question("What is the topic?"))
