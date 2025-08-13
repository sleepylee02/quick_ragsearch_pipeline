from src.main import LectureProcessor


def run(pdf_path: str):
    processor = LectureProcessor()
    result = processor.process_document(pdf_path)
    print(result)


if __name__ == "__main__":
    import sys
    run(sys.argv[1])
