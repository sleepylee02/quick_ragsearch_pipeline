from src.main import LectureProcessor


def ask(question: str):
    processor = LectureProcessor()
    print(processor.ask_question(question))


if __name__ == "__main__":
    import sys
    ask(sys.argv[1])
