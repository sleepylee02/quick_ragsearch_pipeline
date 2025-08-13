from fastapi import APIRouter
from src.main import LectureProcessor

router = APIRouter()
processor = LectureProcessor()


@router.post("/process")
def process(pdf_path: str):
    return processor.process_document(pdf_path)


@router.get("/ask")
def ask(question: str):
    return {"answer": processor.ask_question(question)}
