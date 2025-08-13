from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile

router = APIRouter()

@router.post("/process")
async def process_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are supported.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file") from exc

    # TODO: Replace with actual processing logic using temp_path
    return {"filename": filename, "temp_path": temp_path}
