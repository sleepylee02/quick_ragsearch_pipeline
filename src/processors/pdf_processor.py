from typing import List, Tuple
from PIL import Image
import io
import pdfplumber


class PDFProcessor:
    """Extract text and images from PDF files."""

    def extract(self, pdf_path: str) -> Tuple[str, List[Image.Image]]:
        text_parts: List[str] = []
        images: List[Image.Image] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
                for img in page.images:
                    try:
                        base_image = pdf.extract_image(img["object_id"])
                        image_bytes = base_image["image"]
                        images.append(Image.open(io.BytesIO(image_bytes)))
                    except Exception:
                        continue
        return "\n".join(text_parts), images
