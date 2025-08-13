from typing import List
import io
from PIL import Image
from openai import OpenAI

from config import OPENAI_API_KEY, VISION_MODEL


class ImageProcessor:
    """Generate descriptions for images using OpenAI vision models."""

    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def describe(self, images: List[Image.Image]) -> List[str]:
        descriptions: List[str] = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            response = self.client.responses.create(
                model=VISION_MODEL,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image": img_bytes},
                ]}]
            )
            descriptions.append(response.output[0].content[0].text)
        return descriptions
