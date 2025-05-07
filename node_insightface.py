from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

class InsightFaceLoader:
    def __init__(self):
        self.app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    @classmethod
    def RETURN_TYPES(cls):
        return ("INSIGHTFACE",)

    FUNCTION = "analyze"
    CATEGORY = "face"

    def analyze(self, image):
        if self.app is None:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0)

        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image

        faces = self.app.get(img)

        embeddings = [
            face.embedding.tolist()
            for face in faces if face.embedding is not None
        ]

        return (embeddings,)

NODE_CLASS_MAPPINGS = {
    "CustomInsightface": InsightFaceLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomInsightface": "InsightFace 모델 분석기"
}
