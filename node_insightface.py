from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

TYPE_CLASSES = {
    "INSIGHTFACE": list  # ComfyUI가 받아들일 수 있는 타입 지정
}

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

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "analyze"
    CATEGORY = "face"

    def analyze(self, image):
        # 모델 준비
        if self.app is None:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0)

        # 이미지 전처리
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image

        # 얼굴 분석 수행
        faces = self.app.get(img)

        # embedding 추출
        embeddings = [
            face.embedding.tolist()
            for face in faces if face.embedding is not None
        ]

        return (embeddings,)
