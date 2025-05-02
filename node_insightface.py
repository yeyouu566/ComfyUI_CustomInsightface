import insightface
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
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)  # ComfyUI 호환을 위해 STRING으로 반환
    FUNCTION = "analyze"
    CATEGORY = "face"

    def analyze(self, image):
        # 모델 준비
        if self.app is None:
            self.app = FaceAnalysis(name="buffalo_l")
            self.app.prepare(ctx_id=0)

        # PIL.Image -> np.array 변환
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
        else:
            img = image

        # 얼굴 분석
        faces = self.app.get(img)
        face_count = len(faces)
        return (f"Detected {face_count} faces.",)
