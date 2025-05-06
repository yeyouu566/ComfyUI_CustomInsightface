from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np

class InsightFaceLoader:
    def __init__(self):
        self.app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INSIGHTFACE",)
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

        # ✅ ComfyUI에서 처리 가능한 형태로 변환
        face_infos = []
        for face in faces:
            face_infos.append({
                "bbox": face.bbox.tolist(),            # 얼굴 좌표
                "kps": face.kps.tolist(),              # keypoints
                "gender": face.gender,
                "age": face.age,
                "embedding": face.embedding.tolist()   # 벡터
            })

        return (face_infos,)
